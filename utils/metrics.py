import json
import os
import gzip
import sys
import subprocess
from mxeval.evaluation import evaluate_functional_correctness
from pathlib import Path


def read_problems(eval_file):
    return {
        str(task["task_id"]): task for task in (
            stream_jsonl(eval_file)
        )
    }

def stream_jsonl(filename):
    if isinstance(filename, str):
        if filename.endswith(".gz"):
            with open(filename, "rb") as gzfp:
                with gzip.open(gzfp, "rt") as fp:
                    for line in fp:
                        if any(not x.isspace() for x in line):
                            yield json.loads(line)
        else:
            with open(filename, "r") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        # Here we expect the filename input to be actually HF DataSet.
        for problem_data_dict in filename:
            yield problem_data_dict

def run_passatk_eval(
    problem_file, # actual problems, not problem file
    language,
    output_dir,
    num_samples_per_example,
    override_previous_results,
):
    print(f"started passatk func")
    # aggregate all generations in a jsonl file
    output_samples_file = os.path.join(output_dir, "samples.jsonl")
    file_exists = os.path.isfile(output_samples_file)
    if isinstance(problem_file, str):
        problems = read_problems(problem_file)
    elif isinstance(problem_file, dict):
        problems = problem_file
    else:
        raise ValueError(f"Unkown type for problem_file {type(problem_file)=}, needs to be path to problem file or dict")
    if file_exists:
        samples = read_problems(output_samples_file)
    else:
        samples = []

    if override_previous_results or not (file_exists and len(problems) == len(samples)):
        with open(output_samples_file, "w") as fw:
            for task_id in problems:
                task_idx = int(task_id.split("/")[1])
                for completion_idx in range(num_samples_per_example):
                    _fname = os.path.join(
                        output_dir,
                        "output",
                        f"taskid-{task_idx}-gen{completion_idx}.json",
                    )
                    prediction = json.load(open(_fname, "r", encoding="utf8"))
                    fw.write(json.dumps(prediction) + "\n")

    # evaluate with pass@k metrics
    local_evaluate_functional_correctness(
        output_samples_file, problem_file
    )


def local_evaluate_functional_correctness(
    output_samples_file, problem_file, language="python", aggregate_file=None
):
    """
    Use subprocess to execute so that the num processes are not bottlenecked
    by pytorch. If we call evaluate_functional_correctness module directly,
    the number of processes can be limited due to not mamy workers being available
    which results in very slow execution.
    """
    print(f"Evaluating from {output_samples_file}")
    results = evaluate_functional_correctness(
        sample_file=output_samples_file,
        k=[1,2,5,10,100,1000],
        # use default for the following
        # n_workers=n_workers,
        # timeout=timeout,
        problem_file=problem_file,
    )

    p_at_k_file = Path(output_samples_file).with_suffix(".passatk.json")
    with open(p_at_k_file, "w") as f:
        f.write(str(results))
    print(results)

    if aggregate_file is not None:
        s = open(p_at_k_file, "r").read()
        aggregate_file.write(s + "\n")