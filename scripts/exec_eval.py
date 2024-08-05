import os

### Config ###
debug = False

# sampling: add --do_sample
num_samples_per_example = 1 # 0: greedy, 1: sampling 1
# num_samples_per_example = 5 # 0: greedy, 1: sampling 1

# for python all cases, we remove --use_stopping_criteria
# should be fine to remove for other languages

outdir = "experiments_v2_pyfix" # balance means fix the truncation balance version
# outdir = "experiments_v2_pyfix_mp1" # min_position=-1 for ablation study, default is -3

fewshot = False
# --fewshot_source ./fewshot_prompting/python_fewshot_v1.py | to have few shot examples attached in the prompt

# model = "starcoder"
# model = "llama"
model = "code_llama"

override = False
#########

cmd_tmp = """
    {} evaluate_model.py \
	--model_name_or_path {} \
	--temperature 0.4 --do_sample \
    --model_context_length 8192 \
    --max_context_length 2048 \
	--max_generation_length 256 \
    --num_samples_per_example {} \
    --batch_size 1 \
    --test_file {} \
    --programming_lang {} \
    --use_custom_generate 1 \
    --use_token_align {} \
	--output_dir {} \
	--bf16 \
"""

# model path
if model == "starcoder":
    model_path = "/mnt/efs/people/skgouda/repos/external/starcoder/starcoder_15b_bf16/"
elif model == "llama":
    model_path = "/mnt/efs/people/gsujan/Llama/hf_models/7B"
elif model == "code_llama":
    # https://huggingface.co/codellama/CodeLlama-7b-hf
    model_path = "codellama/CodeLlama-7b-hf"
else:
    exit(f"{model} not defined!")

fewshot_cmd = " --fewshot_source ./fewshot_prompting/{}_fewshot_v1.py"
if fewshot:
    outdir += "_fs"
if num_samples_per_example > 1:
    outdir += f"_n{num_samples_per_example}"

DEBUG_PYTHON = "python"
RUN_PYTHON = "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.run --nproc_per_node 8 --master_port 20000"
if debug: outdir += "_test"

lang_map = {"mbpp": "python", "mbjp": "java", "mbjsp": "javascript"}

##### customize the datasets and tasks to run #####
# datasets = ["mbpp"]
datasets = ["mbpp", "mbjp", "mbjsp"]
# tasks = ["test"]
# tasks = ["partial"]
# tasks = ["word_full", "word_sub", "punc_sub", "punc_full", "space_sub", "space_full", "prefixind_sub", "prefixind_full", "prefixsep_sub", "prefixsep_full"]
tasks = ["nominal", "partial", "word_full", "word_sub", "punc_sub", "punc_full", "space_sub", "space_full", "prefixind_sub", "prefixind_full", "prefixsep_sub", "prefixsep_full"]

data_folder = "./datasets/perturbed_mbxp_v2"
out_folder = "./"

# use_constrained_geneneration = 1 # 1 to enable, 0 to disable
for use_constrained_geneneration in [1, 0]:
    for dataset in datasets:
        # broken_word: split any words; broken_punc: only split at punctuation
        path_map = {"nominal": f"{data_folder}/{dataset}/nominal/{dataset}_release_v2.jsonl",
                "partial": f"{data_folder}/{dataset}/partial/{dataset}.jsonl",
                "test": f"{data_folder}/{dataset}/test/{dataset}.jsonl",

                "word_sub": f"{data_folder}/{dataset}/word/{dataset}_word_sub_filter.jsonl",
                "word_full": f"{data_folder}/{dataset}/word/{dataset}_word_full_filter.jsonl",

                "punc_sub": f"{data_folder}/{dataset}/punc/{dataset}_punc_sub_filter.jsonl",
                "punc_full": f"{data_folder}/{dataset}/punc/{dataset}_punc_full_filter.jsonl",

                "space_sub": f"{data_folder}/{dataset}/space/{dataset}_space_sub_filter.jsonl",
                "space_full": f"{data_folder}/{dataset}/space/{dataset}_space_full_filter.jsonl",

                "prefixind_sub": f"{data_folder}/{dataset}/prefix-indent/{dataset}_prefix-indent_sub_filter.jsonl",
                "prefixind_full": f"{data_folder}/{dataset}/prefix-indent/{dataset}_prefix-indent_full_filter.jsonl",

                "prefixsep_sub": f"{data_folder}/{dataset}/prefix-sep/{dataset}_prefix-sep_sub_filter.jsonl",
                "prefixsep_full": f"{data_folder}/{dataset}/prefix-sep/{dataset}_prefix-sep_full_filter.jsonl",
                }
        for task in tasks:
            output_dir = os.path.join(f"{out_folder}/{outdir}/{model}/n{num_samples_per_example}", dataset, task, f"consgen{use_constrained_geneneration}")
            sample_out_path = os.path.join(output_dir, "samples.jsonl")
            if os.path.exists(sample_out_path):
                print(f"{sample_out_path} exists, skip..")
                continue
            lang = lang_map[dataset] 
            test_file = path_map[task]
            
            head = DEBUG_PYTHON if debug else RUN_PYTHON
            cmd = cmd_tmp.format(head, model_path, num_samples_per_example, test_file, lang, use_constrained_geneneration, output_dir)
            if fewshot:
                cmd += fewshot_cmd.format(lang)
            if debug:
                # make verbose if debug
                cmd += " --custom_generate_verbose 1"
            if override:
                cmd += " --override_previous_results"
            print(f"=== {cmd} ===")
            os.system(cmd)




