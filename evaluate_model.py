import argparse
import json
import os
import torch
from datasets import load_dataset
from tqdm import tqdm
from utils.utils import count_files_present_nonemtpy
from utils.metrics import read_problems, run_passatk_eval
from utils.truncate import filter_valid_code, inference_cut_off
import torch.distributed as dist
from transformers import (
    GenerationConfig,
    StoppingCriteriaList,
    MaxLengthCriteria,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import transformers
from stopping_criterion.stopping_criterion import get_stopping_criteria_per_language, SCOPE_COMPLETION
from generation import token_align, custom_generate

from utils.utils import (
    build_dict_jsonl_idx,
    has_translation_source,
    translation_prefix,
    valid_translation_source,
)

model_path = "./"

def parse_args():
    parser=argparse.ArgumentParser(
        usage=f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.run --nproc_per_node 8 --master_port 20000 evaluate_model.py \
            --model_name_or_path {model_path}/starcoder/starcoder_15b_bf16/ \
            --do_sample --temperature 0.4 --model_context_length 8192 \
            --max_generation_length 256  --num_samples_per_example 5 --batch_size 1 \
            --task_name mxeval/mbxp --programming_lang python \
            --output_dir ~/results/starcoder/temperature_0.4_num_samples_5/mbxp/python \
            --use_stopping_criteria --bf16",
        description="a script to evaluate public models"
    )
    parser.add_argument('--model_name_or_path', type=str, default=None,
                        help='Name/path of the model we are working with')
    parser.add_argument('--tokenizer_name', type=str, default=None,
                        help='tokenizer name/path')
    parser.add_argument("--max-memory-per-gpu", type=str,
                        help="Defines maximum memory allocated to gpu", default='28GB')
    parser.add_argument('--max_context_length', type=int, default=None,
                        help='maximum output length to generate. If None, default to model context length - max generation length')
    parser.add_argument('--model_context_length', type=int, default=2048,
                        help='Context length of the model minus generation length. Default is 2048, best to set to max(prompt len) of the dataset')
    parser.add_argument('--max_generation_length', type=int, default=256,
                        help='Number of tokens to generate. Default is 256. This should be < model_context_length')
    parser.add_argument('--do_sample', default=False, action='store_true',
                        help='Sample / Greedy generation')
    parser.add_argument('--temperature', type=float, default=0.4,
                        help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='top p')
    parser.add_argument('--top_k', type=int, default=None,
                        help='top k')
    parser.add_argument('--num_beams', type=int, default=1,
                        help='number of beams')
    parser.add_argument('--num_samples_per_example', type=int, default=1,
                        help='number of samples')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for inference')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Run debugging part of the code')
    parser.add_argument('--run_eval_only', default=False, action='store_true',
                        help='Run debugging part of the code')
    parser.add_argument('--bf16', default=False, action='store_true',
                        help='To use brain float 16')
    parser.add_argument('--fp16', default=False, action='store_true',
                        help='To use float 16')
    parser.add_argument('--use_fast_tokenizer', default=False, action='store_true',
                        help='Set to use fast tokenizer'
    )
    parser.add_argument('--use_stopping_criteria', default=False, action='store_true',
                        help='Flag to use language specific stopping criterion')
    parser.add_argument('--int8', default=False, action='store_true',
                        help='Flag to use int8 weights')
    parser.add_argument('--override_previous_results', default=False, action='store_true',
                        help='override previous results')
    parser.add_argument('--test_file', type=str, default='datasets/humaneval/HumanEval.jsonl',
                        help='Test dataset')
    parser.add_argument('--programming_lang', type=str, default='python',
                        help='Programming Language')
    parser.add_argument('--valid_filter', type=str, default='func_ast_first',
                        help='Truncation logic')
    parser.add_argument('--output_dir', type=str, default='./eval_results/humaneval',
                        help='Folder to log the outputs')
    parser.add_argument('--translate_source', type=str, default='',
                        help='Transilation source')
    parser.add_argument('--fewshot_source', type=str, default='',
                        help='Few shot prompting source')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='for torch.distributed')
    parser.add_argument('--single_context_batch_sampling', type=int, default=0)
    # parser.add_argument('--custom_deepspeed', type=int, default=0)
    parser.add_argument('--use_custom_generate', type=int, default=0)
    parser.add_argument('--use_token_align', type=int, default=0)
    parser.add_argument('--custom_generate_verbose', type=int, default=0)
    parser.add_argument(
        '--use_accelerate', default=False, action='store_true',
        help='Set to use accelerate for model loading. If using int8, accelerate is used automatically'
    )
    parser.add_argument(
        "--task_name",
        type=str, default=None,
        choices=[None, "mxeval/mbxp", "mxeval/multi-humaneval", "mxeval/mathqa-x"]
    )
    parser.add_argument('--verbose', type=int, default=0)



    args=parser.parse_args()
    return args

def get_gpus_max_memory(max_memory):
    max_memory = {i: max_memory for i in range(torch.cuda.device_count())}
    return max_memory

def get_distributed_info():
    if not dist.is_initialized() and os.environ.get("RANK") is not None:
        dist.init_process_group(backend="nccl")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    group_size = 1
    num_groups = world_size
    group = rank
    return {
        "local_rank": local_rank,
        "rank": rank,
        "group": group,
        "num_groups": num_groups,
        "group_size": group_size,
        "world_size": world_size,
    }


def execution_eval_generate(
    args,
    generate_fn,
    tokenizer,
    distributed_info
):
    device = distributed_info["local_rank"]
    if distributed_info["rank"] == 0:
        if not os.path.isdir(os.path.join(args.output_dir, "output")):
            os.makedirs(os.path.join(args.output_dir, "output"))

    num_samples = args.num_samples_per_example
    batch_size = args.batch_size

    problems = construct_problem_file(args)

    if args.max_context_length is not None:
        max_context_length = args.max_context_length # still should be explicit
    else:
        max_context_length = args.model_context_length - args.max_generation_length 

    if valid_translation_source(args.translate_source):
        translation_source_dict = build_dict_jsonl_idx(args.translate_source)
    else:
        translation_source_dict = None

    fpath_format = os.path.join(
        args.output_dir, "output", "taskid-{task_idx}-gen{completion_idx}.json"
    )

    generate_config = GenerationConfig(max_new_tokens=args.max_generation_length, max_length=args.model_context_length)

    for enum_idx, task_id in enumerate(tqdm(problems, desc=f"{args.programming_lang.capitalize()} Task Loop")):
        # assume TaskName/ID format. The id part needs not be integer.
        task_idx = task_id.split("/")[1]
        if not (enum_idx % distributed_info["num_groups"] == distributed_info["group"]):
            continue
        if not args.override_previous_results:
            fnames = [
                fpath_format.format(task_idx=task_idx, completion_idx=_idx)
                for _idx in range(num_samples)
            ]
            count, all_count = count_files_present_nonemtpy(fnames)
            if count == all_count:
                if args.verbose:
                    print(
                        f"Result caching mode: Skipping case {task_id}. Generated all {all_count}"
                    )
                continue
            else:
                if args.verbose:
                    print(
                        f"Result caching mode: Only {count} out of {all_count} were generated. Regenerating task {task_id}"
                    )
        prompt = problems[task_id]["prompt"]
        if "falcon" in args.model_name_or_path:
            prompt = prompt.rstrip()
        #if "gpt_bigcode" in model.config.model_type:
        #    prompt = prompt.rstrip("\n")
        if "language" in problems[task_id]:
            execution_language = problems[task_id]["language"]
        else:
            print("Warning -- no language in problem file")
            execution_language = None

        # BenA: translation mode
        # translate_mode = valid_translation_source(data_args.translate_source)
        
        ###### prompt execution prompt ######

        translate_mode = translation_source_dict is not None
        if translate_mode and has_translation_source(translation_source_dict, task_idx):
            execution_prompt = prompt
            # get dict
            translation_prefix_text = translation_prefix(translation_source_dict, task_idx)
            prompt = translation_prefix_text + prompt
        else:
            execution_prompt = None
        # BenA: end translation mode

        # BenA: few shot prompting option
        if args.fewshot_source == "":
            args.fewshot_soruce = None

        if args.fewshot_source is not None and os.path.isfile(
            args.fewshot_source
        ):
            fewshot_str = open(args.fewshot_source, "r").read()
            assert (
                fewshot_str.strip() != ""
            ), f"Empty few shot prompting file {args.fewshot_source}"
            if execution_prompt is None:
                execution_prompt = prompt
            elif not translate_mode:
                execution_prompt = None
            prompt = fewshot_str + prompt

        if 'codet5' in args.model_name_or_path:
            execution_prompt = prompt
            prompt = prompt + "<extra_id_0>"
        # BenA

        inputs = tokenizer(prompt)
        with torch.no_grad():
            if 'bloom' in args.model_name_or_path:
                input_ids = torch.tensor([inputs.input_ids[-max_context_length:-1]]).to(device)
            else:
                input_ids = torch.tensor([inputs.input_ids[-max_context_length:]]).to(device)
            if args.debug:
                print(f"input ids len: {len(input_ids[0])}")
                print(f"max gen length: {args.max_generation_length}")
            completion_idx = -1
            for i in range(0, num_samples, batch_size):
                num_return_sequences = min(num_samples - i, batch_size)
                # try:,

                stopping_criteria_list = StoppingCriteriaList()


                if args.use_stopping_criteria:
                    stopping_criteria_list = get_stopping_criteria_per_language(
                        language=args.programming_lang,
                        eog_type=SCOPE_COMPLETION,
                        # eog_type=LINE_COMPLETION,
                        max_lines=1000,
                        num_return_sequences=num_return_sequences,
                        input_len=len(input_ids[0]),
                        input_indent=0,
                        tokenizer=tokenizer,
                        max_new_tokens=args.max_generation_length,
                        init_input_ids=input_ids,
                    )
                # else:
                #     THIS MAY CHANGE IN FUTURE HF VERSIONS
                #     """FROM THE DOCS
                #     "The class `MaxNewTokensCriteria` is deprecated. "
                #     f"Please use `MaxLengthCriteria(max_length={start_length + max_new_tokens})` "
                #     "with `max_length = start_length + max_new_tokens` instead.",
                #     """
                #     # This is likely set within HF generate because of default generation_config :/
                #     # stopping_criteria_list.append(
                #     #     MaxLengthCriteria(
                #     #         max_length=len(input_ids[0]) + args.max_generation_length
                #     #     )
                #     # )
                #     print(f"Custom stopping criteria is not used. MaxLengthCriteria will be used")

                output_dict = generate_fn(
                    inputs=input_ids,
                    do_sample=args.do_sample,
                    generation_config=generate_config,
                    top_p=args.top_p,
                    max_new_tokens=args.max_generation_length,
                    temperature=args.temperature,
                    num_beams=args.num_beams,
                    num_return_sequences=num_return_sequences,
                    stopping_criteria=stopping_criteria_list,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                sequences = output_dict.sequences
                # initial_context_length counts the token sequence length of the original prompts
                # which might not be reliable to use for constrained generation
                # in truncation we are now using string level matching instead of token level.
                initial_context_length = len(sequences[0]) - len(output_dict.scores)

                if 'codet5' in args.model_name_or_path:
                    initial_context_length = 0
                predictions_post_eos = truncate(
                    args, tokenizer, task_id, prompt, execution_prompt, input_ids, sequences, initial_context_length
                )

                for prediction in predictions_post_eos:
                    completion_idx += 1
                    fpath = fpath_format.format(
                        task_idx=task_idx, completion_idx=completion_idx
                    )
                    if execution_language is not None:
                        prediction["language"] = execution_language
                    with open(fpath, "w", encoding="utf8") as _f:
                        json.dump(prediction, _f)
                    if args.debug:
                        print('--------------------prediction------------------------')
                        print(prediction['input'] + prediction['ori_pred'])
                        print('--------------------completion------------------------')
                        print(prediction['input'] + prediction['completion'])

    if distributed_info["world_size"] > 1:
        dist.barrier()

    if distributed_info["rank"] == 0:
        run_passatk_eval(
            problems,
            args.programming_lang,
            args.output_dir,
            args.num_samples_per_example,
            args.override_previous_results,
        )


def truncate(args, tokenizer, task_id, prompt, execution_prompt, input_ids, sequences, initial_context_length):
    if args.programming_lang == "python":
        predictions_post_eos = filter_valid_code(
                        true_str_input=prompt,
                        execution_prompt=execution_prompt,
                        inputs=input_ids,
                        sequences=sequences,
                        initial_context_length=initial_context_length,
                        tokenizer=tokenizer,
                        task_id=task_id,
                        post_process=args.valid_filter,
                        skip_special_tokens=True,
                        mean_logp=None,
                    )
    else:
        predictions_post_eos = inference_cut_off(
                        true_str_input=prompt,
                        inputs=input_ids,
                        sequences=sequences,
                        token_len_prompt_input=initial_context_length,
                        tokenizer=tokenizer,
                        skip_special_tokens=True,
                        task_id=task_id,
                        language=args.programming_lang,
                        input_indent=0,
                        mean_logp=None,
                    )
    return predictions_post_eos


def debug_code(args, generate_fn, tokenizer, device):
    prompts = []
    prompt1 = """#function to add two numbers
def add(a,b):
    """
    prompts.append(prompt1)
    prompt2 = """#function to sort
def sort(arr):
    """
    prompts.append(prompt2)

    prompt4 = """# write a function to get three maximum numbers from a list
def three_max(l):
    re"""
    prompts.append(prompt4)


    for idx, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        if args.use_stopping_criteria:
            stopping_criteria = get_stopping_criteria_per_language(
                language=args.programming_lang,
                eog_type=SCOPE_COMPLETION,
                max_lines=1000,
                num_return_sequences=1,
                input_len=len(input_ids[0]),
                input_indent=0,
                tokenizer=tokenizer,
                max_new_tokens=args.max_generation_length,
                init_input_ids=input_ids,
            )
        else:
            stopping_criteria = None

        output_dict = generate_fn(inputs=input_ids,
                                    max_length=len(input_ids[0]) + args.max_generation_length,
                                    temperature=args.temperature,
                                    num_beams=args.num_beams,
                                    num_return_sequences=1,
                                    stopping_criteria=stopping_criteria,
                                    use_cache=True,
                                    return_dict_in_generate=True,
                                    output_scores=True,
                                    top_p=args.top_p,
                                    do_sample=args.do_sample)


        sequences = output_dict.sequences
        initial_context_length = len(sequences[0]) - len(output_dict.scores)

        print(tokenizer.convert_ids_to_tokens(sequences))
        generated_string = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        print(f"init ctx len: {initial_context_length}, {len(sequences[0])=}, {len(output_dict.scores)=}")
        print('-----------------------Untruncated output--------------------')
        print(generated_string[0])
        execution_prompt = prompt
        predictions_post_eos = truncate(args, tokenizer, idx, prompt, execution_prompt, input_ids, sequences, initial_context_length)
        print(predictions_post_eos)
        print('------------------------Truncated output---------------------')
        print(predictions_post_eos[0]['input'] + predictions_post_eos[0]['completion'])


def construct_problem_file(args):
    if not args.task_name:
        problems = read_problems(args.test_file)
    else:
        problems = load_dataset(args.task_name, args.programming_lang)
        try:
            problems = problems["test"]
        except KeyError:
            raise ValueError(f"Dataset is in unexpected format, loaded dataset has these keys: {list(problems.keys())}")
        problems = read_problems(problems)
    return problems


def main():
    args=parse_args()
    distributed_info = get_distributed_info()
    if distributed_info["rank"] == 0:
        print('Arguments : ',  args)

    device = distributed_info["local_rank"] if torch.cuda.is_available() else 'cpu'

    if args.run_eval_only:
        if distributed_info["rank"] == 0:
            run_passatk_eval(
                construct_problem_file(args),
                args.programming_lang,
                args.output_dir,
                args.num_samples_per_example,
                args.override_previous_results,
            )
        return

    tokenizer = AutoTokenizer.from_pretrained(
                    args.tokenizer_name or args.model_name_or_path,
                    use_fast=args.use_fast_tokenizer,
                    trust_remote_code=True,
                )
    # for llama tokenizer, it will decode SPIECE_UNDERLINE as whitespace and we cannot just decode as empty as other tokenizers do
    tokenizer.is_llama = (type(tokenizer) in [
                            transformers.models.llama.tokenization_llama.LlamaTokenizer, 
                            transformers.models.code_llama.tokenization_code_llama.CodeLlamaTokenizer
                        ])

    args.tokenizer_name = tokenizer.name_or_path
    print(f"Using tokenizer {args.tokenizer_name=}, {tokenizer.is_llama=}")

    if 'codegen' in args.tokenizer_name:
        print('Setting pad token id')
        tokenizer.pad_token_id = 50256

    if args.int8:
        print(f"Using accelerate for int8 model loading")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            max_memory=get_gpus_max_memory(args.max_memory_per_gpu),
            load_in_8bit = True,
            revision=None,
        )
    else:
        print('Loading model')
        if args.bf16:
            dtype=torch.bfloat16
        elif args.fp16:
            dtype=torch.half
        else:
            dtype="auto"
        model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    device_map="auto" if args.use_accelerate else None,
                    low_cpu_mem_usage=not("falcon" in args.model_name_or_path),
                    trust_remote_code="falcon" in args.model_name_or_path,
                    torch_dtype=dtype
                )
        print("Finished loading model")
        model = model.to(device)
        # BenA -- is there no model.to in the int8 clause above?

    print(f'Loading {args.model_name_or_path} loading complete')

    # construct a generate function and use the function directly instead of the model
    def get_generate_function(model):
        # explicit_broadcast performs broadcasting by reference -- this is used for DS 0.5.9 and pytorch kernel
        # for DS 0.8, this is done in DS, and doing it in the custom generate would make it incorrect
        if args.use_custom_generate:
            print("Using Custom Generate")
            if args.use_token_align:
                token_align.prepare_vocab_trie(tokenizer)
                print("Successfully prepared vocabulary trie for TokenAlign")

            def generate_function(*argss, **kwargs):
                return custom_generate.generate(
                    model,
                    *argss,
                    single_context_batch_sampling=bool(args.single_context_batch_sampling),
                    token_align=args.use_token_align,
                    tokenizer=tokenizer,
                    verbose=bool(args.custom_generate_verbose),
                    # explicit_broadcast=explicit_broadcast,
                    **kwargs,
                )
        else:
            def generate_function(*args, **kwargs):
                return model.generate(*args, **kwargs)

        return generate_function

    generate_fn = get_generate_function(model)

    if args.debug:
        debug_code(args, generate_fn, tokenizer, device)
    else:
        distributed_info = get_distributed_info()
        execution_eval_generate(args, generate_fn, tokenizer, distributed_info)


if __name__ == '__main__':
    main()
