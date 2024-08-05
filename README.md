# Token Alignment via Character Matching for Subword Completion

This repo releasing the code and benchmark datasets for paper "Token Alignment via Character Matching for Subword Completion" in ACL Findings 2024. In our paper, we noticed LLMs usually generate sub-optimal responses when their input prompts ending with partial tokens. The issues are generically associated to tokenization artifacts, and are especially important when applying LLMs in practical applications like code completions. This paper proposes a novel technique, namely token alignment, to well mitigate such issues. We optimized the inference efficiency with pre-built character trie and mask cache. We conducted extensive experiments on code completions and general NL tasks to validate the improvements of our proposed token alignment.



## Evaluation of public models on execution datasets

Code to evaluate public models on MBXP datasets.

Usage :

For Falcon models we need pytorch 2.0 and transformers>=4.30

(Tested in the docker:  "747303060528.dkr.ecr.us-east-1.amazonaws.com/mstar-vector:skgouda_public_model_eval_v2)

```python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.run --nproc_per_node 8 --master_port 20000 evaluate_model.py  \
--model_name_or_path /mnt/efs/people/skgouda/repos/external/starcoder/falcon_models/models/falcon_7b_instruct/ \
--do_sample \
--temperature 0.4 \
--model_context_length 8192 \
--max_generation_length 256  \
--num_samples_per_example 5 \
--batch_size 1  \
--task_name mxeval/mbxp \
--programming_lang python \
--output_dir ~/results/falcon_7b_instruction \
--use_stopping_criteria \
--bf16 \
--override_previous_results

```

```
python3 evaluate_model.py \
--model_name_or_path facebook/opt-6.7b \
--tokenizer_name facebook/opt-6.7b \
--model_context_length 512 \
--output_dir ./eval_results/humaneval/opt-6.7b \
--do_sample \
--bf16 \
--debug
```


```
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 20000 evaluate_model.py \
--model_name_or_path bigscience/bloom-6b3 \
--tokenizer_name bigscience/bloom-6b3 \
--model_context_length 512 \
--output_dir ./eval_results/humaneval/bloom-6b3 \
--debug
```


## Starcoder Notes
- Requires
  - transformers>=4.28.1
  - tokenizers>=0.13
- BigCode models tend to generate garbage when prompts have traling newline chars


## Other dependencies

```
pip install datasets accelerate pygtrie
pip install nvidia-pyindex
pip install pytorch-quantization

```

Need to install the latest mxeval as well
```
https://github.com/amazon-science/mxeval
```


## Token Align
Given a prompt below to the model (StarCoder in this case)

```
# write a function to get three maximum numbers from a list
def three_max(l):
    re
```

The model is not likely to generate `return` due to the fact that `return` due to the artifact of tokenization. Instead, the model will generate the following:

```
# write a function to get three maximum numbers from a list
def three_max(l):
    re = []
    for i in range(len(l)):
        if i == 0:
            re.append(l[0])
        else:
            if l[i] > re[0]:
                re[0] = l[i]
            elif l[i] > re[1]:
                re[1] = l[i]
            elif l[i] > re[2]:
                re[2] = l[i]
    return re
```


However, with TokenAlign, the model aligns with the existing tokens and are able to generate correctly.

```
# write a function to get three maximum numbers from a list
def three_max(l):
    return sorted(l, reverse=True)[:3]
```


Below are the code to replicate


With token align 
```
CUDA_VISIBLE_DEVICES=0 python3 evaluate_model.py \
--model_name_or_path bigcode/starcoder \
--tokenizer_name bigcode/starcoder \
--model_context_length 2048 \
--output_dir ./eval_results/humaneval/starcoder \
--do_sample \
--bf16 \
--debug \
--use_custom_generate 1 \
--use_token_align 1 \ 
--single_context_batch_sampling 1 \
--custom_generate_verbose 1 \
```

Without token align
```
CUDA_VISIBLE_DEVICES=0 python3 evaluate_model.py \
--model_name_or_path bigcode/starcoder \
--tokenizer_name bigcode/starcoder \
--model_context_length 2048 \
--output_dir ./eval_results/humaneval/starcoder \
--do_sample \
--bf16 \
--debug \
--use_custom_generate 1 \
--use_token_align 0 \
--single_context_batch_sampling 1 \
--custom_generate_verbose 1 \
```

## To cite our work

```
@article{tokenalign_2024,
  title = {Token Alignment via Character Matching for Subword Completion},
  author = {Athiwaratkun, Ben and
        Wang, Shiqi and
        Shang, Mingyue and
        Tian, Yuchen and
        Wang, Zijian and
        Gonugondla, Sujan Kumar and
        Gouda, Sanjay Krishna and
        Kwiatowski, Rob and
        Nallapati, Ramesh and
        Xiang, Bing
    },
  url = {https://arxiv.org/pdf/2403.08688},
  publisher = {ACL Findings},
  year = {2024},
  copyright = {Creative Commons Attribution 4.0 International}
}

```