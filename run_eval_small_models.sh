model_names="opt-125m opt-350m opt-1.3b opt-6.7b bloom-560m bloom-1b1 bloom-1b7 bloom-3b bloom-7b1"
model_names="opt-1.3b opt-2.7b opt-6.7b opt-13b bloom-560m bloom-1b1 bloom-1b7 bloom-3b bloom-7b1"
model_names="incoder-6B" # codegen-6B-mono codegen-2B-mono incoder-1B incoder-6B"
model_names="codegen-16B-mono codegen-16B-multi" # codegen-16B-mono codegen-16B-multi 
#model_names="opt-6.7b"
datasets="mbpp humaneval"
# model_names="opt-66b"
# datasets="mbpp mbjp mbjsp"

# model_names="bloom-1b7"
# model_names="opt-66b"
# datasets="mbphp"

num_samples=10
top_p=0.95
temperature=0.2


for dataset in $datasets; do
    if [ $num_samples == 0 ]; then
        policy=greedy
        sample_args="--num_samples_per_example 1"
    else
        policy=sampling
        sample_args="--top_p $top_p --temperature $temperature --do_sample --num_samples_per_example $num_samples"
    fi


    if [ $dataset == humaneval ]; then
        test_file=/mnt/efs/projects/datasets/humaneval/HumanEval.jsonl
        lang=python
    elif [ $dataset == mbpp ]; then
        test_file=/mnt/efs/projects/datasets/mbpp/mbpp_wtest.jsonl
        lang=python
    elif [ $dataset == mbjp ]; then
        test_file=/mnt/efs/people/benathi/data/exec_eval/mbjp_beta_wtest.jsonl
        lang=java
    elif [ $dataset == mbjsp ]; then
        test_file=/mnt/efs/people/benathi/data/exec_eval/mbjsp_beta_wtest.jsonl
        lang=javascript
    elif [ $dataset == mbkp ]; then
        test_file=/mnt/efs/people/benathi/data/exec_eval/mbkp_alpha.jsonl
        lang=kotlin
    elif [ $dataset == mbrbp ]; then
        test_file=/mnt/efs/people/benathi/data/exec_eval/mbrbp_alpha.jsonl
        lang=ruby
    elif [ $dataset == mbphp ]; then
        test_file=/mnt/efs/people/benathi/data/exec_eval/mbphp_alpha.jsonl
        lang=php
    else
        echo "unkown dataset $dataset"
        exit
    fi

    for model_name in $model_names; do
        if [[ "$model_name" == *"opt"* ]]; then
            model_provider=facebook
        elif [[ "$model_name" == *"incoder"* ]]; then
            model_provider=facebook
        elif [[ "$model_name" == *"bloom"* ]]; then
            model_provider=bigscience
        elif [[ "$model_name" == *"codegen"* ]]; then
            model_provider=Salesforce
        else
            echo "unkown model $model_name"
            exit
        fi

        python3 -m torch.distributed.run --nproc_per_node 1 --master_port 20000 evaluate_model.py \
        --model_name_or_path $model_provider/$model_name \
        --tokenizer_name $model_provider/$model_name \
        --output_dir ./eval_results/$dataset/$model_name/$policy/ \
        --programming_lang $lang \
        --test_file $test_file \
        --use_stopping_criteria \
        --bf16 \
        --max_length  512 \
        --max_context_length 512 \
        $sample_args

        ## Run to evaluate models
        python3 evaluate_model.py \
        --model_name_or_path $model_provider/$model_name \
        --tokenizer_name $model_provider/$model_name \
        --output_dir ./eval_results/$dataset/$model_name/$policy/ \
        --programming_lang $lang \
        --test_file $test_file \
        --use_stopping_criteria \
        $sample_args \
        --run_eval_only
    done
done