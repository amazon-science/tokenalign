
model_names="codegen_mono_350M codegen_mono_2B codegen_mono_6B"
# model_names="codegen_mono_2B"
# model_names="codegen_mono_6B"

model_names="incoder_1B incoder_6B"
#model_names="codet5_large_ntppy codet5_base"
datasets="humaneval mbpp"

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
        elif [[ "$model_name" == *"bloom"* ]]; then
            model_provider=bigscience
        elif [[ "$model_name" == *"codegen"* ]]; then
            model_provider=Salesforce
            path=/mnt/efs/people/xiaokaiw/public_models/codegen/"$model_name"_dynamic_lmheadfp.pt
            tokenizer_name='codegen-6B-mono'
        elif [[ "$model_name" == *"incoder"* ]]; then
            model_provider=facebook
            path=/mnt/efs/people/xiaokaiw/public_models/incoder/"$model_name"_dynamic_lmheadfp.pt
            tokenizer_name=incoder-6B
        elif [[ "$model_name" == *"codet5"* ]]; then
            model_provider=Salesforce
            path=/mnt/efs/people/xiaokaiw/public_models/codegen/"$model_name"_dynamic_lmheadfp.pt
            tokenizer_name=codet5-large
        else
            echo "unkown model $model_name"
            exit
        fi

        python3 evaluate_model.py \
        --model_name_or_path $path \
        --tokenizer_name $model_provider/$tokenizer_name \
        --output_dir ./eval_results_dynamic/$dataset/$model_name/$policy/ \
        --programming_lang $lang \
        --test_file $test_file \
        --use_stopping_criteria \
        --dynamic \
        --max_length  512 \
        --max_context_length 512 \
        $sample_args

        # Run to evaluate models
        python3 evaluate_model.py \
        --model_name_or_path $path \
        --tokenizer_name $model_provider/$tokenizer_name \
        --output_dir ./eval_results_dynamic/$dataset/$model_name/$policy/ \
        --programming_lang $lang \
        --test_file $test_file \
        --use_stopping_criteria \
        --override_previous_results \
        $sample_args \
        --run_eval_only
    done
done