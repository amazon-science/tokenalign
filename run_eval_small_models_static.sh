
model_names="incoder-1b-tensor codegen-350M-mono-row codegen-350M-mono-tensor incoder-6b-tensor"
# model_names="codegen-6B-mono-row  codegen-2B-mono-row incoder-6b-tensor incoder-1b-row"
# model_names="codegen-2B-mono-row codegen-6B-mono-row codegen-350M-mono-row codegen-350M-mono-tensor codegen-2B-mono-tensor"
model_names="codegen-350M-mono-row  codegen-350M-mono-tensor codegen-2B-mono-tensor codegen-2B-mono-row codegen-6B-mono-row codegen-350M-mono-row"
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
        if [[ "$model_name" == *"codegen-350M-mono-tensor"* ]]; then
            model_provider=Salesforce
            path=/mnt/efs/people/xiaokaiw/public_models/codegen_mono/static/codegen_mono_350M_w8a8_cali5000_entropy.pt
            tokenizer_name='codegen-350M-mono'
        elif [[ "$model_name" == *"codegen-2B-mono-tensor"* ]]; then
            model_provider=Salesforce
            path=/mnt/efs/people/xiaokaiw/public_models/codegen_mono/static/codegen_mono_2B_w8a8_cali5000_entropy.pt
            tokenizer_name='codegen-2B-mono'
        elif [[ "$model_name" == *"codegen-350M-mono-row"* ]]; then
            model_provider=Salesforce
            path=/mnt/efs/people/xiaokaiw/public_models/codegen_mono/static/codegen_mono_350M_w8a8_cali5000_entropy_axis0.pt
            tokenizer_name='codegen-350M-mono'
        elif [[ "$model_name" == *"codegen-2B-mono-row"* ]]; then
            model_provider=Salesforce
            path=/mnt/efs/people/xiaokaiw/public_models/codegen_mono/static/codegen_mono_2B_w8a8_cali5000_entropy_axis0.pt
            tokenizer_name='codegen-2B-mono'
        elif [[ "$model_name" == *"codegen-6B-mono-row"* ]]; then
            model_provider=Salesforce
            path=/mnt/efs/people/xiaokaiw/public_models/codegen_mono/static/codegen_mono_6B_w8a8_cali5000_entropy_axis0.pt
            tokenizer_name='codegen-6B-mono'
        elif [[ "$model_name" == *"incoder-1b-tensor"* ]]; then
            model_provider=facebook
            path=/mnt/efs/people/xiaokaiw/public_models/incoder/static/incoder_1B_w8a8_cali5000_entropy.pt
            tokenizer_name=incoder-1B
        elif [[ "$model_name" == *"incoder-6b-tensor"* ]]; then
            model_provider=facebook
            path=/mnt/efs/people/xiaokaiw/public_models/incoder/static/incoder_6B_w8a8_cali5000_entropy.pt
            tokenizer_name=incoder-6B
        elif [[ "$model_name" == *"incoder-1b-row"* ]]; then
            model_provider=facebook
            path=/mnt/efs/people/xiaokaiw/public_models/incoder/static/incoder_1B_w8a8_cali5000_entropy_axis0.pt
            tokenizer_name=incoder-1B
        elif [[ "$model_name" == *"incoder-6b-row"* ]]; then
            model_provider=facebook
            path=/mnt/efs/people/xiaokaiw/public_models/incoder/static/incoder_6B_w8a8_cali5000_entropy_axis0.pt
            tokenizer_name=incoder-6B
        else
            echo "unkown model $model_name"
            exit
        fi

        python3 -m torch.distributed.run --nproc_per_node 8 --master_port 20000 evaluate_model.py \
        --model_name_or_path $path \
        --tokenizer_name $model_provider/$tokenizer_name \
        --output_dir ./eval_results_static_mse/$dataset/$model_name/$policy/ \
        --programming_lang $lang \
        --test_file $test_file \
        --use_stopping_criteria \
        --static \
        --mse \
        --max_length  512 \
        --max_context_length 512 \
        $sample_args

        ## Run to evaluate models
        python3 evaluate_model.py \
        --model_name_or_path $model_provider/$model_name \
        --tokenizer_name $model_provider/$model_name \
        --output_dir ./eval_results_static_mse/$dataset/$model_name/$policy/ \
        --programming_lang $lang \
        --test_file $test_file \
        --use_stopping_criteria \
        $sample_args \
        --run_eval_only
    done
done