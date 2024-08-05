model_names="opt-125m opt-350m opt-1.3b opt-6.7b bloom-560m bloom-1b1 bloom-1b7 bloom-3b bloom-7b1"
model_names="opt-1.3b opt-2.7b opt-6.7b opt-13b bloom-560m bloom-1b1 bloom-1b7 bloom-3b bloom-7b1"
model_names="codegen-6B-mono codegen-6B-multi codegen-16B-mono codegen-16B-multi codegen-350M-mono codegen-350M-multi codegen-2B-mono codegen-2B-multi"
model_names="codegen-350M-multi" # codegen-350M-multi"
# model_names="opt-6.7b"
# dtasets="mbkp mbrbp mbphp"
# model_names="opt-2.7b"
model_names="codegen-2B-mono"
datasets="mbphp"

num_samples=0
top_p=0.95
temperature=0.2


for dataset in $datasets; do
    if [ $num_samples == 0 ]; then
        policy=greedy
        sample_args="--num_samples_per_example 1"
    else
        policy=sampling
        sample_args="--top_p $top_p --temperature $temperature --do_sample --num_samples_per_example $num_samples --do_rank"
    fi


    if [ $dataset == humaneval ]; then
        test_file=/mnt/efs/projects/datasets/humaneval/HumanEval.jsonl
        fewshot_source=fewshot_prompting/python_fewshot_v1.py
        lang=python
    elif [ $dataset == mbpp ]; then
        test_file=/mnt/efs/projects/datasets/mbpp/mbpp_wtest.jsonl
        fewshot_source=fewshot_prompting/python_fewshot_v1.py
        lang=python
    elif [ $dataset == mbjp ]; then
        test_file=/mnt/efs/people/benathi/data/exec_eval/mbjp_beta_wtest.jsonl
        fewshot_source=fewshot_prompting/java_fewshot_v1.java
        lang=java
    elif [ $dataset == mbjsp ]; then
        test_file=/mnt/efs/people/benathi/data/exec_eval/mbjsp_beta_wtest.jsonl
        fewshot_source=fewshot_prompting/javascript_fewshot_v1.js
        lang=javascript
    elif [ $dataset == mbkp ]; then
        test_file=/mnt/efs/people/benathi/data/exec_eval/mbkp_alpha.jsonl
        fewshot_source=fewshot_prompting/kotlin_fewshot_v1.kt
        lang=kotlin
    elif [ $dataset == mbrbp ]; then
        test_file=/mnt/efs/people/benathi/data/exec_eval/mbrbp_alpha.jsonl
        fewshot_source=fewshot_prompting/ruby_fewshot_v1.rb
        lang=ruby
    elif [ $dataset == mbphp ]; then
        test_file=/mnt/efs/people/benathi/data/exec_eval/mbphp_alpha.jsonl
        fewshot_source=fewshot_prompting/php_fewshot_v1.php
        lang=php
    else
        echo "unkown dataset $dataset"
        exit
    fi

    translate_source=/mnt/efs/projects/datasets/mbpp/mbpp_wtest.jsonl

    for model_name in $model_names; do
        if [[ "$model_name" == *"opt"* ]]; then
            model_provider=facebook
        elif [[ "$model_name" == *"bloom"* ]]; then
            model_provider=bigscience
        elif [[ "$model_name" == *"codegen"* ]]; then
            model_provider=Salesforce
        else
            echo "unkown model $model_name"
            exit
        fi

        python3 -m torch.distributed.run --nproc_per_node 8 --master_port 20000 evaluate_model.py \
        --model_name_or_path $model_provider/$model_name \
        --tokenizer_name $model_provider/$model_name \
        --output_dir ./eval_results_transilation/$dataset/$model_name/$policy/ \
        --programming_lang $lang \
        --test_file $test_file \
        --translate_source  $translate_source \
        --use_stopping_criteria \
        --fp16 \
        $sample_args

        # Run to evaluate models
        python3 evaluate_model.py \
        --model_name_or_path $model_provider/$model_name \
        --tokenizer_name $model_provider/$model_name \
        --output_dir ./eval_results_transilation/$dataset/$model_name/$policy/ \
        --programming_lang $lang \
        --test_file $test_file \
        --use_stopping_criteria \
        --run_eval_only
    done
done