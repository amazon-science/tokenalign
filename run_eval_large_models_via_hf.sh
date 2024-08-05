
#model_names="bloom"
model_names=$3 #opt-66b opt30b bloom
#datasets="humaneval mbpp mbjp mbjsp"
datasets="mbjsp mbjp mbpp humaneval"
datasets=$1
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
        else
            echo "unkown model $model_name"
            exit
        fi
        
        #python3 evaluate_model.py \
        CUDA_VISIBLE_DEVICES=$2 python3 evaluate_model.py \
        --model_name_or_path $model_provider/$model_name \
        --tokenizer_name $model_provider/$model_name \
        --output_dir ./eval_results/$dataset/$model_name/$policy/ \
        --programming_lang $lang \
        --test_file $test_file \
        --use_stopping_criteria \
        --max-memory-per-gpu $4\
        $sample_args

        # Run to evaluate models
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

# bash run_eval_large_models_via_hf.sh mbkp 0,1,2 opt-30b
# bash run_eval_large_models_via_hf.sh mbrbp 3,4,5 opt-30b
# bash run_eval_large_models_via_hf.sh mbphp 3,4,5 opt-30b

# bash run_eval_large_models_via_hf.sh mbkp 0,1,2,3 opt-66b
# bash run_eval_large_models_via_hf.sh mbrbp 4,5,6,7 opt-66b
# bash run_eval_large_models_via_hf.sh mbphp 3,4,5 opt-66b