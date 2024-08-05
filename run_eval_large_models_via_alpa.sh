
model_names="opt-175b"
#model_names="opt-30b"
datasets="mbjsp mbjp mbpp"
#datasets="mbpp mbjp mbjsp"

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
    else
        echo "unkown dataset $dataset"
        exit
    fi

    for model_name in $model_names; do

        python3 evaluate_model.py \
        --model_name_or_path alpa/$model_name \
        --tokenizer_name facebook/opt-30b \
        --output_dir ./eval_results/$dataset/$model_name/$policy/ \
        --programming_lang $lang \
        --test_file $test_file \
        --use_stopping_criteria \
        --fp16 \
        $sample_args

        # Run to evaluate models
        python3 evaluate_model.py \
        --model_name_or_path alpa/$model_name \
        --tokenizer_name facebook/opt-30b \
        --output_dir ./eval_results/$dataset/$model_name/$policy/ \
        --programming_lang $lang \
        --test_file $test_file \
        --use_stopping_criteria \
        $sample_args \
        --run_eval_only
    done
done