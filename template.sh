batch_size=1
temperature=($1)
num_samples=($2)
tasks=("$3")
languages=("$4")
output_root=$5

for task in ${tasks[@]}; do
    for num_samples in ${num_samples[@]}; do
        for temperature in ${temperature}; do
            for lang in ${languages[@]}; do
                output_dir="${output_root}/temperature_${temperature}_num_samples_${num_samples}/${task}/$lang"
                printf "======== Running with the following parameters ========\n"
                printf "language: ${lang}\ntask: ${task}\nnum_samples: ${num_samples}\ntemperature: ${temperature}\n"
                printf "========================================================\n"
                eval "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 /evaluate-public-models/evaluate_model.py \
                    --model_name_or_path /mnt/skgouda/starcoder_15b_bf16/ \
                    --tokenizer_name bigcode/starcoder \
                    --do_sample \
                    --temperature $temperature \
                    --max_length 8192 \
                    --num_samples_per_example $num_samples \
                    --batch_size 2 \
                    --max_context_length 7680 \
                    --task_name mxeval/${task} \
                    --programming_lang $lang \
                    --output_dir $output_dir \
                    --use_stopping_criteria  \
                    --bf16 "
            done
        done
    done
done

