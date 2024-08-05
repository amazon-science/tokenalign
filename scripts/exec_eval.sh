eval_path=$1
model_path=$2
dataset_path=$3

for lang in python ; do
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.run --nproc_per_node 8 --master_port 20000 evaluate_model.py \
	--model_name_or_path $model_path/starcoder/starcoder_15b_bf16/ \
	--do_sample --temperature 0.4 \
       	--model_context_length 2048 \
	--max_generation_length 256 \
      	--num_samples_per_example 10 \
       	--batch_size 10 \
	--task_name mxeval/mbxp \
       	--programming_lang $lang \
  --use_custom_generate 1 \
  --use_token_align 1 \
	--output_dir $eval_path/results/local/starcoder/bs10_temp0.4_tokenalign/mbxp/$lang \
	--use_stopping_criteria --bf16
done


