
for lang in javascript ; do
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.run --nproc_per_node 8 --master_port 20000 evaluate_model.py \
	--model_name_or_path /mnt/efs/people/skgouda/repos/external/starcoder/starcoder_15b_bf16/ \
	--do_sample --temperature 0.4 \
       	--model_context_length 8192 \
  --max_context_length 2048 \
	--max_generation_length 256 \
      	--num_samples_per_example 10 \
       	--batch_size 5 \
  --test_file /mnt/efs/people/wshiqi/robustness_benchmark/datasets/perturbed_mbxp_v2/mbjsp/full/format/mbjsp_broken_word_rand_s0.jsonl \
       	--programming_lang $lang \
  --use_custom_generate 1 \
  --use_token_align 1 \
	--output_dir /mnt/efs/people/benathi/experiments/starcoder_evaluations/results/local/starcoder/mbxp_${lang}_partial_tokenalign1/mbxp/$lang \
	--use_stopping_criteria --bf16
done


for lang in javascript ; do
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.run --nproc_per_node 8 --master_port 20000 evaluate_model.py \
	--model_name_or_path /mnt/efs/people/skgouda/repos/external/starcoder/starcoder_15b_bf16/ \
	--do_sample --temperature 0.4 \
       	--model_context_length 8192 \
  --max_context_length 2048 \
	--max_generation_length 256 \
      	--num_samples_per_example 10 \
       	--batch_size 5 \
  --test_file /mnt/efs/people/wshiqi/robustness_benchmark/datasets/perturbed_mbxp_v2/mbjsp/full/format/mbjsp_broken_word_rand_s0.jsonl \
       	--programming_lang $lang \
  --use_custom_generate 1 \
  --use_token_align 0 \
	--output_dir /mnt/efs/people/benathi/experiments/starcoder_evaluations/results/local/starcoder/mbxp_${lang}_partial_tokenalign0/mbxp/$lang \
	--use_stopping_criteria --bf16
done



for lang in javascript ; do
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.run --nproc_per_node 8 --master_port 20000 evaluate_model.py \
	--model_name_or_path /mnt/efs/people/skgouda/repos/external/starcoder/starcoder_15b_bf16/ \
	--do_sample --temperature 0.4 \
       	--model_context_length 8192 \
  --max_context_length 2048 \
	--max_generation_length 256 \
      	--num_samples_per_example 10 \
       	--batch_size 5 \
  --test_file /mnt/efs/people/wshiqi/robustness_benchmark/datasets/perturbed_mbxp_v2/mbjsp/full/format/mbjsp_broken_word_fix_rand_s0.jsonl \
       	--programming_lang $lang \
  --use_custom_generate 1 \
  --use_token_align 1 \
	--output_dir /mnt/efs/people/benathi/experiments/starcoder_evaluations/results/local/starcoder/mbxp_${lang}_nonpartial_tokenalign1/mbxp/$lang \
	--use_stopping_criteria --bf16
done


for lang in javascript ; do
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.run --nproc_per_node 8 --master_port 20000 evaluate_model.py \
	--model_name_or_path /mnt/efs/people/skgouda/repos/external/starcoder/starcoder_15b_bf16/ \
	--do_sample --temperature 0.4 \
       	--model_context_length 8192 \
  --max_context_length 2048 \
	--max_generation_length 256 \
      	--num_samples_per_example 10 \
       	--batch_size 5 \
  --test_file /mnt/efs/people/wshiqi/robustness_benchmark/datasets/perturbed_mbxp_v2/mbjsp/full/format/mbjsp_broken_word_fix_rand_s0.jsonl \
       	--programming_lang $lang \
  --use_custom_generate 1 \
  --use_token_align 0 \
	--output_dir /mnt/efs/people/benathi/experiments/starcoder_evaluations/results/local/starcoder/mbxp_${lang}_nonpartial_tokenalign0/mbxp/$lang \
	--use_stopping_criteria --bf16
done