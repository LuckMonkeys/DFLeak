
# SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing 250"
SAMPLE_FLAGS="--batch_size 4 --num_samples 4 --timestep_respacing ddim25 --use_ddim True"

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
# MODEL_FLAGS="--attention_resolutions 28,14,7 --class_cond True --diffusion_steps 1000 --image_size 224 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
CUDA_VISIBLE_DEVICES=1 python scripts/classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion.pt $SAMPLE_FLAGS

