
ckpt=checkpints/128-v0.pt # path to your checkpints



HOST_GPU_NUM=1
HOST_NUM=1
CHIEF_IP=127.0.0.1
INDEX=0


torchrun --nproc_per_node=$HOST_GPU_NUM --nnodes=$HOST_NUM --master_addr=$CHIEF_IP --master_port=12594 --node_rank=$INDEX \
mira/scripts/evaluation/ddp_wrapper.py \
--module 'inference' \
--seed 2000 \
--ckpt_path  $ckpt \
--base "./configs/Mira/config_128_mira.yaml" \
--savedir ./Saved_Text2Video-Test/test-mira \
--n_samples 1 \
--bs 1 \
--height 80 \
--width 128 \
--unconditional_guidance_scale 12 \
--ddim_steps  50 --ddim_eta 1.0 \
--prompt_file ./prompts/test_prompt.txt