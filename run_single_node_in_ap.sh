#!/bin/bash

# Using the SDPA attention implementation on multi-gpu setup with ROCM may lead to performance issues due to the FA backend. Disabling it to use alternative backends.
# TORCH_FORCE_SDP_BACKEND="mem_efficient"
#python: --use_peft --peft_method lora --quantization 8bit

#FSDP_CPU_RAM_EFFICIENT_LOADING=1 python $HOME/projects/llama-recipes/recipes/quickstart/finetuning/finetuning.py --model_name /mnt/m2m_nobackup/yushengsu/Meta-Llama-3-8B --output_dir /mnt/m2m_nobackup/yushengsu/fsdp_proj --samsum_dataset.trust_remote_code=True

NNODES=1
NPROC_PER_NODE=1
BATCH_SZIE=1

torchrun --nnodes $NNODES --nproc_per_node $NPROC_PER_NODE $HOME/projects/llama-recipes/recipes/quickstart/finetuning/finetuning.py --enable_fsdp --low_cpu_fsdp --fsdp_config.pure_bf16 --model_name /mnt/m2m_nobackup/yushengsu/Meta-Llama-3-8B --batch_size_training $BATCH_SZIE --dist_checkpoint_root_folder /mnt/m2m_nobackup/yushengsu/fsdp/model_checkpoints --dist_checkpoint_folder /mnt/m2m_nobackup/yushengsu/fsdp/fine-tuned --samsum_dataset.trust_remote_code=True
