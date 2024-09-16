#!/bin/bash

#FSDP_CPU_RAM_EFFICIENT_LOADING=1 python $HOME/projects/ llama-recipes/src/llama_recipes/finetuning.py  --use_peft --peft_method lora --quantization 8bit --model_name /mnt/m2m_nobackup/yushengsu/Meta-Llama-3-8B --output_dir /mnt/m2m_nobackup/yushengsu/fsdp_proj

export FSDP_CPU_RAM_EFFICIENT_LOADING=1

#apptainer exec --bind /mnt/m2m_nobackup/yushengsu:/mnt/m2m_nobackup/yushengsu:rw,$HOME:$HOME:rw $HOME/apptainer_sandbox_images/fsdp_rocm_pytorch-private:exec_dash_pretuned_nightly_inai_FA_ck_v0.1.1_TE_with_CP FSDP_CPU_RAM_EFFICIENT_LOADING=1 python $HOME/projects/ llama-recipes/src/llama_recipes/finetuning.py  --use_peft --peft_method lora --quantization 8bit --model_name /mnt/m2m_nobackup/yushengsu/Meta-Llama-3-8B --output_dir /mnt/m2m_nobackup/yushengsu/fsdp_proj

# --use_peft --peft_method lora --quantization 8bit

export NNODES=1
export NPROC_PER_NODE=1
export BATCH_SZIE=1

apptainer exec --bind /mnt/m2m_nobackup/yushengsu:/mnt/m2m_nobackup/yushengsu:rw,$HOME:$HOME:rw $HOME/apptainer_sandbox_images/fsdp_rocm_pytorch-private:exec_dash_pretuned_nightly_inai_FA_ck_v0.1.1_TE_with_CP torchrun --nnodes $NNODES --nproc_per_node $NPROC_PER_NODE $HOME/projects/llama-recipes/recipes/quickstart/finetuning/finetuning.py --enable_fsdp --low_cpu_fsdp --fsdp_config.pure_bf16 --model_name /mnt/m2m_nobackup/yushengsu/Meta-Llama-3-8B --batch_size_training $BATCH_SZIE --dist_checkpoint_root_folder /mnt/m2m_nobackup/yushengsu/fsdp/model_checkpoints --dist_checkpoint_folder /mnt/m2m_nobackup/yushengsu/fsdp/fine-tuned --samsum_dataset.trust_remote_code=True


