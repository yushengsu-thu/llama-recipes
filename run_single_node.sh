#!/bin/bash

#FSDP_CPU_RAM_EFFICIENT_LOADING=1 python $HOME/projects/ llama-recipes/src/llama_recipes/finetuning.py  --use_peft --peft_method lora --quantization 8bit --model_name /mnt/m2m_nobackup/yushengsu/Meta-Llama-3-8B --output_dir /mnt/m2m_nobackup/yushengsu/fsdp_proj


#apptainer exec --bind /mnt/m2m_nobackup/yushengsu:/mnt/m2m_nobackup/yushengsu:rw,$HOME:$HOME:rw $HOME/apptainer_sandbox_images/fsdp_rocm_pytorch-private:exec_dash_pretuned_nightly_inai_FA_ck_v0.1.1_TE_with_CP FSDP_CPU_RAM_EFFICIENT_LOADING=1 python $HOME/projects/ llama-recipes/src/llama_recipes/finetuning.py  --use_peft --peft_method lora --quantization 8bit --model_name /mnt/m2m_nobackup/yushengsu/Meta-Llama-3-8B --output_dir /mnt/m2m_nobackup/yushengsu/fsdp_proj

# --use_peft --peft_method lora --quantization 8bit


export FSDP_CPU_RAM_EFFICIENT_LOADING=1

# Refer the config: https://github.com/yushengsu-thu/llama-recipes/tree/main/recipes/quickstart/finetuning

NNODES=1
NPROC_PER_NODE=8
BATCH_SZIE=16
MODEL_NAME="/mnt/m2m_nobackup/yushengsu/Meta-Llama-3-8B"
CK_ROOT_FOLDER="/mnt/m2m_nobackup/yushengsu/llama_fsdp/model_checkpoints"
CK_FOLDER="fine-tuned"
ENABLE_FSDP=True # shards model parameters, optimizer states and gradients across DDP ranks
LOW_CPU_FSDP=False # shards model parameters, optimizer states and gradients across DDP ranks
GRADIENT_CLIPPING_THRESHOLD=1.0
NUM_EPOCHS=1
MAX_TRAIN_STEP=20
USE_FP16=False
MIXED_PRECISION=True
USE_PEFT=False
USE_FAST_KERNELS=True
USE_WANDB=True
SAVE_METRICS=True
FLOP_COUNTER=True ## Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
FLOP_COUNTER_START=3 # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.



rm -rf $CK_ROOT_FOLDER

apptainer exec \
  --bind /mnt/m2m_nobackup/yushengsu:/mnt/m2m_nobackup/yushengsu:rw,$HOME:$HOME:rw \
  $HOME/apptainer_sandbox_images/fsdp_rocm_pytorch-private:exec_dash_pretuned_nightly_inai_FA_ck_v0.1.1_TE_with_CP \
  torchrun \
  --nnodes $NNODES \
  --nproc_per_node $NPROC_PER_NODE \
  $HOME/projects/llama-recipes/recipes/quickstart/finetuning/finetuning.py \
  --enable_fsdp \
  --low_cpu_fsdp \
  --fsdp_config.pure_bf16 \
  --model_name $MODEL_NAME \
  --batch_size_training $BATCH_SZIE \
  --dist_checkpoint_root_folder $CK_ROOT_FOLDER \
  --dist_checkpoint_folder $CK_FOLDER \
  --samsum_dataset.trust_remote_code=True \
  --enable_fsdp $ENABLE_FSDP \
  --low_cpu_fsdp $LOW_CPU_FSDP \
  --gradient_clipping_threshold $GRADIENT_CLIPPING_THRESHOLD \
  --num_epochs $NUM_EPOCHS\
  --max_train_step $MAX_TRAIN_STEP \
  --use_fp16 $USE_FP16 \
  --mixed_precision $MIXED_PRECISION \
  --use_peft $USE_PEFT \
  --use_fast_kernels $USE_FAST_KERNELS \
  --use_wandb $USE_WANDB \
  --save_metrics $SAVE_METRICS \
  --flop_counter $FLOP_COUNTER \
  --flop_counter_start $FLOP_COUNTER_START \





ls $CK_ROOT_FOLDER
