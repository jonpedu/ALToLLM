set -x

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-64}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

length_weight=1e-3
OUTPUT_DIR="runs/stage3_${length_weight}"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_grpo.py \
  --model_name_or_path "yayafengzi/ALToLLM-8B" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./example/data_seg.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 1 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --length_weight ${length_weight} \
  --freeze_llm False \
  --freeze_mlp True \
  --freeze_backbone True \
  --freeze_decoder True \
  --num_token_trained 32 \
  --vision_select_layer -1 \
  --dataloader_num_workers 0 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 100 \
  --save_total_limit 1 \
  --learning_rate 1e-6 \
  --max_grad_norm 1.0 \
  --weight_decay 0.05 \
  --warmup_steps 50 \
  --lr_scheduler_type "constant" \
  --logging_steps 1 \
  --max_seq_length 1024 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "config/zero_stage2_config.json" \
  --report_to "tensorboard" \ 
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"