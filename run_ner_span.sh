# nohup sh run_ner_span.sh >span_log.txt  2>&1 &
CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert-base-chinese
export DATA_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
export CUDA_VISIBLE_DEVICES=6
TASK_NAME="cner"

python run_ner_span_cl_zcg.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_adv \
  --do_lower_case \
  --loss_type=ce \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=35 \
  --eval_max_seq_length=35 \
  --per_gpu_train_batch_size=256 \
  --per_gpu_eval_batch_size=256 \
  --learning_rate=2e-5 \
  --num_train_epochs=20 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42

