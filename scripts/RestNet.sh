export CUDA_VISIBLE_DEVICES=1

# 使用普通ResNet
# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/SAHU/ \
#   --model_id SAHU \
#   --model RestNet  \
#   --data SAHU \
#   --step 72 \
#   --seq_len 72 \
#   --num_class 3 \
#   --e_layers 2 \
#   --batch_size 16 \
#   --d_model 512 \
#   --d_ff 2048 \
#   --n_heads 8 \
#   --d_layers 1 \
#   --dropout 0.1 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.0001 \
#   --train_epochs 30 \
#   --patience 10 \
#   --use_gpu True \
#   --gpu 0 \
#   --gpu_type cuda \
#   --gaf_method summation

# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/SAHU/ \
#   --model_id SAHU \
#   --model RestNet  \
#   --data SAHU \
#   --step 72 \
#   --seq_len 72 \
#   --num_class 3 \
#   --e_layers 2 \
#   --batch_size 16 \
#   --d_model 512 \
#   --d_ff 2048 \
#   --n_heads 8 \
#   --d_layers 1 \
#   --dropout 0.1 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.0001 \
#   --train_epochs 30 \
#   --patience 10 \
#   --use_gpu True \
#   --gpu 0 \
#   --gpu_type cuda \
#   --gaf_method difference
# 使用ClusteredResNet（默认通道分组）
# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/SAHU/direct_5 \
#   --model_id SAHU \
#   --model ClusteredResNet  \
#   --data SAHU \
#   --step 72 \
#   --seq_len 72 \
#   --num_class 5 \
#   --e_layers 2 \
#   --batch_size 16 \
#   --d_model 512 \
#   --d_ff 2048 \
#   --n_heads 8 \
#   --d_layers 1 \
#   --dropout 0.1 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.0001 \
#   --train_epochs 50 \
#   --patience 10 \
#   --use_gpu True \
#   --gpu 0 \
#   --gpu_type cuda \
#   --gaf_method difference\
#   --channel_groups "0,1|2|3,4,5,6|7,8,9,10|11,12,13,14|15,16,17,18,19|20,21,22,23|24|25,26,27,28,29"

# 使用ClusteredResNet（自定义通道分组）
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SAHU/direct_5 \
  --model_id SAHU \
  --model ClusteredResNet  \
  --data SAHU \
  --step 72 \
  --seq_len 72 \
  --num_class 4 \
  --e_layers 2 \
  --batch_size 16 \
  --d_model 512 \
  --d_ff 2048 \
  --n_heads 8 \
  --d_layers 1 \
  --dropout 0.1 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  --use_gpu True \
  --gpu 0 \
  --gpu_type cuda \
  --gaf_method summation \
  --channel_groups "0,1|2|3,4,5,6|7,8,9,10|11,12,13,14|15,16,17,18,19|20,21,22,23|24|25,26,27,28,29"
