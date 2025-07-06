export CUDA_VISIBLE_DEVICES=1


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SAHU/direct_5_working \
  --model_id SAHU_group \
  --model MultiImageFeatureNet  \
  --data SAHU \
  --data_type_method uint8 \
  --step 96 \
  --seq_len 96 \
  --num_class 5 \
  --e_layers 2 \
  --batch_size 4 \
  --d_model 512 \
  --d_ff 2048 \
  --feature_dim 128 \
  --n_heads 8 \
  --d_layers 1 \
  --dropout 0.1 \
  --des 'large-kernel-SK' \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 1 \
  --patience 10 \
  --use_gpu True \
  --gpu 0 \
  --gpu_type cuda \
  --gaf_method summation \
  # --hvac_groups "SA_TEMP,OA_TEMP,MA_TEMP,RA_TEMP,ZONE_TEMP_1,ZONE_TEMP_2,ZONE_TEMP_3,ZONE_TEMP_4,ZONE_TEMP_5|OA_CFM,RA_CFM,SA_CFM|SA_SP,SA_SPSPT|SF_WAT,RF_WAT|SF_SPD,RF_SPD,SF_CS,RF_CS|CHWC_VLV_DM,CHWC_VLV|OA_DMPR_DM,RA_DMPR_DM,OA_DMPR,RA_DMPR"
