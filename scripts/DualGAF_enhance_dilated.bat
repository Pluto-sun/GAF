@echo off
REM 模块配置说明:
REM extractor_type: large_kernel (大核卷积), inception (Inception网络), dilated/optimized_dilated (膨胀卷积), multiscale (多尺度)
REM fusion_type: adaptive (自适应), concat (拼接), bidirectional (双向注意力), gated (门控融合), add (相加), mul (相乘), weighted_add (加权相加)
REM attention_type: channel (通道注意力), spatial (空间注意力), cbam (CBAM), self (自注意力), none (无注意力)
REM classifier_type: mlp (多层感知机), simple (简单分类器) 
REM use_statistical_features: 是否使用统计特征
REM stat_type: 'basic', 'comprehensive', 'correlation_focused'
REM multimodal_fusion_strategy: 'concat', 'attention', 'gated', 'adaptive'

REM 增强版双路GAF网络运行示例脚本
call conda activate test_env
REM 替换为你的环境名，例如 test_env

set CUDA_VISIBLE_DEVICES=1
echo 增强版双路GAF网络实验脚本
echo ================================

REM 基础配置
set model=DualGAFNet
set data=DualGAF_DDAHU
set root_path=./dataset/DDAHU/direct_5_working
set epochs=1000
set batch_size=8
set learning_rate=0.001
set step=96
set seq_len=96
set feature_dim=64
set large_feature_dim=128

REM 实验1: 基础增强版网络（使用统计特征）
echo 实验1: 基础增强版网络+concat
python run.py ^
    --model %model% ^
    --data %data% ^
    --root_path %root_path% ^
    --seq_len %seq_len% ^
    --step %step% ^
    --feature_dim %large_feature_dim% ^
    --extractor_type optimized_dilated ^
    --fusion_type adaptive ^
    --attention_type channel ^
    --classifier_type mlp ^
    --use_statistical_features ^
    --stat_type basic ^
    --multimodal_fusion_strategy concat ^
    --train_epochs %epochs% ^
    --batch_size %batch_size% ^
    --learning_rate %learning_rate% ^
    --des "dilated_enhanced"

REM --loss_preset hvac_hard_samples ^

echo 实验1完成
echo --------------------------------

REM 实验2: 使用相关性聚焦的统计特征
REM echo 实验2: 相关性聚焦统计特征 + 注意力融合
REM python run.py ^
REM     --model %model% ^
REM     --data %data% ^
REM     --root_path %root_path% ^
REM     --seq_len %seq_len% ^
REM     --step %step% ^
REM     --feature_dim %feature_dim% ^
REM     --extractor_type dilated ^
REM     --fusion_type gated ^
REM     --attention_type channel ^
REM     --classifier_type mlp ^
REM     --use_statistical_features ^
REM     --stat_type basic ^
REM     --multimodal_fusion_strategy concat ^
REM     --train_epochs %epochs% ^
REM     --batch_size %batch_size% ^
REM     --learning_rate %learning_rate% ^
REM     --des "dilated_enhanced"

REM echo 实验2完成
REM echo --------------------------------

REM 实验3: 使用相关性聚焦的统计特征
REM echo 实验3: comprehensive统计特征 + 注意力融合
REM python run.py ^
REM     --model %model% ^
REM     --data %data% ^
REM     --root_path %root_path% ^
REM     --seq_len %seq_len% ^
REM     --step %step% ^
REM     --feature_dim %feature_dim% ^
REM     --extractor_type dilated ^
REM     --fusion_type bidirectional ^
REM     --attention_type cbam ^
REM     --classifier_type mlp ^
REM     --use_statistical_features ^
REM     --stat_type basic ^
REM     --multimodal_fusion_strategy concat ^
REM     --train_epochs %epochs% ^
REM     --batch_size %batch_size% ^
REM     --learning_rate %learning_rate% ^
REM     --des "dilated_enhanced"

REM echo 实验3完成
REM echo --------------------------------

REM 实验4: 使用相关性聚焦的统计特征
REM echo 实验4: 相关性聚焦统计特征 + 门控融合
REM python run.py ^
REM     --model %model% ^
REM     --data %data% ^
REM     --root_path %root_path% ^
REM     --seq_len %seq_len% ^
REM     --step %step% ^
REM     --feature_dim %feature_dim% ^
REM     --extractor_type dilated ^
REM     --fusion_type gated ^
REM     --attention_type cbam ^
REM     --classifier_type mlp ^
REM     --use_statistical_features ^
REM     --stat_type basic ^
REM     --multimodal_fusion_strategy concat ^
REM     --train_epochs %epochs% ^
REM     --batch_size %batch_size% ^
REM     --learning_rate %learning_rate% ^
REM     --des "dilated_enhanced"

REM echo 实验4完成
REM echo --------------------------------

REM echo 实验5: 基础统计特征 + 注意力融合
REM python run.py ^
REM     --model %model% ^
REM     --data %data% ^
REM     --root_path %root_path% ^
REM     --seq_len %seq_len% ^
REM     --step %step% ^
REM     --feature_dim %feature_dim% ^
REM     --extractor_type dilated ^
REM     --fusion_type adaptive ^
REM     --attention_type channel ^
REM     --classifier_type mlp ^
REM     --use_statistical_features ^
REM     --stat_type basic ^
REM     --multimodal_fusion_strategy attention ^
REM     --train_epochs %epochs% ^
REM     --batch_size %batch_size% ^
REM     --learning_rate %learning_rate% ^
REM     --des "dilated_enhanced"

REM echo 实验5完成
REM echo --------------------------------

echo 所有实验完成！
echo 结果保存在 ./result/ 目录下

pause 