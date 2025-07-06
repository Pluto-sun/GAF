# 模块配置说明:
# extractor_type: large_kernel (大核卷积), inception (Inception网络), dilated/optimized_dilated (膨胀卷积), multiscale (多尺度)
# fusion_type: adaptive (自适应), concat (拼接), bidirectional (双向注意力), gated (门控融合), add (相加), mul (相乘), weighted_add (加权相加)
# attention_type: channel (通道注意力), spatial (空间注意力), cbam (CBAM), self (自注意力), none (无注意力)
# classifier_type: mlp (多层感知机), simple (简单分类器) 
# use_statistical_features: 是否使用统计特征
# stat_type: 'basic', 'comprehensive', 'correlation_focused'
# multimodal_fusion_strategy: 'concat', 'attention', 'gated', 'adaptive'

# 增强版双路GAF网络运行示例脚本
# 激活 conda 环境
& conda activate test_env  # 替换为你的环境名，例如 test_env

$env:CUDA_VISIBLE_DEVICES = "1"
Write-Host "增强版双路GAF网络实验脚本"
Write-Host "================================"

# 基础配置
$model = "DualGAFNet"
$data = "DualGAF_DDAHU"
$root_path = "./dataset/DDAHU/direct_5_working"
$epochs = 1000
$batch_size = 8
$learning_rate = 0.001
$step = 96
$seq_len = 96
$feature_dim = 64
$large_feature_dim = 128

# 实验1: 基础增强版网络（使用统计特征）
Write-Host "实验1: 基础增强版网络+concat"
python run.py `
    --model $model `
    --data $data `
    --root_path $root_path `
    --seq_len $seq_len `
    --step $step `
    --feature_dim $large_feature_dim `
    --extractor_type optimized_dilated `
    --fusion_type adaptive `
    --attention_type channel `
    --classifier_type mlp `
    --use_statistical_features `
    --stat_type basic `
    --multimodal_fusion_strategy concat `
    --train_epochs $epochs `
    --batch_size $batch_size `
    --learning_rate $learning_rate `
    --des "dilated_enhanced"

# --loss_preset hvac_hard_samples `

Write-Host "实验1完成"
Write-Host "--------------------------------"

# 实验2: 使用相关性聚焦的统计特征
<# 
Write-Host "实验2: 相关性聚焦统计特征 + 注意力融合"
python run.py `
    --model $model `
    --data $data `
    --root_path $root_path `
    --seq_len $seq_len `
    --step $step `
    --feature_dim $feature_dim `
    --extractor_type dilated `
    --fusion_type gated `
    --attention_type channel `
    --classifier_type mlp `
    --use_statistical_features `
    --stat_type basic `
    --multimodal_fusion_strategy concat `
    --train_epochs $epochs `
    --batch_size $batch_size `
    --learning_rate $learning_rate `
    --des "dilated_enhanced"

Write-Host "实验2完成"
Write-Host "--------------------------------"

# 实验3: 使用相关性聚焦的统计特征
Write-Host "实验3: comprehensive统计特征 + 注意力融合"
python run.py `
    --model $model `
    --data $data `
    --root_path $root_path `
    --seq_len $seq_len `
    --step $step `
    --feature_dim $feature_dim `
    --extractor_type dilated `
    --fusion_type bidirectional `
    --attention_type cbam `
    --classifier_type mlp `
    --use_statistical_features `
    --stat_type basic `
    --multimodal_fusion_strategy concat `
    --train_epochs $epochs `
    --batch_size $batch_size `
    --learning_rate $learning_rate `
    --des "dilated_enhanced"

Write-Host "实验3完成"
Write-Host "--------------------------------"

# 实验4: 使用相关性聚焦的统计特征
Write-Host "实验4: 相关性聚焦统计特征 + 门控融合"
python run.py `
    --model $model `
    --data $data `
    --root_path $root_path `
    --seq_len $seq_len `
    --step $step `
    --feature_dim $feature_dim `
    --extractor_type dilated `
    --fusion_type gated `
    --attention_type cbam `
    --classifier_type mlp `
    --use_statistical_features `
    --stat_type basic `
    --multimodal_fusion_strategy concat `
    --train_epochs $epochs `
    --batch_size $batch_size `
    --learning_rate $learning_rate `
    --des "dilated_enhanced"

Write-Host "实验4完成"
Write-Host "--------------------------------"

Write-Host "实验5: 基础统计特征 + 注意力融合"
python run.py `
    --model $model `
    --data $data `
    --root_path $root_path `
    --seq_len $seq_len `
    --step $step `
    --feature_dim $feature_dim `
    --extractor_type dilated `
    --fusion_type adaptive `
    --attention_type channel `
    --classifier_type mlp `
    --use_statistical_features `
    --stat_type basic `
    --multimodal_fusion_strategy attention `
    --train_epochs $epochs `
    --batch_size $batch_size `
    --learning_rate $learning_rate `
    --des "dilated_enhanced"

Write-Host "实验5完成"
Write-Host "--------------------------------"
#>

Write-Host "所有实验完成！"
Write-Host "结果保存在 ./result/ 目录下"

# 等待用户按键后结束（类似于 pause）
Read-Host "按任意键继续..." 