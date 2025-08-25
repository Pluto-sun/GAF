export PYTHONPATH=$(cd "$(dirname "$0")/.."; pwd)
python scripts/visualize_ddahu_gaf_samples.py \
  --root_path ./dataset/SAHU/direct_5_working \
  --seq_len 96 \
  --step 96 \
  --num_samples 3 \
  --output_dir gaf_samples \
  --format svg