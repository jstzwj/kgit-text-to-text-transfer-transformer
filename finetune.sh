MODEL_DIR="/data/wangjun/github/kbqat/models/unifiedqa-v2-t5-large-1363200/"
DATA_DIR=""
t5_mesh_transformer  \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="dataset.gin" \
  --gin_param="utils.run.mesh_shape = 'model:2,batch:2'" \
  --gin_param="utils.run.mesh_devices = ['gpu:0','gpu:1','gpu:2','gpu:3']" \
  --gin_param="MIXTURE_NAME = 'glue_mrpc_v002'" \
  --gin_file="gs://t5-data/pretrained_models/small/operative_config.gin"
