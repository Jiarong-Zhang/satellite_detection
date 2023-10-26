INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=mobiledet_qat/train_mobiledet_qat.config
TRAINED_CKPT_PREFIX=mobiledet_qat/model.ckpt-40000
EXPORT_DIR=mobiledet_qat/export

python  /home/user/Documents/tensorflow/models/research/object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
