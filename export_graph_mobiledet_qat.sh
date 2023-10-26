python /home/user/Documents/tensorflow/models/research/object_detection/export_tflite_ssd_graph.py \
  --pipeline_config_path="mobiledet_qat/train_mobiledet_qat.config" \
  --trained_checkpoint_prefix="mobiledet_qat/model.ckpt-40000" \
  --output_directory="tflite/mobiledet_qat" \
  --add_postprocessing_op=true
