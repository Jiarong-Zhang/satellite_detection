python /home/user/Documents/tensorflow/models/research/object_detection/export_tflite_ssd_graph.py \
  --pipeline_config_path="spaghettinet_qat/train_spaghettinet_qat.config" \
  --trained_checkpoint_prefix="spaghettinet_qat/model.ckpt-40000" \
  --output_directory="tflite/spaghettinet_qat" \
  --add_postprocessing_op=true
