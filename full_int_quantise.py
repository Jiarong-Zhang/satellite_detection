import tensorflow as tf

print(tf.__version__)
tf.compat.v1.enable_eager_execution() #required 
# help(tf.lite.TFLiteConverter)

IMAGE_SIZE = 320  # model expects images of 320 by 320

'''
# Full integer quantise the non-QAT model (Post-training Quantisation)
def representative_data_gen():
  # there were 50 images in the representative_imgs folder
  dataset_list = tf.data.Dataset.list_files("representative_imgs/*.jpg" )
  for image_path in dataset_list:
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    resized_img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    resized_img = resized_img[tf.newaxis, :]
    yield [resized_img]

# Both graphs were saved using "export_tflite_ssd_graph.py" from the tensorflow object detection API
GRAPH_DIR = "tflite/spaghettinet/tflite_graph.pb"

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file=GRAPH_DIR,
    input_arrays=['normalized_input_image_tensor'],
    output_arrays=['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'],
    input_shapes={'normalized_input_image_tensor': [1, IMAGE_SIZE, IMAGE_SIZE, 3]}
)
converter.allow_custom_ops = True
converter.representative_dataset = representative_data_gen  # must have representative dataset to do PTQ
converter.inference_input_type = tf.uint8
#converter.inference_output_type = tf.float32
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open('tflite/full_int_quant/spaghettinet_int8.tflite', 'wb') as f:
  f.write(tflite_model)

'''
# for QAT
model_to_be_quantized = "tflite/spaghettinet_qat/tflite_graph.pb"

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file=model_to_be_quantized,
    input_arrays=['normalized_input_image_tensor'],
    output_arrays=['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'],
    input_shapes={'normalized_input_image_tensor': [1, IMAGE_SIZE, IMAGE_SIZE, 3]},

)
converter.allow_custom_ops = True
converter.change_concat_input_ranges = True
converter.inference_type = tf.uint8 #<!!!!!!!
#converter.inference_input_type = tf.uint8
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINSIN, tf.lite.OpsSet.SELECT_TF_OPS]
#
converter.quantized_input_stats = {"normalized_input_image_tensor": (128, 128)}

tflite_model = converter.convert()

with open('tflite/full_int_quant/spaghettinet_qat_int8.tflite', 'wb') as f:
  f.write(tflite_model)
