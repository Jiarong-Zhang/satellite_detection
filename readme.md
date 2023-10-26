# satellite_detection
- the environment setup guide can be found [here](https://gist.github.com/Jiarong-Zhang/6fc37d3d7af6dabbf17c57c8c861e779)

- The entire dataset was taken from [SPEED](https://purl.stanford.edu/dz692fn7184)'s `train` split (`speed/images/train`)

- Ground truth data (`bbox.json`) was taken from [here](https://github.com/BoChenYS/satellite-pose-estimation/tree/master), then converted to a `.csv`

- I split the dataset into train/test using [`split_train_test.py`](/split_train_test.py)

- `tfrecord` files for the train and test set can be generated using [`generate_tfrecord.py`](/generate_tfrecord.py)

- Optionally, move images in the test split into its own folder using [`extract_test_set.py`](/extract_test_set.py)

- The training configs and checkpoints for each model are in their respective folders
	- [`/mobiledet`](/mobiledet) (PTQ)
	- [`/mobiledet_qat`](/mobiledet_qat)
	- [`/spaghettinet_qat`](/spaghettinet_qat)

- The `tflite` graphs and files for each model are in their respective folders in [`/tflite`](/tflite)
	- the full integer quantised & Edge TPU compiled files are in[ `/tflite/full_int_quant`](/tflite/full_int_quant)

- Training was executed through the [`retrain_od_gpu`](retrain_od_gpu.sh) script.
