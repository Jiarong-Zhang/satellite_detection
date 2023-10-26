import tensorflow as tf
import pandas as pd
import os
from tqdm import tqdm

from object_detection.utils import dataset_util



image_path = 'train_img/'


def make_tfrecord(type):
    file = pd.read_csv(type + '.csv')

    writer = tf.io.TFRecordWriter(type + '.record')


    for i in tqdm(range(0, len(file))):

        with tf.io.gfile.GFile(os.path.join(image_path, file.loc[i].filename), 'rb') as fid:
            encoded_jpg = fid.read()

        
        image_format = b'jpg'
        filename = file.loc[i].filename.encode('utf8')
        width = file.loc[i].width
        height = file.loc[i].height

        xmin = file.loc[i].xmin
        xmax = file.loc[i].xmax
        ymin = file.loc[i].ymin
        ymax = file.loc[i].ymax
        
        if xmax > width:
            xmax = width
        
        if ymax > height:
            ymax = height
        
        xmins = [xmin / width]
        xmaxs = [xmax / width]

        ymins = [ymin / height]
        ymaxs = [ymax / height]
        classes_text = [b'Satellite']
        classes = [1]
        


        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))

        writer.write(tf_example.SerializeToString())

    writer.close()


make_tfrecord("train")
make_tfrecord("test")