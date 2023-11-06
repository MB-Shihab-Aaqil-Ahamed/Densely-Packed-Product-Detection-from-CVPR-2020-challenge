import tensorflow as tf
import numpy as np
import os

def wrap_bytes(img):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))

def wrap_float(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def convert_tfrecord(images,labels,out_path):
  root='/SKU110K_fixed/images'
  with tf.io.TFRecordWriter(out_path) as writer:
    for i in range(len(images)):
      image=os.path.join(root,images[i])
      img_bytes=read_img(image)
      sku={
            'image':wrap_bytes(img_bytes),
            'x1':wrap_float(np.array(labels[i])[:,0]),
            'y1':wrap_float(np.array(labels[i])[:,1]),
            'x2':wrap_float(np.array(labels[i])[:,2]),
            'y2':wrap_float(np.array(labels[i])[:,3]),
            'class':wrap_float(np.array(labels[i])[:,4])
      }
      feature=tf.train.Features(feature=sku)
      example=tf.train.Example(features=feature)
      serialized=example.SerializeToString()
      writer.write(serialized)
