import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import cv2
import re

from datasets.tfrecord import convert_tfrecord
from models.ssd import ssd_model

#Matched Boxes
def create_data(data):

  i = 200
  images,labels = data.iloc[i].name,data.iloc[i]['label']
  # labels  x1y1x2y2
  root='/content/SKU110K_fixed/images'

  images=os.path.join(root,images)

  #GT boxes creation
  img = images
  label = labels
  show_img(img,label)
  feature_box = create_df_box(feature_layers)
  feature_box = convert_scale(feature_box,'abs')
  feature_box_conv = convert_format(feature_box,'x1y1x2y2')
  iou_matrix = iou(feature_box_conv,np.array(label)[:,:4])
  gt_box,matched = df_match(convert_format(np.array(label),'xywh'),iou_matrix)

  # gt_box xywh
  print(tf.math.count_nonzero(matched))
  pre_process_img(img,convert_format(feature_box,'x1y1x2y2'),matched)
  boxes=gt_box[:,:4]
  classes = gt_box[:,4]
  classes = tf.cast(classes+1, dtype=tf.int32) #0 for background class
  matched = tf.cast(matched,dtype=tf.int32)
  classes = tf.cast(classes*matched,dtype=tf.int32)
  classes = tf.one_hot(classes,depth=nClasses+1,dtype=tf.float32)
  normalised_gtbox = normalised_ground_truth(boxes,feature_box,'encode')
  normalised_gtbox = normalised_ground_truth(normalised_gtbox,feature_box,'decode')
  df_box = tf.concat((normalised_gtbox,classes),axis=-1)
  return df_box

def main(label):
  iou_matrix = iou(feature_box_conv,label)
  gt_box,matched = df_match(convert_format(label,'xywh'),iou_matrix)
  boxes = gt_box[:,:4]
  classes = gt_box[:,4]
  
  classes = tf.cast(classes+1, dtype=tf.int32) #0 for background class
  matched = tf.cast(matched,dtype=tf.int32)
  classes = tf.cast(classes*matched,dtype=tf.int32)
  classes = tf.one_hot(classes,depth=nClasses+1,dtype=tf.float32)
  normalised_gtbox = normalised_ground_truth(boxes,feature_box,'encode')
  df_box = tf.concat((normalised_gtbox,classes),axis=-1)
  df_box.set_shape([feature_box.shape[0], 4+nClasses+1])
  return df_box

def convert_back(serialized):
  feature={
      'image':tf.io.FixedLenFeature([],tf.string),
      'x1':tf.io.VarLenFeature(tf.float32),
      'y1':tf.io.VarLenFeature(tf.float32),
      'x2':tf.io.VarLenFeature(tf.float32),
      'y2':tf.io.VarLenFeature(tf.float32),
      'class':tf.io.VarLenFeature(tf.float32)
  }
  parsed_example = tf.io.parse_single_example(serialized=serialized,
                                            features=feature)
  img = tf.io.decode_image(parsed_example['image'],channels=3)
  img.set_shape([None,None,3])
  img = tf.image.resize(img,[hImage, wImage])
  img = tf.cast(img,tf.float32)
  # normalize image
  img = tf.keras.applications.densenet.preprocess_input(img)


  label=tf.stack([tf.sparse.to_dense(parsed_example['x1']),
            tf.sparse.to_dense(parsed_example['y1']),
            tf.sparse.to_dense(parsed_example['x2']),
            tf.sparse.to_dense(parsed_example['y2']),
            tf.sparse.to_dense(parsed_example['class']) - 1],axis=-1)
  # label
  df_box = main(label)
  return img, df_box


def data_gen(files):
  autotune = tf.data.experimental.AUTOTUNE
  dataset = tf.data.TFRecordDataset(filenames=files)
  dataset = dataset.map(convert_back, num_parallel_calls=autotune)
  dataset = dataset.apply(tf.data.experimental.ignore_errors())
  dataset = dataset.shuffle(16)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.repeat(epochs)
  dataset = dataset.prefetch(autotune)
  return dataset

def total_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    pos_mask = tf.cast(tf.equal(tf.squeeze(y_true[:, :, 4:5], axis=-1), 0.0),
                       tf.float32)
    num_pos = tf.maximum(
        1.0, tf.cast(tf.math.count_nonzero(pos_mask, axis=-1), tf.float32))
    loc_loss = tf.compat.v1.losses.huber_loss(labels=y_true[:, :, :4],
                                              predictions=y_pred[:, :, :4],
                                              reduction="none")

    loc_loss = tf.reduce_sum(loc_loss, axis=-1)
    loc_loss = tf.where(tf.equal(pos_mask, 1.0), loc_loss, 0.0)
    loc_loss = tf.reduce_sum(loc_loss, axis=-1)
    loc_loss = loc_loss / num_pos

    cce = tf.losses.CategoricalCrossentropy(from_logits=True,
                                            reduction=tf.losses.Reduction.NONE)
    cross_entropy = cce(y_true[:, :, 4:], y_pred[:, :, 4:])

    #neg:pos 3:1
    num_neg = 3.0 * num_pos

    #Negative Mining
    neg_cross_entropy = tf.where(tf.equal(pos_mask, 0.0), cross_entropy, 0.0)
    sorted_dfidx=tf.cast(tf.argsort(neg_cross_entropy,\
                            direction='DESCENDING',axis=-1),tf.int32)
    rank = tf.cast(tf.argsort(sorted_dfidx, axis=-1), tf.int32)
    num_neg = tf.cast(num_neg, dtype=tf.int32)
    neg_loss = tf.where(rank < tf.expand_dims(num_neg, axis=1),
                        neg_cross_entropy, 0.0)

    pos_loss = tf.where(tf.equal(pos_mask, 1.0), cross_entropy, 0.0)
    clas_loss = tf.reduce_sum(pos_loss + neg_loss, axis=-1)
    clas_loss = clas_loss / num_pos
    totalloss = loc_loss + clas_loss
    return totalloss


