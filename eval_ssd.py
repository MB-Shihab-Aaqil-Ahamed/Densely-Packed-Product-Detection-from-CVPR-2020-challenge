import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2
import re

from datasets.data import pre_process_data
from datasets.data import create_df_box, convert_scale
from datasets.tfrecord import convert_tfrecord 
from .train_ssd import create_data
from models.ssd import ssd_model
from .utils import visualize_detections

# Pre-Processing image and label

val_path='/SKU110K_fixed/annotations/annotations_val.csv'
val_data=pre_process_data(val_path)

path='/SKU110K_fixed/annotations/annotations_train.csv'
data=pre_process_data(path)

test_path='/SKU110K_fixed/annotations/annotations_test.csv'
test_data=pre_process_data(test_path)

# TF RECORDS

out_path='/val.tfrecords'
convert_tfrecord(val_data.index,val_data['label'],out_path)

out_path='/train.tfrecords'
convert_tfrecord(data.index,data['label'],out_path)

out_path='/test.tfrecords'
convert_tfrecord(test_data.index,test_data['label'],out_path)

# Matched Boxes

df_box = create_data(data)

# ssd_model training

optimizer = tf.optimizers.Adam(1e-4)

model = ssd_model()

model.compile(optimizer=optimizer,
            loss=total_loss)

callback=tf.keras.callbacks.ModelCheckpoint(
        filepath='/ssd_model/ssd_model_{epoch:02d}.h5',
        monitor='loss',
        save_best_only=False,
        save_weights_only=True,
        mode='min',
        verbose=1)

step_per_epoch = len(data)//batch_size
val_steps = len(val_data)//batch_size

model.fit(train_dataset,           
          epochs=epochs,
          validation_data=val_dataset,
          steps_per_epoch=step_per_epoch,
          validation_steps=val_steps,
          callbacks=callback)
     
model.load_weights('/ssd_model/SKU110K_model_10.h5')

class_map = {
    1: 'product'
}
for i in range(20, 30):
  image,label = test_data.iloc[i].name,test_data.iloc[i]['label']
  root='/SKU110K_fixed/images'

  image_path=os.path.join(root,image)
  
  image = cv2.imread(image_path)
  image = cv2.resize(image, (hImage,wImage))

  image_ = tf.keras.applications.densenet.preprocess_input(image)

  label = main(np.array(label))

  predictions = model(image_[None, ...], training=False)
  feature_box=create_df_box(feature_layers)
  feature_box=convert_scale(feature_box,'abs')
  final_boxes, final_cls_idx, final_cls_scores = decode(predictions,feature_box)
  visualize_detections(image, final_boxes, final_cls_idx, final_cls_scores)
