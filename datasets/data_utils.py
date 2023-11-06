import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2
import re
     
def convert_format(out,format):
  if format == 'x1y1x2y2':
    return tf.stack([out[...,0]-out[...,2]/2.0,
    out[...,1]-out[...,3]/2.0,
    out[...,0]+out[...,2]/2.0,
    out[...,1]+out[...,3]/2.0]
    ,axis=-1)

  elif format == 'xywh':
    return tf.stack([(out[...,0]+out[...,2])/2.0,
    (out[...,1]+out[...,3])/2.0,
    out[...,2]-out[...,0],
    out[...,3]-out[...,1],
    out[...,4]],axis=-1)  ##sending the class also

def convert_scale(matrix,scale):
  if scale == 'abs':
    return tf.stack([matrix[:,0]*wImage,
    matrix[:,1]*hImage,
    matrix[:,2]*wImage,
    matrix[:,3]*hImage],axis=-1)

  elif scale == 'rel':
    return tf.stack([matrix[:,0]/wImage,
    matrix[:,1]/hImage,
    matrix[:,2]/wImage,
    matrix[:,3]/hImage],axis=-1)

def normalised_ground_truth(matched_boxes,feature_box,return_format):
  matched_boxes = tf.cast(matched_boxes,dtype=tf.float32)
  feature_box = tf.cast(feature_box,dtype=tf.float32)
  if return_format == "encode":
    return tf.stack([(matched_boxes[:,0] - feature_box[:, 0]) / (feature_box[:, 2]),
                   (matched_boxes[:,1] - feature_box[:, 1]) / (feature_box[:, 3]),
        tf.math.log(matched_boxes[:,2] / feature_box[:, 2]),
        tf.math.log(matched_boxes[:,3] / feature_box[:, 3])],
        axis=-1) / [0.1, 0.1, 0.2, 0.2]

  elif return_format == "decode":
    matched_boxes *= [0.1, 0.1, 0.2, 0.2]
    return tf.stack([matched_boxes[:,0] * feature_box[:, 2] + (feature_box[:, 0]),
                    matched_boxes[:,1] * feature_box[:, 3] + (feature_box[:, 1]),
          tf.math.exp(matched_boxes[:,2]) * feature_box[:, 2],
          tf.math.exp(matched_boxes[:,3]) * feature_box[:, 3]],
          axis=-1)

def create_df_box(feature_layers):
#   s_min+(s_max-s_min)/(m-1)*(k-1)
#   s_min = 0.2
#   s_max = 0.9
#   m = 6

#   scale=[]
#   for k in range(2,8):
#     sk = s_min+(s_max-s_min)/(m-1)*(k-1)
#     scale.append(sk)
#   scale.insert(0,s_min)
#   scale.extend([s_max])

  scale =  [0.03, 0.05, 0.08, 0.12, 0.15, 0.25, 0.35]

  feature_boxes=[]
  for feature_layer in feature_layers:
    if (feature_layer == 128 or feature_layer == 14 or feature_layer == 12):
    #   aspect_ratios=[1,2/3,1/2]
      aspect_ratios=[0.333, 0.416, 1.401]

    else:
    #   aspect_ratios=[1,2/3,3/2,1/2,1/3]
      aspect_ratios=[0.416, 0.553, 0.722, 1.401, 3.131]

    x2_ar=[]
    y2_ar=[]
    for i in aspect_ratios:
      if int(i) == 1:
        x2=scale[feature_layers.index(feature_layer)]*np.sqrt(i)
        y2=scale[feature_layers.index(feature_layer)]/np.sqrt(i)
        x2_ar.append(x2)
        y2_ar.append(y2)
        sk_1 = np.sqrt(scale[feature_layers.index(feature_layer)]*
                     scale[feature_layers.index(feature_layer)+1])
        x2 = sk_1*np.sqrt(i)
        y2 = sk_1/np.sqrt(i)
      else:
        x2 = scale[feature_layers.index(feature_layer)]*np.sqrt(i)
        y2 = scale[feature_layers.index(feature_layer)]/np.sqrt(i)
      x2_ar.append(x2)
      y2_ar.append(y2)

    x_axis = np.linspace(0,feature_layer,feature_layer+1)
    y_axis=np.linspace(0,feature_layer,feature_layer+1)
    xx,yy = np.meshgrid(x_axis,y_axis)
    x = [(i+0.5)/(feature_layer) for i in xx[:-1,:-1]]
    y = [(i+0.5)/(feature_layer) for i in yy[:-1,:-1]]

    if (feature_layer == 128 or feature_layer == 14 or feature_layer == 12):
      ndf_box = 4
    else:
      ndf_box = 6
    ndf_boxes = feature_layer*feature_layer*ndf_box
    nbox_coordinates = 4
    feature_box = np.zeros((ndf_boxes,nbox_coordinates))
    x = np.array(x).reshape(feature_layer*feature_layer)
    x = np.repeat(x,ndf_box)
    y = np.array(y).reshape(feature_layer*feature_layer)
    y = np.repeat(y,ndf_box)

    x2_ar = np.tile(x2_ar,feature_layer*feature_layer)
    y2_ar = np.tile(y2_ar,feature_layer*feature_layer)
    feature_box[:,0] = x
    feature_box[:,1] = y
    feature_box[:,2] = x2_ar
    feature_box[:,3] = y2_ar
    feature_boxes.append(feature_box)
  df_box = np.concatenate(feature_boxes,axis=0)
  return df_box

def iou(box1,box2):
  box1 = tf.cast(box1,dtype=tf.float32)
  box2 = tf.cast(box2,dtype=tf.float32)

  x1 = tf.math.maximum(box1[:,None,0],box2[:,0])
  y1 = tf.math.maximum(box1[:,None,1],box2[:,1])
  x2 = tf.math.minimum(box1[:,None,2],box2[:,2])
  y2 = tf.math.minimum(box1[:,None,3],box2[:,3])

  #Intersection area
  intersectionArea = tf.math.maximum(0.0,x2-x1)*tf.math.maximum(0.0,y2-y1)

  #Union area
  box1Area = (box1[:,2]-box1[:,0])*(box1[:,3]-box1[:,1])
  box2Area = (box2[:,2]-box2[:,0])*(box2[:,3]-box2[:,1])

  unionArea = tf.math.maximum(1e-10,box1Area[:,None]+box2Area-intersectionArea)
  iou = intersectionArea/unionArea
  return tf.clip_by_value(iou,0.0,1.0)

def df_match(labels,iou_matrix):
  max_values = tf.reduce_max(iou_matrix,axis=1)
  max_idx = tf.math.argmax(iou_matrix,axis=1)
  matched = tf.cast(tf.math.greater_equal(max_values,0.5),
                  dtype=tf.float32)
  gt_box = tf.gather(labels,max_idx)
  return gt_box,matched

def pre_process_img(img,feature_box_conv,matched):
  img = cv2.imread(img)
  img = cv2.resize(img, (hImage,wImage), interpolation = cv2.INTER_AREA)
  color = (0,255,0)
  matched_idx = np.where(matched)

  for i in matched_idx:
    for j in i:
      start = tuple(tf.cast(feature_box_conv[j, :2], tf.int32).numpy())
      end = tuple(tf.cast(feature_box_conv[j, 2:4], tf.int32).numpy())
      cv2.rectangle(img, start, end, color, 2)

  plt.title('Matched Boxes')
  imshow(img)
