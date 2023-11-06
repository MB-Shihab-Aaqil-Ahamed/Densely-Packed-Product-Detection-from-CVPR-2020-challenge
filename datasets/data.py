import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

def pre_process_data(path):
  colnames=['image_name','x1','y1','x2','y2','class','image_width','image_height']

  annotate_data=pd.read_csv(path, names=colnames)
  tokenizer = tf.keras.preprocessing.text.Tokenizer()
  tokenizer.fit_on_texts(annotate_data['class'])
  idx=tokenizer.word_index
  print(idx)
  annotate_data = annotate_data.astype({"x1": float, "x2": float, "y1": float, "y2": float})

  #Resize bb according to image
  annotate_data['x1'] = (annotate_data['x1'] * wImage) / annotate_data['image_width']
  annotate_data['y1'] = (annotate_data['y1'] * hImage) / annotate_data['image_height']
  annotate_data['x2'] = (annotate_data['x2'] * wImage) / annotate_data['image_width']
  annotate_data['y2'] = (annotate_data['y2'] * hImage) / annotate_data['image_height']

  annotate_data['class'] = idx['object']
  annotate_data['label'] = annotate_data[['x1','y1','x2','y2','class']].to_numpy().tolist()


  #Converting to x,y,w,h
  annotate_data['x1'] = (annotate_data['x1'] + annotate_data['x2']) / 2.0
  annotate_data['y1'] = (annotate_data['y1'] + annotate_data['y2']) / 2.0
  annotate_data['x2'] = annotate_data['x2'] - annotate_data['x1']
  annotate_data['y2'] = annotate_data['y2'] - annotate_data['y1']


  annotate_data['boxes_xywh'] = annotate_data[['x1','y1','x2','y2','class']].to_numpy().tolist()
  annotate_data = annotate_data.groupby('image_name').aggregate(lambda tdf: tdf.tolist())

  return annotate_data

def read_img(img):
  with tf.io.gfile.GFile(img, 'rb') as fp:
    image = fp.read()
  return image

def imshow(image):
  plt.figure(figsize=(8, 8))
  plt.imshow(image)

def show_img(img,label):
  img = cv2.imread(img)
  color = (0,255,0)
  img = cv2.resize(img,(hImage,wImage))
  for i,val in enumerate(label):
    start = tuple((np.array(label[i][:2])).astype('int'))
    end = tuple((np.array(label[i][2:4])).astype('int'))
    cv2.rectangle(img,start,end,color,2)
  imshow(img)
