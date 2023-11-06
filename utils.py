import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

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

def decode(y_pred,df_box):
  y_preds = tf.squeeze(y_pred,axis=0)
  df_box = tf.cast(df_box,dtype=tf.float32)
  boxes = y_preds[:,:4]
  boxes = normalised_ground_truth(boxes,df_box,'decode')
  boxes_x1y1 = convert_format(boxes,'x1y1x2y2')
  y_preds = tf.nn.softmax(y_preds[:,4:])
  cls_idx = tf.argmax(y_preds, axis=-1)
  cls_scores = tf.reduce_max(y_preds, axis=-1)
  #Filter out the backgrund class
  foreground_idx = tf.where(cls_idx != 0)[:, 0]
  print(foreground_idx.shape)

  filtered_boxes = tf.gather(boxes_x1y1, foreground_idx)
  filtered_cls_idx = tf.gather(cls_idx, foreground_idx)
  filtered_cls_scores = tf.gather(cls_scores, foreground_idx)
  filtered_cls_idx = filtered_cls_idx-1

  filtered_boxes_y1x1 = tf.stack([filtered_boxes[:,1],
                                filtered_boxes[:,0],
                                filtered_boxes[:,3],
                                filtered_boxes[:,2]],axis=-1)
  nms_idx = tf.image.non_max_suppression(filtered_boxes_y1x1,
                                          filtered_cls_scores,
                                          max_output_size=200,
                                          iou_threshold=0.5,
                                          score_threshold=1e-2)
  final_boxes = tf.gather(filtered_boxes, nms_idx)
  final_cls_idx = tf.gather(filtered_cls_idx, nms_idx)
  final_cls_scores = tf.gather(filtered_cls_scores, nms_idx)
  return final_boxes, final_cls_idx, final_cls_scores

def visualize_detections(image, boxes, classes, scores):
  figsize=(7, 7)
  linewidth=1
  color=[0, 1, 0]
  image = np.array(image, dtype=np.uint8)
  plt.figure(figsize=figsize)
  plt.axis('off')
  plt.imshow(image)
  ax = plt.gca()
  for box, _cls, score in zip(boxes, classes, scores):
      text = '{}: {:.2f}'.format(_cls, score)
      x1, y1, x2, y2 = box
      w, h = x2 - x1, y2 - y1
      patch = plt.Rectangle([x1, y1], w, h, fill=False,
                            edgecolor=color, linewidth=linewidth)
      ax.add_patch(patch)
      ax.text(x1, y1, text, bbox={'facecolor':color, 'alpha':0.4},
              clip_box=ax.clipbox, clip_on=True)
