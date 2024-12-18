# Object Detection

- One-stage vs Two-stage
  - Two-stage
    - Region proposal network
    - Classifier
    - Is more accurate and is used when we can accommodate slower speeds. 
- Loss
  - $\frac{1}{n} \sum_{i=1}^n\left[\left(x_i-\hat{x}_i\right)^2+\left(y_i-\hat{y_i}\right)^2+\left(w_i-\hat{w_i}\right)^2+\left(h_i-\hat{h}_i\right)^2\right]$
- Offline Metrics
  - Precision needs a concept of when your bounding box is correct: $IOU = \frac{Overlap Area}{UnionArea} \geq x$
- NMS
  - NMS is a post-processing algorithm designed to select the most appropriate bounding boxes. It keeps highly confident bounding boxes and removes overlapping bounding boxes.