# Computer Vision

## Autoregressive Image Modeling

- For autoregressive image modeling, we predict pixel by pixel in raster scan order. 
- Let's study the PixelCNN Model
- To prevent looking ahead, we use masked convolution kernels
  - ![masked_kernel.png](masked_kernel.png)[Source](https://arxiv.org/pdf/1606.05328)
  - However, these introduce blind spots, and so we instead use horizontal and vertical convolutions
    - ![blind_spot.png](blind_spot.png)[Source](https://arxiv.org/pdf/1606.05328)
    - Additional explanation
      - Blind spots for 3x3 masked filters: for every feature map (not on the boundary) at position (i,j) does not depend feature maps on (i-1,j+2) in the previous layer.
      - Horizontal stacks look left only, including current spot.
      - Vertical stacks look at all rows at/above the current spot.
        - Note that causality within the same row for vertical stack is not ensured
      - The trick to ensure overall causality is that i-th row in vertical stack will only be used in the computation of (i+1)-th row in horizontal stack.
- Loss Function
  - In PixelCNN, for each pixel, we output 256 logits for each possible pixel value, then compute cross entropy loss.
    - This is costly in terms of memory.
    - In addition, the model doesn't know that pixel value $x$ is close to $x-1$. 
  - In PixelCNN++, we instead assume that the output is a mixture of logistic distributions
    - $P(x \mid \pi, \mu, s)=\sum_{i=1}^K \pi_i\left[\sigma\left(\left(x+0.5-\mu_i\right) / s_i\right)-\sigma\left(\left(x-0.5-\mu_i\right) / s_i\right)\right]$
    - The model then tries to learn the mean and scale parameters for each mixture.

## Vision Transformers 

- Classification:
  - [Lippe's implementation](https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial15/Vision_Transformer.ipynb):
    - Split an image up into a sequence of $L$ image patches, each image patch is now a "token", where its embedding is a flattened vector of its pixel values. 
    - Add an additional classification token to each sequence, which is initialized with noise. The final embedding for this token is used for classification. 
    - We learn positional encodings which has $L \times Cp_Hp_W$ parameters. Comparing rows dictate the relationship between patches and comparing columns dictate hte relationship between pixels of the same patch. 
      - Convolutional kernels is the normal way we learn these relationships, although these have more restrictions (but are probably easier to learn).
- Generation
  - See [Diffusion](../10_diffusion/notes.md)

## Object Detection

- Bounding boxes vs image segmentation
  - Segmentation
    - Given per-pixel labelled data, 
      -  Segmentation labels regions on a pixel level.
    - Fully convolutional networks are useful here
      - ![fully_convolutional.png](fully_convolutional.png)[Source](http://d2l.ai/chapter_computer-vision/fcn.html)
      - One channel per class
  - Bounding boxes
    - Given images with "correct" bounding boxes, 
      - The goal of the detector is to propose bounding boxes that "looks similar" to the "correct" bounding boxes
        - Looks similar: The loss has two parts
          - We want the bounding boxes to be at the same place (localization error), e.g. l1 loss for the offset
          - We want to correctly predict the class of the object it encapsulates
        - Implicit in this is that we need to map the proposed bounding box to the ground truth bounding box
          - To do so, we [use the IOU metric](http://d2l.ai/chapter_computer-vision/anchor.html)
    - At inference, we use Non-Maximum Suppression to reduce the number of proposed bounding boxes
      - Iteratively keep the box with the highest predicted class probability 
      - Remove any box that predicts the same class, that overlaps heavily (IOU) with the selected box
- Two-Stage Detectors
  - Stage 1: Proposes a set of regions 
  - Stage 2: Classifier processes the region candidates
    - For each region candidate, we make a prediction of the class of the encapsulated object and the offset to best encapsulate this object.
  - [R-CNN and friends](https://lilianweng.github.io/posts/2017-12-31-object-recognition-part-3/)
      - ![rcnn.png](rcnn.png)[Source](https://lilianweng.github.io/posts/2017-12-31-object-recognition-part-3/)
      - Start with a pre-trained CNN network on image classification tasks, e.g. VGG or ResNet. 
      - R-CNN
        - Propose region proposals with selective search, which works by oversegmenting an image and iteratively grouping adjacent segments based on similarity 
        - Warp region proposals to fit CNN input
        - Fine-Tune CNN to generate a feature vector for each region proposal
        - Feature vector is consumed by: 
          - Binary SVMs trained for each class independently
          - A regression model to reduce localization errors
      - Fast R-CNN
        - CNN forward propagation performed on the entire image, rather than each proposed region.
          - Replaces the last max pooling layer of the pre-trained CNN with a RoI pooling layer, which outputs fixed-length feature vectors regardless of proposed region size. 
        - Uses a softmax estimator for classes instead of individual SVMs
      - Faster R-CNN
        - Uses a region proposal network rather than selective search. 
          - The region proposal network considers multiple regions of various scales and ratios for each $n \times n$ window that is sliding across the image.
            - Positive samples have IoU > 0.7 and negative samples have IoU < 0.3 
      - Mask R-CNN 
        - Introduces an additional fully convolutional network to leverage pixel-level labels to further improve the accuracy of object detection.
- Two-Stage Detector
  - Single Shot Multibox Detection (one stage)
    - Reuse a base network, and use multiscale feature maps to draw anchor boxes of different resolutions.
  - YOLO (one stage)
    - Splits image into $S \times S$ cells. 
    - For each cell, we predict $B$ bounding boxes of the same class and its confidence for each of the $C$ classes.
    - The output of YOLO is a tensor of $S \times S \times (B\times 5+C)$.
    - This can be followed by NMS to remove duplicate detections.

## Neural Style Transfer

- Leveraging pretrained networks
  - ![nst.png](nst.png)[Source](http://d2l.ai/chapter_computer-vision/neural-style.html)
  - Use pre-trained networks to: 
    - Extract content and style from reference images
      - In general, 
        - The closer to the input layer, the easier to extract details of the image (style).
        - The further from the input layer, the easier to extract the global information of the image.
    - Initialize the new generative network
  - Loss
    - Content loss + Style loss + Variation loss
    - We tend to use a gram matrix for style loss because it calculates correlations across features, focusing on the overall distribution of features rather than their exact locations.