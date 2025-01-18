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
    - We learn positional encodings which has $L \times C \times p_H \times p_W$ parameters. Comparing rows dictate the relationship between patches and comparing columns dictate the relationship between pixels of the same patch. 
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
          - The region proposal network considers multiple anchor boxes of various scales and ratios for each $n \times n$ window that is sliding across the image.
            - Positive samples have IoU > 0.7 and negative samples have IoU < 0.3 
      - Mask R-CNN 
        - Introduces an additional fully convolutional network to leverage pixel-level labels to further improve the accuracy of object detection.
- One-Stage Detector
  - Faster and simpler, but might potentially drag down performance
  - YOLO 
    - Splits image into $S \times S$ cells. 
    - If an object’s center falls into a cell, that cell is “responsible” for detecting the existence of that object.
    - For each cell, we predict $B$ bounding boxes of the same class and its confidence for each of the $C$ classes.
    - The output of YOLO is a tensor of $S \times S \times (B\times 5+C)$.
  - Single Shot Multibox Detection (SSD)
    - Uses predefined anchor boxes for every location of the feature map. 
    - Feature maps at different levels have different receptive field sizes. Intuitively, large fine-grained feature maps at earlier levels are good at capturing small objects and small coarse-grained feature maps can detect large objects well.
  - YOLOv2
    - Builds on YOLO
      - Adds BatchNorm
      - Fine-tuning the base model with high resolution images improves the detection performance
      - YOLOv2 uses convolutional (instead of fc) layers to predict the location of anchor boxes 
      - YOLOv2 runs k-mean clustering on the training data to find good priors on anchor box dimensions
      - YOLOv2 formulates the bounding box prediction in a way that it would not diverge from the center location too much
      - Adds residual connections
      - Multi-scale training: a new size of input dimension is randomly sampled every 10 batches
      - A lighter weight base model is used
  - RetinaNet
    - Focal loss is designed to assign more weights on hard, easily misclassified examples, and to down-weight easy examples. 
      - Let $p_t = p$ if $y=1$ and $1-p$ otherwise. ($CE(p_t)=-\log p_t$)
      - $FL(p_t)=-(1-p_t)^\gamma\log p_t$
    - Uses a featurized image pyramid: similar intuition as SSD where we get different receptive field sizes.
  - YOLOv3
    - Changes
      - Logistic regression for confidence scores
      - Multiple independent logistic classifier for each class rather than one softmax layer
      - Darknet + ResNet as the base model:
      - Multi-scale prediction: Adds several convolutional layers and makes prediction at three different scales among these conv layers (image pyramid-like)
      - Skip-layer concatenations

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