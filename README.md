# AlexNet
Implementation of AlexNet from ILSVRC-2012 Competition.

![AlexNet Architecture](./figure1.png)

## Things to Implement (or Find APIs if available)
- Multi GPUs
   * > with tf.device('/gpu:*'):
- Rectified Linear Unit (ReLU) Activation Function
   * (TensorFlow) [tf.nn.relu](https://www.tensorflow.org/api_docs/python/tf/nn/relu)
- Local Response Normalization Technique
   * (TensorFlow) [tf.nn.local_response_normalization](https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization)
- Overlapping Pooling
   * (TensorFlow) [tf.nn.max_pool](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool)
- Overall Architecture

## Overall Architecture
1. Input Layer of Image Size (224 x 224 x 3)
2. Convolutional Layer (96 x (11 x 11 x 3)) + stride size of 4
   - Local Response Normalization
   - Max Pooling (Overlapping Pooling)
3. Convolutional Layer (256 x (5 x 5 x 48)) 
   - Local Response Noramlization
   - Max Pooling (Overlapping Pooling)
4. Convolutional Layer (384 x (3 x 3 x 128))
5. Convolutional Layer (384 x (3 x 3 x 192))
6. Convolutional Layer (256 x (3 x 3 x 192))
7. Fully Connected Layer (4096)
8. Fully Connected Layer (4096)
9. Fully Connected Layer (1000)

Total of 8 Layers (except the input layer)

## References
- [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (Original Paper)
- [Summary Slide](http://cvml.ist.ac.at/courses/DLWT_W17/material/AlexNet.pdf)
- Local Response Normalization
  - https://stats.stackexchange.com/questions/145768/importance-of-local-response-normalization-in-cnn