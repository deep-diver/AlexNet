# AlexNet
Implementation of AlexNet from ILSVRC-2012 Competition.

![AlexNet Architecture](./figure1.png)

## Resources
- **main.py**
  - providing only codes
  - single_gpu_convnet()
  - multi_gpu_convnet() is not supported yet

- **AlexNet.pdf**
  - my own summarization focused on implementation detail

- **AlexNet.ipynb**
  - Experimental workflow code on CIFAR-10 dataset

- **External Checkpoint files**
  - providing pre-trained checkpoint file on CIFAR-10 dataset
  - [Download Link](https://drive.google.com/drive/folders/1-bUYAWx6dQ8b5Nw6O_juvZwnNVk-M1Qu?usp=sharing)

## Things to include (or Find APIs if available)
- **Multi GPUs (Not Implemented Yet)**
   * > with tf.device('/gpu:*'):

- **Layers**
  - **Overlapping Pooling Layer**
     * (TensorFlow) [tf.contrib.layers.max_pool2d](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/max_pool2d)

  - **Convolutional Layer**
       * (TensorFlow) [tf.contrib.layers.conv2d](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv2d)

  - **Fully Connected Layer**
      * (TensorFlow) [tf.contrib.layers.fully_connected](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected)

- **Techniques**
  - **Rectified Linear Unit (ReLU) Activation Function**
     * (TensorFlow) [tf.nn.relu](https://www.tensorflow.org/api_docs/python/tf/nn/relu)

  - **Local Response Normalization**
     * (TensorFlow)  [tf.nn.local_response_normalization](https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization)

  - **Dropout**
     * (TensorFlow) [tf.nn.dropout](https://www.tensorflow.org/api_docs/python/tf/nn/dropout)

## Overall Architecture
**1. Input Layer of Image Size (224 x 224 x 3)**

**2. Convolutional Layer (96 x (11 x 11 x 3)) + stride size of 4**
   - Bias with constant value of 1
   - ReLU Activation
   - Local Response Normalization
   - Max Pooling (Overlapping Pooling)

**3. Convolutional Layer (256 x (5 x 5 x 48))**
   - ReLU Activation
   - Local Response Noramlization
   - Max Pooling (Overlapping Pooling)

**4. Convolutional Layer (384 x (3 x 3 x 128))**
   - Bias with constant value of 1

**5. Convolutional Layer (384 x (3 x 3 x 192))**
   - Bias with constant value of 1

**6. Convolutional Layer (256 x (3 x 3 x 192))**
   - Max Pooling (Overlapping Pooling)

**7. Fully Connected Layer (4096)**
   - Bias with constant value of 1
   - Dropout

**8. Fully Connected Layer (4096)**
   - Bias with constant value of 1
   - Dropout

**9. Fully Connected Layer (1000)**

## Training
- **Optimizer (Paper)**
  - SGD (momentum: 0.9, decay: 0.0005)

- **Optimizer (Implementation)**
  - AdamOptimizer

## Experiment on CIFAR-10 dataset
- **Environment**
  - [Floydhub](https://www.floydhub.com/) GPU2 instance (Tesla V100)

- **Approximate running time**
  - 1 hour 45 mins

- **Hyperparameters**
  - Learning rate: 0.00005
  - Epochs: 20
  - Batch size: 64

- **Test Accuracy: 0.6548566878980892**

![Experiment Result](./experiment.png)

## References
- [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (Original Paper)
- [Summary Slide](http://cvml.ist.ac.at/courses/DLWT_W17/material/AlexNet.pdf)
- [Summary Slide 2](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf)
- Local Response Normalization
  - https://stats.stackexchange.com/questions/145768/importance-of-local-response-normalization-in-cnn
