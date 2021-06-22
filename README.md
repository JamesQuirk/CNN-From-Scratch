# CNN From Scratch

> **Status**: Draft

This 'CNN From Scratch' is a simple CNN framework that has been coded using only NumPy for matrix mathematics. This project was done to explore, in detail, how Convolutional Neural Networks work.

## Intro to CNNs

A Convolutional Neural Networks (CNN) is a deep learning algorithm that combines a fully connected neural network with convolutional and pooling layers that results in a spacially more efficent (less parameters) algorithm, often with improved performance. CNNs were developed by the computer vision community and are often used with images - though they can, in theory, be used for other types of input.

CNNs exhibite spacial-invariance - meaning they can often do much better than standard NNs when identifying features that are not confined to a specific region of the image.

The diagram below shows an example forwards flow of a CNN.

![CNN Forwards](./imgs/CNN_Forwards.png)

The architecture of a CNN is somewhat down to the developer and optimum choices can vary between use cases. However, they all start with some combination of 'Convolutional' and 'Pooling' layers and then end with a small number (usually) of fully connected layers.

## Convolutional Layer

In the convolutional layer, a number of filters (aka kernels/ masks) are systematically slid over the input image, from top left to bottom right. 

> [Illustration of convolution process.]

Each filter has the same number of channels as the input image and the layer output has equal numbers of channels as there are filters.


## Pooling Layer


## Fully Connected Layer


## Cost


## Optimisers


## The Model Object

















## Useful References

 - [TowardsDataScience - Convolutional Neural Networks from the ground up](https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1)

 - [TowardsDataScience - Applied Deep Learning](https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2)

 - [Convolutional Networks](https://cs231n.github.io/convolutional-networks/)

 - [MNIST database](http://yann.lecun.com/exdb/mnist/)

 - [SuperDataScience - The Ultimate Guide to Convolutional Neural Networks (CNN)](https://www.superdatascience.com/blogs/the-ultimate-guide-to-convolutional-neural-networks-cnn)
 
 - [Convolutions and Backpropagations](https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c)
 
 - [3Blue1Brown Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8)

- [ML Cheatsheet](https://ml-cheatsheet.readthedocs.io)

- [Medium - Backpropagation for Convolution with Strides [> 1, w.r.t. inputs]](https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710)

- [Medium - Backpropagation for Convolution with Strides [> 1, w.r.t. filters]](https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-fb2f2efc4faa)

- [On Vectorization of Deep Convolutional Neural Networks for Vision Tasks](http://lxu.me/mypapers/vcnn_aaai15.pdf)

