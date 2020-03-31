# machine-learning
Simple machine learning experiments

The first is an attempted implementation of the DCGAN architecture. Currently, it has an architecture based on the PyTorch
tutorial for DCGANs. It seems to often generate images that look vaguely like the top-right to bottom-left diagonal that
is often seen in shoes in the dataset. It occasionally changed to generating other images over the course of 10,000
training epochs, but it never generated images that were close to as clear as those in the Fashion MNIST dataset. More work
will have to be done to figure out how to train it effectively. I have also experimented with increasing the number of feature
maps in the convolutional layers and changing the kernel size, but I have yet to generate samples that could be confused
with the dataset. I have also just trained a fully-connected GAN on MNIST, but it doesn't produce anything distinguishable
from noise. It seems that the discriminator trains too quickly and that then the generator can't learn to produce more 
convincing images.

The comma_speedchallenge directory contains code to solve Comma.ai's speed challenge to infer vehicle speeds from dash camera video frames.

