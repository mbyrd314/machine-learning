# machine-learning
Simple machine learning experiments

The first is an attempted implementation of the DCGAN architecture. Currently, it has an architecture based on the PyTorch
tutorial for DCGANs. It seems to often generate images that look vaguely like the top-right to bottom-left diagonal that
is often seen in shoes in the dataset. It occasionally changed to generating other images over the course of 10,000
training epochs, but it never generated images that were close to as clear as those in the Fashion MNIST dataset. More work
will have to be done to figure out how to train it effectively. I have also experimented with increasing the number of feature
maps in the convolutional layers and changing the kernel size, but I have yet to generate samples that could be confused
with the dataset.

The original plan of this was to first train a relatively simple GAN on images to get some intuition for training GANs and
then progress to using GANs to generate natural language. Much of the existing work in this domain uses reinforcement
learning to overcome the problem of generating discrete outputs instead of differentiable ones, so I will have to learn about
reinforcement learning before I am able to progress to text generation.
