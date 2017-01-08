# CarND-BehaviorCloning-P3
This is the third project for Udacity's Self-Driving Car program: Behavioral Cloning

## Overview
This is my convolutional neural network for the third project in [Udacity's Self-Driving Car program](https://www.udacity.com/drive). The objective of the network is to predict steering angles in a simulator developed by Udacity. The model is based on [NVidia's architecture] (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) for determining steering angles from camera images. It is built with [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/?__s=baq6agagypi9xbmhddyb) is used as the backend.

I used Udacity's training data although at first I used all of my own collected data. I switched to Udacity's own data to be sure I had a good sample of data for training as I learned this would be the most important part to training my model. The center, left, and right images were all used to train the model. The left and right images steering angles were adjusted by adding +.25 and -.25 respectively.

I've been through several iterations with the modified NVidia architecture and [Comma.ai's] (https://github.com/commaai/research) as well many of my own with varying success on the results. I finally settled on using the NVidia model and adjusting from there to get an optimized result. In the process I turned every knob probably that can be adjusted with a neural network. With the model finalized I adjusted batch size, number of epochs, learning rates, and the size of the training set with different augmentation techniques. My best results so far (I plan to keep experimenting with everything) have been with a batch size of 64, 100 epochs, an Adam Optimizer with learning rate at .004, and the use of several augmentation/preprocessing techniques.

##Augmentation and Preprocessing
A generator is used to process the images and augment before being fitted to the model. This helps to prevent overfitting by adjusting the image through each loop in the generator so that the probability that any two images are exactly alike are very small.

Images are preprocessed for the model first by converting to RGB color and cropping the top 20% and bottom 25 pixels. This removes some of the horizon and the inside of the car from the image.  The image is then resized to 66 x 200 for the model to evaluate. I then read [Vivek Yadav's augmentation techniques](https://chatbotslife.com/learning-human-driving-behavior-using-nvidias-neural-network-model-and-image-augmentation-80399360efee#.7nbpge45u) which included changing the tint of the image (the brightness) and translating the image a small amount and adding an adjustment to the steering angle proportional to the translation of the image. This helped make a big improvement to the model's accuracy so I was very grateful for these techniques and Vivek's help in sharing them with everyone in the Udacity program.

The next augmentation technique I used was to flip the image around it's center y-axis and change the sign of the angle. This gives an equal amount of images driving in the reverse direction around the track. In the preprocessing I flip the images half of the time to balance the data closer to a normal distribution. In addition if the steering angle is 0 then it is removed from the set half of the time to reduce the bias to going straight rather than turning at the curves.

In previous iterations I converted the images to YUV in preprocessing but then learned that this can be done in effect with a convolutional layer in the model with 3 1x1 filters. This does not convert to YUV but does change the color space to be optimized for the neural network. This was one of the most fascinating parts of Vivek Yadav's process to me and I want to explore why changing the color space makes such a difference in the learning process of the network.

Normalization of the images is also done in the model through a lambda layer at the start.

## Model architecture
The model architecture is nearly identical to NVidia's with the exception of an extra convolutional layer to optimize color space.

The NVidia Architecture:
![NVidia Architecture for Determining Steering Angles][https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture.png]

The architecture consists of the lambda layer for normalization; a single convolutional layer with 3 filters of size 1 x 1 to transform the color space; three convolutional layers with 24, 36, and 48 5 x 5 filters and stride of 2 respectively; two convolutional layers of 64 3 x 3 filters and stride of 1; and finally three fully connected layers to output a single floating point number to represent the predicted steering angle. The model is fitted using an Adam optimizer with learning rate .0001. I experimented with several batch sizes, epoch counts, and learning rates before finding this one that worked best for me which turned out to be a batch size of 64, 100 epochs, and learning rate of .0001.

After each epoch I saved the weights so that I could go through the different iterations and see which weights best generalized the correct steering angle.

## Performance on Track 1
The performance for the first track where the car is trained is not optimal but does keep the car on the track consistently even at its top speed of 30mph. One technique I also used was to divide the throttle by a multiple of the absolute value of the steering angle (plus a little more in case of zero division). This has the effect of slowing the car at the curves once the steering angle become large. The shortcoming of doing this is that the car slows down coming out of turns rather than prior to turning. Its more important to slow the car before turning in the simulator but much more so in real life where gravity, friction, and momentum mean much more to speed and the tires' ability to maintain grip. By slowing the throttle at curves we can allow the model to better process the correct the steering angle in time and correct the car's trajectory before it goes off the road.

##Performance on Track 2
By adjusting the maximum throttle on the car I was able to get the car to accelerate enough and drive on the second track. The car made it through several turns but did eventually run into the side of the mountain. By incorporating more data augmentation and perhaps some training on the second track I hope to make adjustments to get through all of the track eventually. I also plan to experiment with more models and experiment with my own to see the results.

##Conclusions and Reflection
This was a very challenging project, and, as a result, hugely rewarding. I have learned a lot about convolutional neural networks and their application in self-driving cars within a relatively short amount of time. One of the most important things I have learned is the importance of good data. A large amount of data that is well-distributed (a Normal distribution) has really been the key to my model learning enough to solve the problem. In the future I plan to always explore the data for the given problem and make sure that techniques like augmentation can be used to best generalize the problem for the network to solve. I have also learned the importance and experimentation in the process. It has been very interesting reading all the different solutions that have worked for people as there is a huge variety of them. There's an endless amount of knobs to be turned to optimize these networks and I would like to keep experimenting to see the results I can achieve through the different combinations.
