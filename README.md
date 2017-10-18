# CarND_Project03BehavioralCloning
Solution for Udacity Self driving car nano degree third project: Behavioral Cloning

---

Behavioral Cloning

---

### Reflection

Udacity Self Driving Car Nanodegree third project is to train a neural network to clone the behavior of a human steering a car in a simulated environment using an image produced by a camera mounted in the middle of the car front as input and generating needed steering wheel angle to stay in the middle of the road.

Speed and Brake signals were out of scope for the neural network, they were generated using a PI controller to keep the car at a constant previously set speed.

Before discussing the solution to achieve this goal, it is important to understand the problem and how to actually achieve the needed output.

---

### Car Simulator

[//]: # (Image References)

[TI1]: ./TrainingImages/Track1/center_2017_10_18_01_23_56_585.jpg
[TI2]: ./TrainingImages/Track1/center_2017_10_18_01_24_09_922.jpg
[TI3]: ./TrainingImages/Track1/center_2017_10_18_01_24_48_608.jpg
[TI4]: ./TrainingImages/Track1/center_2017_10_18_01_25_01_397.jpg
[TI5]: ./TrainingImages/Track2/center_2017_10_18_01_43_29_162.jpg
[TI6]: ./TrainingImages/Track2/center_2017_10_18_01_43_46_147.jpg

---

To solve this problem we need a lot of labeled training data, i.e. label here refers to precise steering angle, to correctly train the neural network.

To collect this data, udacity team has developed a simple car simulator based on the Unity game engine, it has two tracks and the car features 3 cameras (left, right and center).

The tracks have a lot of Varity in curvature, road texture, elevation, road side and scenery which helps in collected a dataset with a large variance in inputs.

Below is a sample of the two tracks:

![alt text][TI1]
![alt text][TI2]
![alt text][TI3]
![alt text][TI4]
![alt text][TI5]
![alt text][TI6]

---

File Structure:

I have separated my implementation to separate files for easier documentation and ease of readability:

| File               |     Description                                                                  |
|:------------------:|:--------------------------------------------------------------------------------:|
| model.py           | Contains the main function, CSV processing, and image preprocessing functions.   |
| KerasModel.py      | Contains the Neural Network Implementation in Keras                             |
| HelperFncs.py      | contains some subroutines that I used in the other two files.                    |

---

###Collecting Training Data:

Using the Udacity simulator and the suggestions in the lectures I made some trials the one that I settled for and gave me the best results were:
  * Three Forward laps at track 1, I varied the speed to make the module able to react to different speeds. and for one of the laps I made sure to go extra slow at the turns to make sure there are a lot of images with large steering angles.
  * two backward laps at track 1, to try and remove any turning bias the track has. By going the opposite direction we are basically introducing images with exactly counter the track bias so they should help network learn more general approach than memorize the model.
  * two forward laps at track 2, also to help and have a lot of images with steering angles, I decided to stay in the middle of a lane(left for first, right for second) for the laps, opposed to the middle of the road in track 1.

Using those approaches I was able to collect 30,888 data points that I think have a very large variance in features and steering angles.

---

###Data Preprocessing

##Image Preprocessing

From previous udacity projects, I have seen that good preprocessing of the input images could affect the module accuracy by a large impact. I decided to try and make preprocessing pipeline as good as possible. The initial Idea was to perform image sharpening to improve image quality, adaptive histogram equalization to make the images more robust to varying lighting conditions and converting to LAB color space so colors can be a little more easier in separation and learning for the network. However, I was surprised by some points as will be discussed in the network training below.

##Data Augmentation

I have augmented the data to increase training points as suggested in the lectures, I didn't see added benefit in performing random translations, rotation or brightness transformations on the image as I consider them an already variant dataset.

So the augmentation was mainly to flip the center images, add the left and right camera images with a constant offset to steering angle.

With the above steps I have around 41,184 data points (around 33% increase).

This posed a challenge actually as at after some point the images won't fit in memory at all once, so I have decided to use a generator as discussed in the lectures.

---

###Network architecture

I started out with Lanet as in the previous project, However no matter how I modified lanet structure, layout, filter size and tuned the hyper parameters the car would always hit an obstacle and not consistently also, which is a little weird, so as an example one time you find the car passes the bridge on track 1 and then goes off road to go onto the sand area off road, the next run you find that it hits the bridge side.

This was not satisfying as clear, I decided to try the nvidia architecture proposed in the lectures, doing so proved to be very helpful in getting consistent results however network training times was quite large compared to lanet.

To address this I modified the script to train multiple networks in sequence so I can start it leave it run over night/work and test it afterwards.

I have tweaked it a little to try and have the best performance, however the change i settled in for at the end was to add some dropout layers and the output was quite satisfying.

The final network architecture is:

| Layer              |     Description                               |
|:------------------:|:---------------------------------------------:|
| Input              | 160x320x3 Color image                         |
| Normalization      | normalized input to 0 by dividing by value for each channel by 255 and subtracting 0.5 |
| Cropping           | Crop unneeded part of image, output: 160x225x3 Color image |
| Convolution 5x5    | 24 filter, 2x2 stride                         |
| RELU               |                                               |
| Convolution 5x5    | 36 filter, 2x2 stride                         |
| RELU               |                                               |
| Convolution 5x5    | 48 filter, 2x2 stride                         |
| RELU               |                                               |
| Convolution 3x3    | 64 filter                                     |
| RELU               |                                               |
| Convolution 3x3    | 64 filter                                     |
| RELU               |                                               |
| Flattening         | Flattens the network                          |
| Fully connected    | Output 116                                    |
| DROPOUT            | 0.5 Keep probability                          |
| Fully connected    | Output 100                                    |
| DROPOUT            | 0.5 Keep probability                          |
| Fully connected    | Output 50                                     |
| DROPOUT            | 0.5 Keep probability                          |
| Fully connected    | Output 10                                     |
| Fully connected    | Output 1                                      |

I have used an adam optimizer for 5 epochs to train the network.

---

### Network Training

Due to long network training time I have modified the model.py script to train multiple network in series.

To my surprise all image preprocessing techniques that I used in previous projects seems to lead to worse results that using training images directly.

For two of the preprocessing techniques is quite clear:
  * Adaptive histogram equalization: this proves to be very helpful with images that have a large variation in brightness to it provide a more uniform light distribution but it can sensitive to noise, less than a global histogram but still sensitive, in an image, checking the simulator I found that some images, taken at fastest simulator quality, had a lot of noise pixels (black, distorted and some the road was colored incorrectly) so this leads to worse performance.
  * Sharpening due to most of the images taken at the highest resolution there is not much the sharpening technique can add as the images are already quite sharp.

However converting to LAB color space should, as per my understanding increase performance, but it lead to a worse performance in my tests which I am still trying to figure out.

I have tried training for a varied number of epochs but settled on 5 epochs in the end as best point before overfitting occurs in the network.

From all tested networks that one I found best is passing the image directly to the network described above, named model1.

I have decided to split my data to:
  * Training set contains 80% of training points, around 32,947 labeled images.
  * Validation set contains 20% of training points, around 8,237 labeled images.

---

Running the model

The model can be tested by running the simulator and choosing autonomous mode and running the following command:

python drive.py model1.h5

The network successfully made a full lap around track 1 without getting off the road, please view the output video in model1.mp4

I was getting bored as how slow the car was actually going, so I tried increasing the speed in the drive.py file to see if my network can actually operate at higher speeds so less time to correctly calculate steering angle.

The maximum I tried at was 27 mph, please view the output video in model1_fast.mp4, although a lot wobbly, going left and right, the network managed to complete a full lap without getting out of the road.

I think the unnecessary steering was due to overshoot so the network would turn too hard to the left or right then it needs to correct and it would overshoot again and so on, I can only attribute this to the constant offset used with the right and left camera images during data augmentation, this constant factor introduce a very clear error as it is not a real data point at low speeds its effect in some what limited as the network has a lot of time to correct the steering angle without noticeable changes, but increasing the speed shows it's effect as car response time is limited compared to before.

---

### Summary

The network is trained successfully, it is achieves intended goals and mimics the driver behaviour.
