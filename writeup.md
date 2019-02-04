# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[centerImg]: ./examples/center.jpg "Cneter Image"
[leftImg]: ./examples/left.jpg "Left Image"
[rightImg]: ./examples/right.jpg "Right Image"
[flipImg]: ./examples/flip.jpg "Flipped Image"
[centerPreImg]: ./examples/center_preprocessed.jpg "Preprocessed Center Image"
[leftPreImg]: ./examples/left_preprocessed.jpg "Preprocessed Left Image"
[rightPreImg]: ./examples/right_preprocessed.jpg "Preprocessed Right Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Included Files

My project includes the following files:
* **model.py**: containing the script to create and train the model
* **drive.py**: for driving the car in autonomous mode
* **model_final.h5**: containing a trained convolution neural network
* **writeup.md**: summarizing the results
* **jungle.mp4, lake.mp4**: video recording of vehicle driving autonomously around the tracks

#### 2. Code information

The project is using TensorFlow 1.12.0 & Keras 2.0.9.
Running by Nvidia RTX 2080 on Windows 10.

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_final.h5
```

The **model.py** file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

**drive.py** is set the desired speed to 15mph, and add the preprocessing for the input images.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My training model is based on [Nvidia End-to-End learning model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/), because it is verified to work with this self-driving problem.

Here I'm using ELU (Exponential Linear Unit) activation instead of RELU for all convolutional layer, since it is speed-optimized.
The data is normalized in the model using a Keras lambda layer (model.py line 152). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 158). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (line 141). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an **adam** optimizer, so the learning rate was not tuned manually (model.py line 166).
Loss Function is using MSE(Mean Squared Error), as it is efficient for regression problem.

#### 4. Appropriate training data

The training data I used is a combination of center lane driving, recovering from the left and right sides of the road, especially at the sharp turning part.
First I was recording data from 2 ~ 3 loops of Lake track, 1 loop with recovering drving and other loops to drive as smooth as I can, then drvie Counter-Clockwise about 1~2 loops to collect balanced data.
Second, collect the Jungle track data, also 2 ~ 3 loops in same way with recovering driving and smooth driving, and reverse driving for the balanced data.

### Model Architecture and Training Strategy

First, I used the [Nvidia Model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) which is introduced in instruction videos to see how it perform.
Because is is tested to be work in Nvidia's self driving research, so I assume it is a proper starting point for this project.

#### 1. Data Preprocessing

The input image of Nvidia Model is 66 x 200 x 3 in YUV color space.
The original input camera image is 160 x 320 x 3 in RGB.
To filter out unnecessary part of image to avoid distraction, I crop off (top: 70, bottom: 20).
Then resize the image into YUV 66 x 200 x 3.

Here is an example image of center camera :

![alt text][centerImg]

After proprocessing:

![alt text][centerPreImg]

Left camera Image:

![alt text][leftImg]

After proprocessing:

![alt text][leftPreImg]

Right camera Image:

![alt text][rightImg]

After proprocessing:

![alt text][rightPreImg]

I set the corrected steering angle for left and right camera to ± 0.45

To augment the data sat, I also flipped images and angles:

![alt text][flipImg]

The last step is to randomly shuffled the data set in output of image generator

#### 2. Final Model Architecture

Here's the architecture looks like:

| Layer         		|     Description	        					|  output size
|:---------------------:|:---------------------------------------------:|:------------:| 
| Input Lamda         	| 66x200x3 YUV image, normalized to -0.5 ~ 0.5	| (66, 200, 3) |
| Convolution 5x5     	| filter: 24, strides: 2x2, activation: ELU     | (31, 98, 24) |
| Convolution 5x5     	| filter: 36, strides: 2x2, activation: ELU     | (14, 47, 36) |
| Convolution 5x5     	| filter: 48, strides: 2x2, activation: ELU     | (5, 22, 48)  |
| Convolution 3x3     	| filter: 64, strides: 1x1, activation: ELU     | (3, 20, 64)  |
| Convolution 3x3     	| filter: 64, strides: 1x1, activation: ELU     | (1, 18, 64)  |
| Drop out		        | dropoup prob 0.5     			                | (1, 18, 64)  |
| Flatten	         	| outputs 1152              		     		| (1152)       |
| Fully connected		| output 100       				     			| (100)        |
| Fully connected		| output 50                         			| (50)         |
| Fully connected		| output 10         							| (10)         |
| Fully connected		| output 1          							| (1)          |

The final model architecture (model.py lines 151-163) consisted of 5 convolution layer and following 4 fully connected layer.

#### 3. Training & Validation Process

In order to evaluate how well the model was working, I split my image and steering angle data into a training and validation set in 80% vs 20%. 
During training, when I saw the training loss is low but the validation loss is high, that means the model was overfitting.
To avoid overfitting, I applied and dropout layer and more balanced training & validation data.
Another thing is it looks like if the training **epoch > 5**, the training & validation loss is not really improving, so I add a callback of **EarlyStopping** and save the best model of every epoch to make sure I always get the best result.

Final step was to run the simulator to see how well the car was driving around both lake & jungle track. 
There were a few spots where the vehicle fell away of the track, to fix that, I tried to collect more recovering & smooth drving data in those specific tricky turns.

Finally, the vehicle is able to drive autonomously around the track without leaving the road and able to correct itself.

I tried 2 ways for data generation, one is load all data in memory, the other one is using data generator.
Both working well, the differece is the loading data in memory and using `model.fit()` to train is much faster than using data generator by using `model.fit_generator()`.
`model.fit()` takes about 15 ~ 20 minute to process 50 epoch with my training data about 50k images.
`model.fit_generator()` takes about 15 ~ 20 minute to process each epoch with my training data about 50k images. So 5 epoch will be 75 ~ 100 minutes.

The ideal number of epochs was **5** for `model.fit_generator()`, if more epoch, it looks like didn't improve the loss.
The ideal number of epochs was **50 ~ 100** for `model.fit()`, I used **50** epoch in this project, which the result seem to be work well enough.

## Result
Here are the videos:  
Lake:  
https://youtu.be/kjrcxg0QH0w

Jungle:  
https://youtu.be/MHXeek1U_t0
