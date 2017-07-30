#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./class_hist.png "Histagram of Road Sign Classes"
[image2]: ./straight_ahead.png "straight"
[image3]: ./30_sign.png "30_sign.png"
[image4]: ./roadwork2.jpg         "Traffic Sign 1"
[image5]: ./Do-Not-Enter.jpg      "Traffic Sign 2"
[image6]: ./general_caution.jpg   "Traffic Sign 3"
[image7]: ./50_sign.jpg           "Traffic Sign 4"
[image8]: ./right_of_way.jpg      "Traffic Sign 5"
[image9]: ./road-work.jpg         "Traffic Sign 6"
[image10]: ./turn-right.jpg       "Traffic Sign 7"
[image11]: ./turnright_probabilities.png      "test"
[image12]: ./roadwork1_probabilities.png      "test"
[image13]: ./rightofway_probabilities.png     "test"
[image14]: ./50sign_probabilities.png         "test"
[image15]: ./general_caution_probabilities.png       "test"
[image16]: ./donotenter_probabilities.png       "test"
[image17]: ./roadwork2_probabilities.png       "test"

[image18]: ./conv1.png       "test"
[image19]: ./conv2.png       "test"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/kilaorange/proj2-traffic-sign-classifier.git)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* The size of the validation set is ?
* Number of testing examples = 12630
* Image data shape = (32, 32)
* Number of classes = 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

![alt text][image2]

![alt text][image3]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

* The data was shuffled to prevent the order of the images affecting the model
* The data was normalised to help the model converge
* Images were augmented with random changes including:
	* Flipping left and right
	* Random rotation
	* Random blur

The augmentation of the images helped the model generalise over the test set as the model was able to train on a larger variety of images than what was gathered in the training set.



####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 |
| RELU					|					|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6, padding valid |
| Dropout					|		Dropout rate 0.2 |
| Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x16 |
| RELU					|						|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6, padding valid |
| Dropout				|		Dropout rate 0.2 |
| Flatten  			| Flatten conv layer to 400 fully connected layer |
| Fully connected	| 400 nodes    		|
| RELU 				|			 |
| Dropout				|		Dropout rate 0.2 |
| Fully connected	| 120 nodes    		|
| RELU 				|			 |
| Dropout				|		Dropout rate 0.2 |
| Fully connected	| 84 nodes    		|
| Fully connected	| 43 nodes    		|
| Softmax				| 						|



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimiser.
      
* EPOCHS = 30  
* BATCH_SIZE = 256  
* dropout = 0.2  
* num_classes = 43  
* Learning rate = 0.001  

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 7 German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9] ![alt text][image10]

The third and seventh image might be difficult to classify because they contain a water mark which may interfere with the feature detection. The other images should be easily identified as they are similar to training images.



####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The algorithm correctly identified 57.14% of new images. This is poor compared to the test set accuracy of 90.60%




####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]

For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image18]

Somethiung something  
![alt text][image19]
