
# Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[data_visualization]: ./figures/bar_chart_initial.png "Data Visualization"
[new_image_0]: ./examples/02_speed_limit_50_rotated.jpg "Speed limit 50"
[new_image_1]: ./examples/07_speed_limit_100.jpg "Speed limit 100"
[new_image_2]: ./examples/09_no_passing.jpg "No passing"
[new_image_3]: ./examples/11_right_of_way.jpg "Right of way"
[new_image_4]: ./examples/13_yield.jpg "Yield"
[new_image_5]: ./examples/35_ahead_only.jpg "Ahead only"
[new_image_6]: ./examples/00_stau.jpg "Traffic Jam"
[bar_chart_0]: ./figures/bar_chart_0.png "Bar Chart for Speed limit 50"
[bar_chart_1]: ./figures/bar_chart_1.png "Bar Chart for Speed limit 100"
[bar_chart_2]: ./figures/bar_chart_2.png "Bar Chart for No passing"
[bar_chart_3]: ./figures/bar_chart_3.png "Bar Chart for Right of way"
[bar_chart_4]: ./figures/bar_chart_4.png "Bar Chart for Yield"
[bar_chart_5]: ./figures/bar_chart_5.png "Bar Chart for Ahead only"
[bar_chart_6]: ./figures/bar_chart_6.png "Bar Chart for Traffic Jam"
[feature_maps_limit_50]: ./figures/speed_limit_50_visualization_output_conv1.png "Feature maps at conv1 for 'Speed Limit 50' rotated"
[feature_maps_ahead_only]: ./figures/ahead_only_visualization_output_conv1.png "Feature maps at conv1 for 'Ahead only'"

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is **34799**.
* The size of the validation set is **4410**.
* The size of test set is **12630**.
* The shape of a traffic sign image is **(32, 32, 3)**.
* The number of unique classes/labels in the data set is **43**.

#### 2. Include an exploratory visualization of the data set.

Here is an exploratory visualization of the data set. It is a bar chart showing the number of each road sign appeared in the training, validation and test data sets. Obviously, the road signs are not equally distributed throughout the data sets. For example, pictures of "Speed limit (20km/h)" are 10 times less present in the data sets than "Speed limit (30km/h)" (see first and second bars in the bar chart). This means that the first road sign might be underlearned by the neural network because of lack of samples.

![alt text][data_visualization]

### Design and Test a Model Architecture

As a first step, I decided to normalize the image data as a preprocessing step because it makes the problem to become well conditioned which, in turn, makes it easier for the optimizer to do its job.

Apart of normalization, there is no difference between the the original data set and the augmented data set. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model reproduces the standard *LeNet* configuration taking into account *RGB* image inputs. It consists of the following layers:

| Layer         		|     Description	        					| Output |
|:--------------------- |:--------------------------------------------- |:--- 
| Input         		|  *RGB* image   							| 32x32x3
| Convolution 5x5     	| 1x1 stride, valid padding 	| 28x28x18
| RELU					|												|
| Max pooling	      	| 2x2 stride  				| 14x14x18
| Convolution 5x5	    | 1x1 stride, valid padding 	| 10x10x54
| RELU					|												|
| Max pooling	      	| 2x2 stride    				| 10x10x54
| Flatten               |                                   | 1350
| Fully connected		|  									| 400
| RELU					|												| 
| Fully connected		|  									| 250
| RELU					|												|
| Fully connected		|  									| 43
| Softmax				|            									|


### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I use *Adam Optimizer* provided by *TensorFlow*. The batch size is set to 128. The learning rate is 0.001. The number of epochs is 10. The initial weighs of convolutional and fully connected layers are chosen based on a truncated normal distribution with mean 0 and standard deviation 0.1. The difference to the normal distribution is in re-picking those generated values that are more than 2 standard deviations away from the mean.

### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.


The standard *LeNet* configuration is chosen that performs well on recognizing patterns on the images. The model results are:
* training set accuracy of 100%,
* validation set accuracy of 94.5%,
* test set accuracy of 93.9%.

The final model's accuracy shows that the model works well.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Since I'm living in Germany, I've taken photos of 7 road signs myself and tried to classify them using the outcome of the previous steps. Here are the images:

![alt text][new_image_0]
![alt text][new_image_1] ![alt text][new_image_2] ![alt text][new_image_3] 
![alt text][new_image_4] ![alt text][new_image_5] ![alt text][new_image_6]

The first image might be difficult to classify because it is rotated and there is lack of brightness. The last image is not in the data set, and it is interesting to see how the model classifies the image in this case as well.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------|:---------------------------------------------| 
| Speed Limit 50 (dark, rotated)     	| Speed limit (60km/h)			| 
| Speed Limit 100      		| Speed Limit 100   									| 
| No passing     			| No passing 										|
| Right-of-way at the next intersection | Right-of-way at the next intersection |
| Yield					| Yield											|
| Ahead only	      		| Ahead only					 				|
| Traffic jam (not in the data set)			| Bumpy road      							|


The model is not supposed to classify the first and the last images correctly. Moreover, the first image rotated properly is classified accurately. Therefore, the model correctly guesses all remaining traffic signs that gives an accuracy of 100%. I cannot compare the accuracy based on 5 images to the accuracy on the test data set of 92%. For this I need to classify more images.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

For the first image, which is rotated "Speed Limit 50km/h", the model is sure that this is a "Speed Limit 60km/h" (probability of 0.985), which is wrong. The top five soft max probabilities are

| Probability         	|     Prediction	        					| 
|:---------------------|:---------------------------------------------| 
| .985         			| Speed Limit 60km/h   									| 
| .006   				| End of all speed and passing limit					|
| .004					| Speed limit (70km/h)											|
| .003	      			| Speed limit (80km/h)					 				|
| .002				    | Keep right     							|

![alt text][bar_chart_0]

For the second image, the model correctly classifies that this is a "Speed limit (100km/h)" sign (probability of .999999166). The top five soft max probabilities are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------| 
| .999999166         			| Speed limit (100km/h)   									| 
| 8 * 10^(-7)  	| Speed limit (120km/h) 										|
| 1 * 10^(-9)					| Speed limit (50km/h)											|
| 4 * 10^(-10)	      			| Speed limit (80km/h)					 				|
| 2 * 10^(-10)				    | Roundabout mandatory      							|

![alt text][bar_chart_1]

For the third image, the model correctly classifies that this is a "No passing" sign (probability of almost 1). The top five soft max probabilities are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------| 
| 1.        			| No passing   									| 
| 1 * 10^(-20)   	| Speed limit (120km/h) 										|
| 5 * 10^(-21)					| Speed limit (50km/h)											|
| 1 * 10^(-21}	      			| Speed limit (80km/h)					 				|
| 5 * 10^(-22}				    | Roundabout mandatory      							|

![alt text][bar_chart_2]

For the fourth image, the model is sure that this is a "Right-of-way at the next intersection" sign (probability of almost 1). The top five soft max probabilities are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------| 
| 1.           			| Right-of-way at the next intersection   									| 
| 9 * 10^(-14)   	| Beware of ice/snow										|
| 2 * 10^(-23)					| Double curve											|
| 4 * 10^(-25)	      			| Pedestrians					 				|
| 9 * 10^(-27)				    | Turn left ahead     							|

![alt text][bar_chart_3]

For the fifth image, the model is relatively sure that this is a yield sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------| 
| 1.         			| Yield   									| 
| 1 * 10^(-27)   	| Children crossing 										|
| 6 * 10^(-30)					| No passing											|
| 7 * 10^(-32)	      			| No vehicles					 				|
| 3 * 10^(-33)				    | Speed limit (60km/h)      							|

![alt text][bar_chart_4]

For the sixth image, the model is relatively sure that this is a ahead only sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------| 
| 1.         			| Ahead only   									| 
| 7 * 10^(-18)   	| Turn right ahead 										|
| 2 * 10^(-18)					| Roundabout mandatory											|
| 5 * 10^(-19)	      			| Go straight or right					 				|
| 3 * 10^(-19)				    | Turn left ahead      							|

![alt text][bar_chart_5]

For the seventh image, the model is sure that this is a bumpy road sign (probability of 0.999), but the image contains a sign that is not present in the data set (a traffic jam sign). The top five soft max probabilities are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------| 
| .999         			| Bumpy road   									| 
| 3 * 10^(-4)      				| Bicycles crossing 										|
| 3 * 10^(-5) 					| Road work											|
| 2 * 10^(-6) 	      			| Beware of ice/snow					 				|
| 5 * 10^(-7) 				    | Slippery Road      							|

![alt text][bar_chart_6]


### Visualizing the Neural Network
The visual output of the trained network's feature maps is provided below for two signs "Speed Limit 50" and "Ahead only". These feature maps are built for the first convolutional layer *conv1*. The inner network feature maps react with high activation to the sign's round boundary outline for both images. The ahead only sign activates also the reaction on the contrast in the sign's arrow up simbol.

![alt text][feature_maps_limit_50]
![alt text][feature_maps_ahead_only]

