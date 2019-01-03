
# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to analyze the data set provided. The data analysis was done using the code in code cell 2

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

These are plotted in the code cells 4 and 5 in notebook

I have visualized the data sets by plotting the data using matplotlib.

 - Plotted 43 unique traffic signs with their labels
 - Plotted bar graph containing all the data set (training, validation, testing) vs. the count for each unique class
 
### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I applied two steps of pre-processing for all the datasets

 1. Grayscaling
 2. Normalization

**Grayscaling**: 
I did grayscaling because the colors in the traffic sign images added nothing valuable to my training model. For my training model, colors are not necessary, hence to reduce the workload of the classifier I decided to grey out the image

**Normalization**:
The image contained pixel values ranging from 0 to 255. To make the mean close to zero and standard deviation as low as possible I did normalization.

Both the steps can be found in the code blocks 6 in the notebook. Also I have output the images after applying pre-processing.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 greyscale image   							| 
| Convolution 5x5 followed by RELU     	| 1x1 stride, same padding, outputs 28x28x6 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|		
| Convolution 5x5 followed by RELU     	| 1x1 stride, same padding, outputs 10x10x32 	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				|		
| Convolution 3x3 followed by RELU     	| 1x1 stride, same padding, outputs 5x5x64 	|
| Convolution 1x1 followed by RELU     	| 1x1 stride, same padding, outputs 3x3x128 	|
|Flatten network | outputs 1x1152
| Fully connected	followed by RELU	| outputs 1x600        									|
| 			Dropout         		|50%							|
|				Fully connected	followed by RELU		| outputs 1x300	|	
| 			Dropout         	|50%								|
|				Fully connected	followed by RELU	| outputs 1x43		|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam's optimizer. 
Following Hyper parameters were used:
mu = 0
sigma = 0.1
dropout = 0.5
learning_rate = 0.001
epochs = 25
batch_size = 128

These are available in the code block following the model block in notebook

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.964
* test set accuracy of 0.944

If an iterative approach was chosen:
* **What was the first architecture that was tried and why was it chosen?**
	* I started with the trusted LeNet architecture. Since it was taught in the classroom and I was more familiar with the architecture.
	
* **What were some problems with the initial architecture?**
	* I used unmodified LeNet initially. The validation accuracy came around 0.921 with the preprocessing applied. It would not have performed well with the test data. So I thought of modifying it.
	
* **How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.**
	* To increase the number of neurons, I added two more convolution layers. By this way I was able to extract more details from the image set. The I added one extra fully connected layer to summarize the dataset.
	
* **Which parameters were tuned? How were they adjusted and why?**
	* I tuned the dropout rate and learning rate for many iterations. By trial and error method, I found out the final parameters which were best fit for my classifier
	
* **What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?**
	* Convolution layers added more details to the network by extracting whatever details available and RELU was applied to it.
	* Dropout layer helped to keep check on overfitting issue
	* Fully connected network helped to classify the image on a high level
	* For my final model, I chose 25 epochs since having more epochs resulted in nothing regarding accuracy but wastage of precious resource.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The five German traffic signs I found on web has been populated in the notebook cells. Please refer the notebook cells 13.

The first, second and last image were pretty easy to classify since it has good contrast and legibility. The third and fourth images were a bit lack of exposure. We might need to adjust the histogram according to the image exposure.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 0.80
Test accuracy for default test data set was around 0.944 - but in images from web, the classifier predicts on average. There is a lot more room for improvement. The new traffic sign would have benefited from image augmentation, which is single most powerful technique to increase accuracy.

I have neglected augmentation for time being, yet got 0.968 validation accuracy. It would have increased the prediction greatly. I would definitely try image augmentation as it seems very interesting.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

As said for the above answer, model is good at predicting signs when the training data set is large. Also this model has a 0.80 prediction for the new images.

For detailed softmax probabilities for each image, it has been plotted in the notebook along with the images. Please refer the notebook cell 16 (the second last one !)

> Best Regards
> Vivek Mano



