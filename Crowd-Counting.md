# Crowd Counting: Video Surveillance Upgrade for Monitoring Physical Distancing (BDG2-D)

## What is Crowd Counting? Why do we need that?
Crowd Counting is a technique used to count the number of objects in a picture.
Counting how many objects in an image is essential for various industries and researchers. Some of the use-cases are:

* **Counting Cells/Viruses/Bacteria/Pathogens:** \
  1.Revealing the information about the progress of infectious disease.\
  2.Providing growth and death rate for research.\
  3.Adjusting the amount of chemical substance to be given.
  
* **Community Events Counting:**\
  1.Quantifying the performances or shows based on the number of people.\
  2.Providing data for further analysis.
  
* **Safety and Road:** \
  1.Monitoring high traffic roads, or public places.\
  2.Preventing people from entering the forbidden, or dangerous places.\
  3.Detecting accident or fight when people are suddenly assembling.
  
* **Marketing:**\
  1.Giving information about the favorite spots where a high number of people gather.\
  2.Analyzing the road which many vehicles pass-through for advertising.
  
There are so many others use-cases of Crowd Counting that are not mentioned here. This shows the usefulness of crowd counting in real life. Because of that reason too, we will focus to develop crowd counting techniques on this project. However, crowd Counting is not an easy task instead of a laborious task and often occurred counting error when dealing with something like these pictures:

![](/images/density_ex2.jpg)
<div align="center">Source:nutraceuticalbusinessreview.com</div>
<br /> 

![](/images/density_ex.jpg)
<div align="center">Source:digest.bps.org.uk by Christian Jarrett</div>
<br /> 

Too reduce the wasted time because of counting endlessly every image or forgetting the count after counting to 100, We can use **Convolutional Neural Network** as a crowd counting method. 

Started from something simple, The first thing we consider to use is a plain CNN with an image as the input and the count as the label. 
Before we move into the model, we'll show the dataset!

## The Dataset. Why do we choose this dataset?

## Baseline Model. How does the result?
First, we tackle this problem with State of the Art method on Convolutional Neural Network. We create three convolution blocks with ReLU activation. We add BatchNormalization layer inside this block to reduce internal covariate shift of the model. This will cause the model to train more faster. We also add AveragePooling layer to reduce the dimension of the feature map. Reducing the dimension of the feature map will decrease the number of parameters in the model, so the model will much more easy to train. Then, we add Dropout layer to prevent overfitting. At the end of the model, we add GlobalMaxPooling to reduce the depth of the feature map and connect it to final activation ReLU layer. We use MeanSquaredError loss for this task and Adam method to perform the optimization. We also use MeanAbsoluteError for the metrics. After we train the model within 150 iteration, we get MEA around 1 for training set and around 5 for test set.

Then, we also try to use transfer learning method to tackle this problem. We use MobileNet architecture which have been trained on the ImageNet dataset. We use this model because the model is quite simple compared to another model architecture. We reduce the size of the image from 640x480 to 150x150 while maintain the aspect ratio (by using padding). This was done because the amount of the RAM that we have aren't enough if we don't reduce the size of the image. After we train the model within 150 iteration, we get MEA around 0.5 for training set and around 2 for test set which is an improvement from the SOTA model that we created before.


## Improvement. What can be improved?
