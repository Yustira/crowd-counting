# Crowd Counting: Video Surveillance Upgrade for Monitoring Physical Distancing (BDG2-D)

By:

Harrianto Sunaryan\
Iyas Yustira\
Muhammad Farrel M\
Rinda Nur Hafizha

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
  
There are so many others use-cases of crowd counting that are not mentioned here. This shows the usefulness of crowd counting in real life. Because of that reason too, we will focus to develop crowd counting techniques on this project. However, crowd counting is not an easy task if it is done manually by our hand because we can lost count in the middle of doing this laborious task, especially when dealing with object that intersects with each other or dense crowd like these pictures:

![](/images/density_ex2.jpg)

<div align="center">Source:nutraceuticalbusinessreview.com</div>
<br /> 

![](/images/density_ex.jpg)

<div align="center">Source:digest.bps.org.uk by Christian Jarrett</div>
<br /> 

To solve this problem, we can automate the counting process by building a machine learning system that can receive an image that contains objects that we want to count as an input and then the model will output number of objects in that image. We build the model using **Convolutional Neural Network (CNN)** technique.
<br/>

The system that we build is capable of **counting pedestrians in a mall**. The images are generated from CCTV that is placed somewhere in the mall. From those images, the system will tell us how many pedestrians at that particular place in the mall.
<br/>


Started from something simple, The first thing we consider to use is a plain CNN with an image as the input and the count as the label. 
Before we move into the model, we'll show the dataset!

## The Dataset.
The dataset that we use to build the machine learning model is <a href="http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html">mall dataset<a/>, which is dataset that contains 2000 images with size 480x640 pixels at 3 channels (RGB) as input and its corresponding number of people as the label. Each image has different number of persons. The images are generated from a single CCTV that is placed somewhere in a mall of the same spot, which contains pedestrians who walks around the CCTV. 
<br/>

## Baseline Model
First, we tackle this problem with State of the Art method on Convolutional Neural Network. We create three convolution blocks with ReLU activation. We add BatchNormalization layer inside this block to reduce internal covariate shift of the model. This will cause the model to train more faster. We also add AveragePooling layer to reduce the dimension of the feature map. Reducing the dimension of the feature map will decrease the number of parameters in the model, so the model will much more easy to train. Then, we add Dropout layer to prevent overfitting. At the end of the model, we add GlobalMaxPooling to reduce the depth of the feature map and connect it to final activation ReLU layer. We use MeanSquaredError loss for this task and Adam method to perform the optimization. We also use MeanAbsoluteError for the metrics. After we train the model within 150 iteration, we get MEA around 1 for training set and around 5 for test set.

![](/images/Plot_baseline.jpeg)

From the plot above, we can see that the model perfom poorly when the number of people in an image is very low or very high.

## Improvement
After we implemented our baseline model successfully, we seek some improvement in order to get the best result.
### Change the label to density map 
![](/images/comparison.png)
From the images above, we clearly see that the position of people in both pictures are different. However, in order to get the predicted count, we didn't use spatial information about people location.


In the first improvement, we tried to use the spatial information of people location by using the annotated frame like the image below as the label instead of the count.

We assigned the location of people head by 1, otherwise 0.

![](/images/annotated.png)

When we tried to use the annotated frame as a label, the model didn't learn well because it was too hard to guess the position of people's head location. Only around 30 pixels which annotated as people position in one image. Therefore, we suggested using density map estimation like this image as an example.

![](/images/density.png)

Clearly, there are more pixels which value is not zero. Thus, the model will learn by making the right prediction near people's heads. The method to use this density map is based on the work of (Zhang et al., 2015).

to generate density function we proposed using Adaptive Gaussian Kernel, where each person's head is blurred with a Gaussian Kernel with different sigma that represents people's head size in the image. People on the top side of the image are having a smaller size of heads due to the perspective distortion. Hence, generating different sigma for each people would help the model to learn the different sizes of people's heads in the image.

After using the gaussian kernel, we can use normalization to make the sum of the image still represent the number of people.

Another improvement in this section, we assigned the pixel of people head by 1000 instead of 1. Thus, MSE will penalize more when the model doesn't predict accurately near the people's head.

### Structural Similarity Index (SSIM) and Local Pattern Consistency loss

We used the Structural Similarity Index to measure the similarity between 2 images. Then, use the similarity measure as a loss function named Local Pattern Consistency loss + the loss from Euclidean loss. (Cao et al., 2018) purposed the works of crowd counting with Local Pattern Consistency loss together with Euclidean Loss as the loss function.

For each predicted density map and actual density map, we count the similarity between a small patch of the image with a gaussian kernel of 12x16 as the weight. We summed the total similarity between each small patch and penalized the model if there are many unsimilarity between the small patch.

We define our model loss function as follow:
Euclidean Loss (MSE) + alpha * Local Pattern Consistency Loss

Local pattern consistency loss helps model to learn similarity between small patches of the image.

### Network Arhitecture
The following image describes our model. Where 'CONV a' denotes Convolution with kernel size a x a and 'CONV T' means Transposed Convolutional layer.

![](/images/Network_Architecture.PNG)

Suppose we have an image with a size of 96 x 128.

First, we used the first ten layers of VGG16 to extract the features of the image. The work of (Yosinski et al., 2014) considers that the front-end of the network learns task-independent general features which are similar to Gabor filters and color blob. After we extracted the features with the first ten layers of VGG16, we got an output with size 24 x 32 which is a quarter of the original size.

Second, we fed our output from VGG16 to four filters with different sizes. The work of (Zhang et al., 2016) says that due to perspective distortion, the images usually contain heads of very different sizes, hence filters with receptive fields of the same size are unlikely to capture characteristics of crowd density at different scales. While, the work of (Li et al., 2018) suggests not using many pooling layers because of the loss of spatial information from the feature map. Hence, we didn't use the pooling layer in the set of convolutional layers.

Third, inspired by the work of (Tian et al., 2020) we used feature enhancement layer, where we concatenate our output (x_conct, contain 4 filter from the previous layers) and used flatten+MLP with softmax function in the output layer to get the weight for each input filters. Thus, the model will learn to give a high weight for the filter which best represents the image.

Last, we need to upsampling the image with size 24 x 32 to 96 x 128. We used Transposed Convolutional layer for the upsampling method rather than the conventional upsampling method. We also used concatenated the x_conct with the filter that generated by convolution the weighted x_conct. In the last layer, we set the filter size with 1, and that filter represents the predicted density map.

#for each convolutional layer, we used batch-normalization and ReLu as the activation function.\
#we also set the VGG16 layers non-trainable only for the first two epoch in order to make the other layers learn first and prevent the model to predict 0 in each pixel of the density map.

## Result of the Improvement Model

This section will provide the result of our improvement model. We trained our model with random cropping images with a size of 96 x 128.
We also used the Gaussian kernel to blur the predicted image without affecting the count.

### Result of 900 random cropping from test images.
![](/images/mae_1.PNG)
Our model can predict the random cropping image better with the MAE 0.51. To show how good our model predicted the test images, we can plot the predicted density map along with the real density map.

![](/images/Result_1.png)
<div align="center">Predicted Density Map (Left), Actual Density Map (Right)</div>
<br /> 


Even though there's a difference between the count, the density map still represent the location of people in there
![](/images/Result_2.png)

We also used a plot with the x-axis represents the actual count while the y-axis represents the predicted count.
![](/images/plot_2.png)

The plot above also showed the **distribution of error**.
A good model shows the random deviations/errors, our model represents the random deviations but when the actual count increase the predicted count seems to predict a bit lower than the actual count.
This is due to the chance of learning from highly-crowded random cropping is low.

### Result of 300 original test images.
![](/images/mae_2.PNG)

We tested our model to predict the original size of 480 x 640. Surprisingly, our model did a good job of predicting the count with MAE 2.28. 

![](/images/Result_3.png)

<div align="center">Predicted Density Map (Left), Actual Density Map (Right)</div>
<br /> 

When we see the image above, we can see the crowded location has brighter pixels, so we plot the density map together with the original image.

![](/images/Result_4.png)

The bright pixels in the middle represent several people sitting together. Based on this example, we confident to use our model for monitoring physical distancing.

![](/images/plot_1.png)

We also do the same thing and checked the deviations. The full images show that the deviations do not have any pattern and even better than the previous plot.

After checking the result of our model, we can conclude that our model predicts better than the previous baseline model.
Not only provides a better result, but this model also gives information about people location and density through the density map.

## References

Cao, X., Wang, Z., Zhao, Y., & Su, F. (2018). Scale aggregation network for accurate and efficient crowd counting. Computer Vision – ECCV 2018, 757-773. https://doi.org/10.1007/978-3-030-01228-1_45

Chen, K., Gong, S., Xiang, T., & Loy, C. C. (2013). Cumulative attribute space for age and crowd density estimation. 2013 IEEE Conference on Computer Vision and Pattern Recognition. https://doi.org/10.1109/cvpr.2013.319

Chen, K., Loy, C. C., Gong, S., & Xiang, T. (2012). Feature mining for localised crowd counting. Procedings of the British Machine Vision Conference 2012. https://doi.org/10.5244/c.26.21

Crowd counting. (n.d.). Kaggle: Your Machine Learning and Data Science Community. https://www.kaggle.com/fmena14/crowd-counting

Li, Y., Zhang, X., & Chen, D. (2018). CSRNet: Dilated Convolutional neural networks for understanding the highly congested scenes. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. https://doi.org/10.1109/cvpr.2018.00120

Loy, C. C., Chen, K., Gong, S., & Xiang, T. (2013). Crowd counting and profiling: Methodology and evaluation. Modeling, Simulation and Visual Analysis of Crowds, 347-382. https://doi.org/10.1007/978-1-4614-8483-7_14

Loy, C. C., Gong, S., & Xiang, T. (2013). From semi-supervised to transfer counting of crowds. 2013 IEEE International Conference on Computer Vision. https://doi.org/10.1109/iccv.2013.270

Mall dataset - Crowd counting dataset. (n.d.). https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html

Tian, Y., Lei, Y., Zhang, J., & Wang, J. Z. (2020). PaDNet: Pan-density crowd counting. IEEE Transactions on Image Processing, 29, 2714-2727. https://doi.org/10.1109/tip.2019.2952083

Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? Advances in Neural Information Processing Systems 27.

Zhang, Y., Zhou, D., Chen, S., Gao, S., & Ma, Y. (2016). Single-image crowd counting via multi-column Convolutional neural network. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). https://doi.org/10.1109/cvpr.2016.70


