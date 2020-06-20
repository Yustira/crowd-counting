# Improvement. What can be improved?
After we implemented our baseline model successfully, we seek some improvement in order to get the best result.
## Change the label to density map 
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

## Structural Similarity Index (SSIM) and Local Pattern Consistency loss

In this section, we used the Structural Similarity Index to measure the similarity between 2 images. Then, use the similarity measure as a loss function named Local Pattern Consistency loss + the loss from Euclidean loss. (Cao et al., 2018) purposed the works of crowd counting with Local Pattern Consistency loss with Euclidean Loss as the loss function.

For each predicted density map and actual density map, we count the similarity between a small patch of the image with a gaussian kernel of 12x16 as the weight. We summed the total similarity between each small patch and penalized the model if there are many unsimilarity between the small patch.

We define our model loss function as follow:
Euclidean Loss (MSE) + alpha * Local Pattern Consistency Loss

## 



