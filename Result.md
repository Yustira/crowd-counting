# Result of the Improvement Model

This section will provide the result of our improvement model. We trained our model with random cropping images with a size of 96 x 128.
We also used the Gaussian kernel to blur the predicted image without affecting the count.

## Result of 900 random cropping from test images.
![](/images/mae_1.PNG)
Our model can predict the random cropping image better with the MAE 0.47-0.52. Even though
To show how better our model predicted the test images we can plot the predicted density map along with the real density map.
![](/images/Result_1.png)
Even though there's a difference between the count, the density map still represent the location of people in there
![](/images/Result_2.png)

We also used a plot with the x-axis represents the actual count while the y-axis represents the predicted count.
![](/images/plot_2.png)

The plot above also showed the **distribution of error**.
A good model shows the random errors, our model represents the random errors but when the actual count increase the predicted count seems to predict lower than the actual.
This is due to the chance of learning from highly-crowded random cropping is low.

## Result of 300 original test images.
![](/images/mae_2.PNG)

We tested our model to predict the original size of 480 x 640. Surprisingly, our model did a good job of predicting the count with MAE 2.28. 
![](/images/Result_3.png)

When we see the image above, we can see the crowded location has brighter pixels, so we plot the density map together with the original image.
![](/images/Result_4.png)

The bright pixels in the middle represent several people sitting together. Based on this example, we confident to use our model for monitoring physical distancing.
![](/images/plot_1.png)

We also do the same thing and checked the distribution of error. The full images show the errors without any pattern and even better than the previous plot.

After checking the result of our model, we can conclude that our model predicts better than the previous baseline model.
Not only provides a better result, but this model also gives information about people location and density through the density map.

