# OCT_image_classification
Binary classification of medical images using Convolutional Neural Network.

The dataset is balanced. There are 24000 images in each of the two classes. Out of these 75% are used for training and 6000 are used for testing. A simple Convolutional Neural Network (CNN) configuration is used.

The following figure compares a normal retina image to abnormal retina image.  Clearly the abnormal images have perforations which distinguishes it from the normal retina images.

<p align="left">
<img width="400" height="250" src="images/retina_compare.png">
</p>  

 The following figure compares three images of each class. 

<p align="left">
<img width="400" height="250" src="images/retina_compare_many.png">  
</p>  


The following shows the basic configuration of CNN used in this work. There are two convolutional layers and one dense layer. The sensitivity of the accuracy to the number of neurons in the dense layer, the learning rate and dropout is explored.

![](images/keras_CNN_configuration.png)

<br>
The following shows the sensitivity to learning rate.

<p align="center">
<img width="500" height="300" src="images/number_learning_rate.png"> 
</p> 


<br>
The following shows the sensitivity of the accuracy to the number of neurons. For the configuration with 8 and 16 neurons there are too few neurons for the model to fit to the observations. Hence the accuracy is too low. With a the number of neurons equal to 128, the model is flexible enough to fit to the observations and hence the accuracy increases compared to the case of too few neurons. However, there is evidence of overfitting in that the training accuracy is close to 1 while the validation accuracy is much lower ( ~ 0.88). Decreasing the # of neurons to 32 gives a better result in that the overfitting decreases. This is quantified by the decrease in gap between the training and validation accuracy. The training accuracy is 0.92 and validation accuracy is 0.88. Consquently 32 is the best choice for this problem. 

<p align="center">
<img width="500" height="300" src="images/number_neurons_5.png"> 
</p>  

The following explores the sensitivity of accuracy to the number of filters.

<p align="center">
<img width="500" height="300" src="images/number_filters.png"> 
</p>

The following explores the sensitivity of accuracy to the magnitude of dropout.

<p align="center">
<img width="500" height="300" src="images/number_dropout.png"> 
</p>

The following explores the effect of data augmentation on accuracy.

<p align="center">
<img width="500" height="300" src="images/number_dataaugmentation.png"> 
</p>

# References
https://towardsdatascience.com/balancing-the-regularization-effect-of-data-augmentation-eb551be48374

https://stats.stackexchange.com/questions/295383/why-is-data-augmentation-classified-as-a-type-of-regularization
