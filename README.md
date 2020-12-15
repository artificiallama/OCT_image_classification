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
The number of neurons along with the number of hidden layers quantify the capacity of the neural network. Higher the number of neurons (and hidden layers) higher the number of free (tunable) parameters of the neural network model. Given a fixed amount of training data, higher number of parameters tend to overfit the training data. An overfit model tends to perform very well on the training data but fails miserably on the test data. The generalization error of such models is high. An overfit model has low bias and high variance.

The following figure shows the sensitivity of the accuracy to the number of neurons. For the configuration with 8 and 16 neurons there are too few neurons for the model to fit to the observations. Hence the accuracy is too low. The model does not have enough capacity (i.e. flexibility)  to fit the data. The low number of neurons result in an underfit model. The high bias results in a low accuracy (0.5). The other extreme is realized with N = 128. The training accuracy is close to 1 while the validation accuracy is ~ 0.88. The gap of ~0.12 between the training and validation accuracy quantifies the overfitting. Therefore the optimal value of N is between 16 and 128. Decreasing the number of neurons to 32 gives a best result in that the overfitting decreases. This is quantified by the decrease in gap between the training and validation accuracy. The training accuracy is 0.92 and validation accuracy is 0.88. Consquently N = 32 is the best choice for this problem. The overfitting is not eliminated completely with N = 32. Decreasing N from 128 to 32 decreases the variance of the model.

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

Saikat Biswas : How Regularization helps in data overfitting, *medium.com*, 2019.

