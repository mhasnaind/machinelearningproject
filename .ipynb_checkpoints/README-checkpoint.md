# Project 2 - AI Machine Learning Fintech


![project2.png](Images/project2.png)

Data Sources:
* ASL Digit pictures dataset source: https://www.kaggle.com/orhansertkaya/convolutional-neural-network-sign-language-digits.  However these are 64x64 pixel pictures in numpy array.
* ASL Alphabet pictures dataset source: https://www.kaggle.com/datamunge/sign-language-mnist?select=sign_mnist_test. However these are 28x28 pixel pictures in csv format.

---
## Project Goal:

Train a machine learning model to successfully recognise American Sign Language alphabet and digit hand signals.

## Methodology:

- Build initial Machine Learning model to successfully recognise ASL digit hand signals. (10 symbols).

- Determine if the model can be re-fit for both ASL alphabet & digit hand signals. 
(34 symbols.  The symbols for J and Z are excluded as they are movement based.)

- Test additional hand pictures to see how model responds to other hand shapes.

## Instructions:

In order to run the model, you will need to create an 'input' folder and add 4 datasets (X.npy, Y.npy, sign_mnist_test.csv and sign_mnist_train.csv) in it. Then under the input folder, create a subfolder called New_Pictures, and put the additional pictures in that new folder.

## Digits Model:

In this section, we have used a dataset of 2062 images of ASL digit hand symbols, 64 x 64 pixel black and white images and 80/20 train vs test split.

![DM.png](Images/DM.png)

### Structure Digits Model:

![DMS.png](Images/DMS.png)


### New Model Used - CNN: 
A convolutional layer contains a set of parameters that need to be learned. Each filter is convolved with the input volume to compute an activation map. The activation map is slid across the width and height of the input in a "pool size" set in the model architecture with the weights computed at every spatial position.


### Digits Model Validation:
After 100 epochs, validation accuracy at 98.5%+ and validation loss at 0.06%.

![accloss.png](Images/accloss.png)

### Digits Model Confusion Matrix:

![CM.png](Images/CM.png)

## Alphabet Dataset:

In this section, we have first used a dataset of 34,627 images of ASL alphabet hand symbols, 28 x 28 pixel black and white images, in csv format, and the train vs test split were 80/20.

Then, we had to transform the dataset as required with Pixel values adjusted from 0-255 to between 0-1, One hot encoding of y class, mage augmentation (rotate, flip, shift, apply whitening, divide by std) and Resized images to 64 x 64.

![ALD1.png](Images/ALD1.png) ![ALD2.png](Images/ALD2.png) 

## Combined Alphabet & Digits Model:

In this section, we have combined the two dataset ASL digit and alphabet hand symbols with 36,684 images in total. The data was amended to numpy array using block_diag to join the 2 datasets. The train vs test split were 80/20.

Finally, we used the same structure and compile as Digits model.

### Model Validation:
After 150 epochs, validation accuracy at 97.6%+ and validation loss at 24.6%.

![accloss3.png](Images/accloss3.png)

### Model Confusion Matrix:

![CM3.png](Images/CM3.png)

## Testing Our Model with Additional Images:

### Image # 1

We ran our model by providing it a new input/image and expected a particular output; our model provided...

![image1.png](Images/image1.png)

### Image # 2

We ran our model by providing it a new input/image and expected a particular output; our model provided...

![image2.png](Images/image2.png)

### Image # 3

We ran our model by providing it a new input/image and expected a particular output; our model provided...

![image3.png](Images/image3.png)

### Image # 4

We ran our model by providing it a new input/image and expected a particular output; our model provided...

![image4.png](Images/image4.png)

### Image # 5

We ran our model by providing it a new input/image and expected a particular output; our model provided...

![image5.png](Images/image5.png)

## Learnings:

Data integrity - We encountered some issues with the labelling and ordering of data which gave us incorrect output on the trained model. In order to rectify it, we had to adjust our code accordingly to align with the dataset. We also realised the low resolution version of the images perhaps made it difficult for the model to accurately recognize the images. One of the key learnings for us was to ensure that we should go through our dataset quite rigorously before using it for machine learning.
 
Significant difference in model accuracy output between running on own computer vs running on Google Colab - this was perhaps due to Google Colab using an advanced version of Machine Learning software available
 
Colab run models may not be compatible to your computer - we struggled to run notebooks that were created on Jupyter Lab and encountered compatibility issues when we tried to upload it on Colab
 
Early stopping of epochs - When we tried to train model3, we realised that after a particular number of epochs, the validation-accuracy score started to decrease and the losses started to increase. Through trial and error, we had to stop at a particular number to prevent the model from overfitting.
