# Feedforward-Neural-Network
Implementation of a feed forward neural network with backpropagation from scratch with various optimization varieties like stochastic, momentum based, mini batch, adam and nadam. The network is trained and tested on the Fashion-MNIST dataset with different combinations of parameters like #epochs, #neurons, #hidden-layers, activation function, etc.


## Problem Statement (IITM CS6910 Asssignment-1) [link](https://wandb.ai/cs6910_2024_mk/A1/reports/CS6910-Assignment-1--Vmlldzo2ODQ1ODYy)

The problem statement of this assignment is to implement own feed forward neural network , backpropagation code for training the network with
gradient desecent (with many variations) algorithm from scratch. Train the implemented neural network with fashion mnist dataset and do experiments
with different hyperparameters to get maximum accuracy. Use wand.ai tool to keep track of all experiments.     


## Installing Libraries

 - For running in google colab, install wandb library -
  ``` !pip install wandb ```
 - For running locally, install following libraries  
  ``` 
  pip install wandb
  pip install numpy
  pip install keras
  ```

## Question 1

Solution Approach:
- Loaded fashion_mnist dataset using fashion_mnist.load_data().
- Picked one image from each class from the training dataset.
- Integrated wandb.io and plotted the images for each class.
The code for question 1 : [link](https://github.com/Rupak-Paul/Feedforward-Neural-Network/blob/main/Q1.py).

## Question 2 and 3
Solution Approach:
- Created a class **NuralNetwork** to represent a neural network. While initializing the NuralNetwork object, number of hidden layers, size of hidden layers, input size, output size, activation function, and weight initialization method has to be sent as parameters
  
- To train the neural network, first create an object of the NuralNetwork class by passing appropriate parameters. Then decide the training algorithm you want to use (for example Nadam). Call **NeuralNetwork.trainByNadam()** method and pass the required parameters.

- After training if you want to predict the class for some image then call NuralNetwork.predict() function and pass the input image. It will return the class in which the image belongs based on the highest probability.

- It is very easy to change hyperparameters in this implementation. Just change the value of these variables in the **main()**, Nothing else.
  ```
  epochs = 10
  hiddenLayers = 4
  hiddenLayerSize = 64
  weightDecay = 0
  learningRate = 0.01
  optimizer = 'nadam'
  batchSize = 32
  weightInitialisation = 'Xavier'
  activationFunction = 'ReLU'
  lossFunction = 'cross entropy'
  beta = 0.9
  beta1 = 0.9
  beta2 = 0.999
  inputSize = 28*28
  outputSize = 10
  ```
	
The code for question 2 and Question 3 : [link](https://github.com/Rupak-Paul/Feedforward-Neural-Network/blob/main/Q2_Q3.py).


## Question 4

Solution Approach:

 - Implemented Logging function in **NuralNetwork** class. It calculates the training accuracy, training loss, validation accuracy, and validation loss in each epoch, and then sends these data to wandb.ai for plotting the graphs.

The code for question 4 : [link](https://github.com/Rupak-Paul/Feedforward-Neural-Network/blob/main/Q4.py).


## Question 5

The wandb visualization for question 5 : [link](https://wandb.ai/cs23m056/CS23M056_DL_Assignment_1/sweeps/wjf12n56).


## Question 6

The wandb visualisation for question 6 : [link](https://wandb.ai/cs23m056/CS23M056_DL_Assignment_1/sweeps/wjf12n56).

## Question 7

 - Identified the best model based on extensive experiments on Question 5 and Question 6 and noted down the hyperparameters.
 - Implemented Code to build **confusion matrix** and to integrate with wanddb.ai to generate a visual confusion matrix graph.
 - The best model configuration is-
  ```
  epochs = 10
  hiddenLayers = 4
  hiddenLayerSize = 64
  weightDecay = 0
  learningRate = 0.01
  optimizer = 'nadam'
  batchSize = 32
  weightInitialisation = 'Xavier'
  activationFunction = 'ReLU'
  lossFunction = 'cross entropy'
  beta = 0.9
  beta1 = 0.9
  beta2 = 0.999
  inputSize = 28*28
  outputSize = 10
  ```
The code for question 7 : [link](https://github.com/Rupak-Paul/Feedforward-Neural-Network/blob/main/Q7.py).
The wandb visualization for question 7 : [link](https://wandb.ai/cs23m056/CS23M056_DL_Assignment_1_Q7).

## Question 8
- By Fixing the hyperparameters and traind the model with 'cross validation' and 'squared loss' loss functions. we idetified that with cross
  entropy loss function accuracy is more than squared loss.
- Generated comparion plots in wandb.ai for accuracy and loss  

The code for question 8 : [link](https://github.com/Rupak-Paul/Feedforward-Neural-Network/blob/main/Q8.py).
The wandb visualisation for question 8 : [link]().

## Question 10
- Trained the netowrk with mnist dataset by taking best hyperparameters derived using fashion mnist dataset.
- Made some few changes to parameters to impove the mnist dataset accuracy. 

The code for question 10 : [link](https://github.com/Rupak-Paul/Feedforward-Neural-Network/blob/main/Q10.py).
The wandb visualisation for question 10 : [link]().

## Report
The report for this assignment : [link]().
