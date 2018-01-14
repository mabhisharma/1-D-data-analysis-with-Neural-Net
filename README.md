# 1-D-data-analysis-with-Neural-Net
Analysis of how Neural Networks transform the data so that non linear classification boundary can be learned

<p>To get a better sense of what’s going on, let’s consider an dataset that’s 1-dimensional: </p>
<p>Class A - [-1/3, 1/3]</p>
<p>Class B - [-1, -2/3] &cup; [2/3, 1]</p>

The representation of the data can be shown as follows -
</br>
<p align="center">
  <img src="/Plots/One-Dimensional-data.png" alt="One dimensional data with two classes" height="300" width="400" />
</p>

<p> I have taken 500 samples from both the classes. As we can see this One Dimensional data cannot be seperated using a linear boundary. So if we use Neural Network with 2 layers i.e. Hidden Layer and Output Layer (input layer not included in number of layers), with Hidden Layer having only 1 hidden unit, we will not be able to classify this one dimensional data. To classify this one dimensional data we will require minimum 2 hidden units.</p>

# Network Details
<p> In order to classify this particular data we will require minimum 2 hidden units in the hidden layer. So our network becomes</br> Input layer which is one dimensional, Hidden layer with 2 units and Output layer which outputs 0 or 1 classifing the data into Class A or Class B. The activation function we will use in the hidden layer is Tanh and activation function used in the output layer is sigmoid because we want to classify input as 0 and 1. Also to understand what this neural network is doing we will simply store the output from hidden layer. We will use gradient descent procedure to optimize the weights and bias vectors. Now as the hidden layer has 2 units, the input has transformed into 2 dimensional data and which we store output in every gradient descent step. We also store the weights and bias vectors of layer 2. Now see the diagram for the neural network below</p>
<p align="center">
  <img src="/Plots/1-D-NN.PNG" alt="Neural Network for 2 class problem" height="150" width="200" />
</p>
