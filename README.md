# ai_number_read
AI that reads numbers

@RingaTech
https://www.youtube.com/watch?v=aFZEvQDTSyA&list=PLZ8REt5zt2Plph03SfZKijMWB68FUYfNc&index=2&t=4s

The neural network takes 28x28 image as input. The network has an entry layer with 784 neuron and uses 2 hidden layers with 64 neurons.The output has 10 neurons representing the prediction of the digit from 0-9. Every neuron in a layer connects to all neurons in the next layer (dense nmetwork). It uses more than one hiddel layer, so it is also a deep network.

Cost functions:
https://www.analyticsvidhya.com/blog/2021/02/cost-function-is-no-rocket-science/

The cost function determines the prediction error. The goal is to minimize this cost function.

The dataset used for training has 70.000 hadwritten numbers, which 60.000 are used to train the network and 10.000 for validation (supervised learning). 

#Requirements
Python 3.7

Tensorflow:
pip install tensorflow
