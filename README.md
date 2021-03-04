# Stock-Price-Prediction-using-RNN
## Overview

In this work, we carry out a mix of machine learning algorithms to predict the future price of stock. This project predicts the stock price of the JP Morgan Chase and Co. Inc based on the stock price datasets from 2010 to June, 2019. The profit or loss calculation is determined by the closing price of a stock for the day. The adjusted closed price (almost exactly the same as the stock price) is used for this project. The method used is the Recurrent Neural Network (RNN) with the tensorflow in python script.

![image](https://user-images.githubusercontent.com/54149747/109921635-cc96ea80-7c81-11eb-8e5e-06900296cd2e.png)


## Motivation/Goal

One of the most challenging tasks is predicting the stock price. This is because the it involves so many rational and irrational behaviour which makes it very difficult to predict with large number of accuracies. A successful and accurate prediction of stock price will definitely yield significant profit. Many argued that predicting the future value of stock is unpredictable while others disagree. Machine learning algorithms have the accuracy potential for predictions. With the machine learning techniques, an unseen pattern could be accurately predicted. This project implemented machine learning algorithm using the Recurrent Neural Network (RNN) with the tensorflow in python. A recurrent neural network is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. It is an example of regression under supervised learning.

## Data Collection /Preparation
The data used for this work is the JP Morgan Chase and Co Inc. stock prices from 2010 through June 2019;  obtained through the yahoo finance data source which was filtered based on our months/years of preference. The link for the data is; https://finance.yahoo.com/quote/JPM/history?period1=1262325600&period2=1561870800&interval=1d&filter=history&frequency=1d

## Generate Features

We only considered the “Adjusted Close” column of the stock price data (nearly exactly the same as the “close” column). For a better model fit and very fast prediction with a lot of data, the features are normalized using the minMaxScaler function. With this, we are able to scale all the values between 0 and 1.

## Generate ML Model

In this part, the ML model we considered is the Recurrent Neural Network (RNN). We defined a sequential (linear stack of layers) tensorflow model and add the predefined layers in order to build the RNN model. The Tensorflow library is used to simplify this prediction. We have a total of 20 layers. These includes the sequential, LSTM (Long Term Short Memory), Drop-out and the Dense layer.

## Train the Model

This is the part where our dataset is being transformed. This simply means transforming our trained dataset into tensor data. After creating the tensor, we reshape our feature data into [x_train, (x_train.shape[0], x_train.shape[1], 1) ] as shown in the line of the code. Then, our tensor is now used with the model in the RNN. For the compilation, we set up the optimizer by using the “adam” since we want our learning rate to for each weight unlike the SGD (Stochastic Gradient Descent) whose learning rate is always the same for all weights. We also set up our loss function to be “meanSquaredError” and the metrics as “accuracy”. For the fitting, we call fit function on the tensorflow model and send the x-train (features) and y-train (response) sets. We set up our “epochs” as 200 (for accuracy and underfitting). After the training, it returns the result, then the model is set for making future predictions. It should be noted that: we decide to train the first 2000 datapoints of our data set and then, based our test on the remaining datapoints.

## Test the Model

In this part, we try to visualize how well our model fits the training sets by comparing both the actual and the predicted datasets. We use our model to predict on the same set that we train our data. This prediction is done by calling the prediction function. In other to get the real values of the feature, we run the “inverse min-max” by calling the minMaxInverseScaler function where we send the predicted data along with the min and max function.

## Result

![image](https://user-images.githubusercontent.com/54149747/109923019-d3bef800-7c83-11eb-9ffb-08d076d248a9.png)

From the result above, the red line shows the actual price while the blue line represents our predicted values. We observed that both are very close (approximately the same).
