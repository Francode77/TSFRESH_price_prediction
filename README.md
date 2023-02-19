# TSFRESH_price_prediction
A demonstration of tsfresh feature extraction library to predict price of a crypto asset.

With this project, I demonstrate how the tsfresh library can be applied for building a regression model on market data. The aim is to predict the value of the next data point in a given timeseries. 

The model will predict the price from the beginning of the current month until now. The model is trained on a user selected number of days, prior to the beginning of the current month.

# Prerequisites 

This script was tested on Python 3.7.9

# Installation

Install the necessary libraries from requirements.txt

# Usage

Run the file predict_price from a Jupyter notebook IDE

## Choose the variables
The following variables can be chosen freely:

- ticker : crypto asset ticker that is available on Binance 
- freq : interval ('1D','4H', ...)
- train_days: number of days of training data
- init : do not (0) or (1) calculate the best regressor
- lookahead : predict price for this number of datapoints in the future

# Method

1. Setting variables
2. Import libraries
3. Loading data
4. Feature extraction
5. Select the best sklearn regression model
6. Train the model with the best regression method
7. Plot the graph
8. View the performance
9. Get the prediction

# Results

We can plot the predicted values for this month versus the actual values to see how well our regression model performs. The higher the number of days in the training data, the better our model will score. 

We can also predict the value of the next datapoint.

# Improvements

We could improve this code to 
- See how well it performs for selling / buying, ie. just to use it as a sell or buy indicator
- We could extend the range for predictions by making it predict on a rolling window of predictions
- We could extract only the relevant features to reduce the size of the dataframes

# Conclusion

While this script is by all means not intended to predict the price of a crypto asset, it demonstrates the power of the tsfresh library.



![image](https://user-images.githubusercontent.com/113235815/219978114-45f0dcb4-4f11-45f2-95c3-74f65f350033.png)
