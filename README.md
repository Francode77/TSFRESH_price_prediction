# TSFRESH_price_prediction
A demonstration of tsfresh feature extraction library to predict price of a crypto asset.

With this project, I demonstrate how the tsfresh library can be applied for building a regression model on market data. The aim is to predict the value of the next data point in a given timeseries. 

The model will predict the price from the beginning of the current month until now. The model is trained on a user selected number of days, prior to the beginning of the current month.

It can be used to predict the price with a regression model for a point in time, based on the lookahead value. This will predict the price for this value's next datapoint.

# Prerequisites 

This script was tested on Python 3.7.9

# Installation

Install the necessary libraries from requirements.txt

# Usage

Run the file `predict_price.ipynb` from a Jupyter notebook IDE
The functions are stored in `functions.py`

## Choose the variables
The following variables can be chosen freely:

- ticker : crypto asset ticker that is available on Binance 
- freq : interval ('1D','4H', ...)
- train_days: number of days of training data
- init : do not (0) or (1) calculate the best regressor
- lookahead : predict price for this number of datapoints in the future
- verbose : whether to output the regression metrics during calculation

## Other variables:
TSFresh rolling window size
- max_window_size : max length of the rolled window for feature extraction
- min_window_size : minimum number of days for a rolling window


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

Selecting the best_regressor takes a little bit of time on first run, but this time is reduced greatly on second runs through the removal of the worst performing regressors.

The higher the number of days in the training data, the better our model scores. Here we see the predicted values versus the actual values from a regression model that has been trained on 720 days of training data with 4 hour intervals. 
![image](https://user-images.githubusercontent.com/113235815/219980203-56757dcb-8f38-4348-ba62-97c13a7a3472.png)

We can also plot the predicted values for this month versus the actual values to see how well our regression model performs. This is particularly useful to see the effect of the number of days in the training data

![image](https://user-images.githubusercontent.com/113235815/219980214-c4781397-b758-456a-9d14-83b4ac88dda1.png)

We can also predict the value of a future datapoint.

# Improvements

We could improve this code to 
- See how well it performs for selling / buying, ie. just to use it as a sell or buy indicator
- We could extend the range for predictions by making it predict on a rolling window of predictions
- We could extract only the relevant features to reduce the size of the dataframes

# Conclusion

While this script is by all means not intended to predict the price of a crypto asset, it demonstrates the power of the tsfresh library.

# Contributors

This script was written by [Frank Trioen](https://www.linkedin.com/in/frank-trioen-21b71135)
 
