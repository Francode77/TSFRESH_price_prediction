import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from binance import Client 
from sklearn.utils import all_estimators
from sklearn import base
import sklearn.metrics as metrics 

"""
This function will import all available sklearn estimators, checks if they are regression methods, and if so add them to the regressor dictionary. 
Not that we exclude a few estimators which produced bad results or longer processing times
"""
def init_regressors(ticker):
    
    regressor_names=[]
    estimators = all_estimators()
    print ('Number of available regression models: ',len(estimators))

    # Check if the estimator is of subclass Regressor, if so append them to our list
    for name, estimator in estimators:
        if issubclass(estimator, base.RegressorMixin):        
            regressor_names.append(name)

    # Make a dataframe from the estimators
    regressors=pd.DataFrame(regressor_names,columns=['name'])
    
    # These estimators cause errors, bad results or long processing times, so we can remove them
    regressors=regressors[regressors['name'] != 'QuantileRegressor']
    regressors=regressors[regressors['name'] != 'StackingRegressor']
    regressors=regressors[regressors['name'] != 'TheilSenRegressor']

    # We save the regressors that we can use to select the best one later 
    regressors.to_csv(f'regressors_{ticker}.csv',index=False)
    
    # When correctly saved, we can load them and turn them into a list of all the names
    regressors=pd.read_csv(f'regressors_{ticker}.csv')
    regressors=regressors['name'].values.tolist()


"""
This function will load the timeseries data from the Binance API and return it as a dataframe.
    
    ticker : our ticker (QNTUSDT)
    freq : the interval (4H)
    lookback : how many days to load from the api
    
 Note: We load only the "close" price.
 """

def load_data(ticker,freq,lookback):
    client=Client()
    frame=pd.DataFrame(client.get_historical_klines(ticker,freq.lower(),f'{lookback} ago UTC'))
    frame=frame[[0,4]]
    frame.columns=['Timestamp','close']
    frame=frame.astype(float)
    frame['Time']=pd.to_datetime(frame['Timestamp'],unit='ms')
    frame=frame.set_index('Time')
    frame.drop('Timestamp',inplace=True,axis=1)
    return frame


"""
This function is to evaluate the regression model's performance on the timeseries data
    y_orig : real close value
    y_pred : predicted close value
    name : name of the estimator
    MAE,MSE,RMSE,R2 : metric dictionaries containing all the values for all tested estimators
    new_removed_regressors : a list of estimators that have a too high RMSE 
"""

def get_metrics(y_orig,y_pred,name,MAE,MSE,RMSE,R2, new_removed_regressors,verbose):
 
    # Calculate the metrics for the current estimator
    mae = metrics.mean_absolute_error(y_orig,y_pred)
    mse = metrics.mean_squared_error(y_orig,y_pred)
    rmse = np.sqrt(mse) # or mse**(0.5)  
    r2 = metrics.r2_score(y_orig,y_pred)

    # Show the results for each estimator while testing
    if verbose==True:
  
        print(f"Sklearn.metrics for {name}:")
        print("MAE:",mae)
        print("MSE:", mse)
        print("RMSE:", rmse)
        print("R-Squared:", r2)
    

    # Add the metric values to their corresponding dictionary
    MAE[name]=mae
    MSE[name]=mse
    RMSE[name]=rmse
    R2[name]=r2 

    # Remove this estimators if rmse is greater than 25% and add it to new_removed_regressors list
    if abs(rmse)>(y_orig.values.mean()/4):
        print(' >>> REMOVING ',name)
        new_removed_regressors.append(name)

    # If the new_removed_regressors list is empty, add the 'None' value to it, so we can return it
    try:
        new_removed_regressors
    except NameError:
        new_removed_regressors = 'None' 

    # Return the metric dictionaries and the list with removed_regressors
    return MAE,MSE,RMSE,R2,new_removed_regressors

"""
With this function we search for the best regressor. The regressor with the smallest mean absolute error (MAE) is chosen.
"""
def get_best_regressor(X_train_selected,X_test,y_train,y,test_start,ticker,verbose):   

    # Create a metrics dictionary
    MAE={}
    MSE={}
    RMSE={}
    R2={} 

    # Create a list with regressors with too big error rates
    removed_regressors=[]      
    new_removed_regressors=[]
    
    # Get all estimators from sklearn
    estimators = all_estimators()
    print ('Number of available regression models: ',len(estimators))

    # Load all suitable estimators from .csv
    regressors=pd.read_csv(f'regressors_{ticker}.csv')

    # Remove the unsuitable estimators
    try: 
        removed_regressors=pd.read_csv(f'removed_regressors_{ticker}.csv')
        for regr_name in removed_regressors['name']:
            regressors=regressors[regressors['name'] != regr_name]
    except:
        removed_regressors=pd.DataFrame()

    # Make a list of all suitable estimators
    regressors=regressors['name'].values.tolist()

    """
     1. Loop through all estimators from the library, 
     2. Check if it is a regression model, 
     3. Check if it is in the suitable estimators list
     4. Fit on training data and predict values on testing data,
     5. Calculate the metrics
     5. Return the metrics and the non suitable estimators
    """
    for name, estimator in estimators:
        if issubclass(estimator, base.RegressorMixin): 
            if name in regressors:
                try:  
                    #print ('____________ ',estimator(),' _______________')

                    # Fit the estimator on the training data 
                    regression_model=estimator().fit(X_train_selected, y_train)

                    # Prepare the test data
                    X_test_selected = X_test[X_train_selected.columns]

                    # Predict values for test data
                    y_pred = pd.Series(regression_model.predict(X_test_selected), index=X_test_selected.index)

                    #this is for getting the metrics
                    y_orig=y[test_start:].dropna().to_frame()           
                    y_pred=y_pred[test_start:].dropna()
                    y_pred = y_pred[y_pred.index.isin(y_orig.index)]

                    MAE,MSE,RMSE,R2,new_removed_regressors=get_metrics(y_orig,y_pred,name,MAE,MSE,RMSE,R2, new_removed_regressors,verbose) 
  
                except:
                    continue
                
    return MAE,MSE,RMSE,R2,removed_regressors,new_removed_regressors

""" 
Function to plot a graph showing real versus predicted values
"""
def view_performance(test_results,train_days,freq):

    # Remove NaN value from test data
    test_results.dropna(inplace=True)

    # Calculate the metrics for the test results
    mae = metrics.mean_absolute_error(test_results.close,test_results.prediction)
    mse = metrics.mean_squared_error(test_results.close,test_results.prediction)
    rmse = np.sqrt(mse)  
    r2 = metrics.r2_score(test_results.close,test_results.prediction)

    # Plot the figure
    fig, ax = plt.subplots(figsize=(6,6))
            
    # Set the coordinates of the plot
    coordinates = [min(test_results.prediction), max(test_results.prediction)]

    # Set title and add a box with metrics
    ax.set_title('Trained on '+str(train_days)+' days / ' + str(freq))
    text = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR2 score: {r2:.2f}'
    fig.text(0.95, 0.95, text, fontsize='large', bbox=dict(facecolor='white', edgecolor='black', pad=5), horizontalalignment='right', verticalalignment='top')

    # Set labels
    ax.set_xlabel('Price Actual Values')
    ax.set_ylabel('Price Predicted Values')

    # Plot the data
    ax.scatter(test_results.close, test_results.prediction, color="orange")
    ax.plot(coordinates, coordinates)

    # Save and show the figure
    plt.savefig(f'Results_ARDR_{train_days}_days_training.png')
    plt.show()

