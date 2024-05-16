#%%

# SARIMA Model (Seasonality Auto Regressive Integrated Moving Average Model)
import pandas as pd
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def parser(a):
    return datetime.strptime(a, '%Y-%m-%d')

# Reading the catfish dataset
cfish_sales = pd.read_csv('C:/Users/SHAIFALI PATWAL/Desktop/Github Projects/catfish.csv', parse_dates=[0], index_col=0, date_parser=parser).squeeze()
cfish_sales.head()
cfish_sales.shape

cfish_sales = cfish_sales.asfreq(pd.infer_freq(cfish_sales.index))
cfish_sales.head()
cfish_sales.shape

#Setting the starting and ending date for our model
start_date = datetime(1994,1,1)
end_date = datetime(2000,1,1)
# Subsetting the catfish_sales from 1994-1-1 to 2000-1-1
lim_cfish_sales = cfish_sales[start_date:end_date]
lim_cfish_sales
#%%
#Plotting the graph for the catfish data from 1994 to 2000
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(lim_cfish_sales)
plt.title('Catfish Sales', fontsize=15)
plt.ylabel('Sales', fontsize=12)
plt.xlabel('Year', fontsize=12)
for year in range(start_date.year,end_date.year):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='r', alpha=0.3, linestyle='-')
    
    
# Here we can observe that there is an upwards trend in the data
# The mean in 1994 is around 18000 which increased to around 24000 in 1999. 
# That means there is a lot of variance among data
# We can observe the yearly repeated seasonal pattern in the graph.
#%%
#Removing the upward trend :

# taking the first difference to make this stationary trend
diff1 = lim_cfish_sales.diff()[1:]


# Plotting the graph :
plt.figure(figsize=(10,6))
plt.plot(diff1)
plt.title('Catfish Sales', fontsize=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Sales', fontsize=12)
for year in range(start_date.year,end_date.year):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), linestyle='-',alpha=0.2,color='r')
plt.axhline(0, color='k', linestyle='--', alpha=0.2)

# here we can see the graph is stationary

#%%

# Plotting ACF and PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(diff1, lags=20)
# here we can observe that we got the first lag in the 12 months so we should start with the seasonal MA process

plot_pacf(diff1,lags=20)
# We can observe that we can start AR with 12 month or 1year lag

#%%  Splitting the Data into Training and Testing sets

train_end = datetime(1999,7,1) # we are going to end our train data at 1999-7-1 
test_end = datetime(2000,1,1) # we are going to end the test data at 2000-1-1

train_data = lim_cfish_sales[:train_end]
train_data
test_data = lim_cfish_sales[train_end + timedelta(days=1):test_end]
test_data

#%% Fitting the SARIMA model

order1 = (0,1,0) # p, d, q for the AR, I and MA orders for non seasonal components
seasonal_order1 = (1, 0, 1, 12) # P, D,Q for the AR, I, MA for the seasonal component. m =12 as we have 12 points in between 1 lag

# Fitting the model 
from statsmodels.tsa.statespace.sarimax import SARIMAX
sarima_model = SARIMAX(train_data, order=order1, seasonal_order=seasonal_order1)
sarima_model_fit = sarima_model.fit()
sarima_model_fit.summary()

#Here we can observe that there are two parameters ar.S.L12  and ma.S.L12 in the model
# The ar.S.L12  parameter is for seasonal AR with 12 months lag. ma.S.L12  is a 12 months lag MA
# for AR part, the coefficient is 0.9452  which is positive shows the positive relationship between the 12 month laged series
# For MA part the coefficient is  -0.6917 which shows there is a negative relationship between the 12 month laged series
# p-values for both parameters are low so both of then are significant to predict and we can use them in the model.

#%%
#getting the predictions and residuals
sarima_predictions = sarima_model_fit.forecast(len(test_data))
sarima_predictions = pd.Series(sarima_predictions, index=test_data.index)
sarima_residuals = test_data - sarima_predictions

plt.figure(figsize=(10,4))
plt.plot(sarima_residuals)
plt.axhline(0, linestyle='--', color='r')
plt.title('SARIMA Residuals', fontsize=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Error', fontsize=12)

# Here we can observe that the residuls are negative in this case 

#%% 
# Plotting Prediction vs the real data
plt.figure(figsize=(10,6))
plt.plot(lim_cfish_sales)
plt.plot(sarima_predictions)
plt.legend(('Data', 'Predictions'), fontsize=12)
plt.title('Catfish Sales', fontsize=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Production', fontsize=12)
for year in range(start_date.year,end_date.year):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='r', alpha=0.2, linestyle='-') 

# Here we can see the predicted and actual trend looks similar but we are predicting more than the actual







