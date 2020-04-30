#!/usr/bin/env python
# coding: utf-8

# In[84]:


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from fbprophet import Prophet
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error 
import matplotlib
import matplotlib.pyplot as plt
import time
import random

get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams["figure.facecolor"] = "w"
plt.rcParams['figure.figsize'] = 14, 7


# # COVID-19 Deaths

# In[3]:


# load the Covid-19 deaths dataset
deaths_df = pd.read_csv("time_series_covid_19_deaths.csv")
# keep only the date columns
deaths_df = deaths_df.drop(columns=['Province/State', 'Country/Region', 
                       'Lat', 'Long'])

# aggregate the deaths from each Country by date
deaths_df = deaths_df.apply("sum", axis=0)
# set index to datetime type
deaths_df = deaths_df.reindex(pd.to_datetime(deaths_df.index))
# fix value for last date
deaths_df.at['2020-03-23'] = 16513

deaths_df.tail()


# In[4]:


# we have data from Jan. 22, 2020 to Mar. 23, 2020
deaths_df.plot(figsize=(14,7))
plt.title('Cumulative Global COVID-19 Deaths')
plt.ylabel("Cumulative Count")


# ## We use Prophet for prediction

# In[5]:


# use entire data as training set
# convert Series to Dataframe
deaths_train = deaths_df.reset_index(name='y')
deaths_train.rename(columns = {'index':'ds'}, inplace=True)

deaths_train.tail()


# In[6]:


# fit a Prophet object
proph = Prophet()
proph.fit(deaths_train)


# In[134]:


# extend dataframe into the future
future = proph.make_future_dataframe(periods=7)
future.tail(7)


# In[135]:


# make predictions on the extended dataframe
forecast = proph.predict(future)
forecast[['ds', 'yhat']].tail(7)


# In[136]:


# plot the forecast
fig1 = proph.plot(forecast)
plt.ylabel("Cumulative Count")
plt.xlabel(None)


# In[137]:


forecast_accuracy(forecast[forecast['ds'] < "2020-03-24"]['yhat'], deaths_train.y)


# ### COVID-19 Deaths Analysis
# 
# Because we only 62 days' worth of observations, a machine learning algorithm such as a LSTM Neural Network would not be appropriate. It might perfectly memorize the data and overfit, generalizing poorly to predict cumulative deaths for future dates. I decided instead to try the Prophet model developed by Facebook's Core Data Science team. It uses an additive model to forecast time series data. According to the developers, "It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well."
# 
# The plot of observations seems to display an exponential growth. However, the Prophet's predictions do not follow an exponential relationship. Rather, from Jan. 22 to Mar. 8, it looks like it the forecast is trying to capture a linear trend with weekly effects, which is why we see local optima occuring every seven days. From Mar. 8 to Mar. 30, the predictions again generally follow a linear trend with a greater slope and weekly effects (there are still peaks in the forecast occuring every seven days). 
# 
# Prophet performs quite well (perhaps predicting weekly effects where there are none) for dates up until Mar. 8, after which the linear growth will severely underestimate the true cumulative deaths from COVID-19. I do not recommend the Prophet model for predicting global deaths from COVID-19. Another model that incorporates an exponential shape assumption would be more appropriate.

# # SOYBEANS data

# In[140]:


# Data from https://www.macrotrends.net/2531/soybean-prices-historical-chart-data

soybeans_df = pd.read_csv("soybean-prices-historical-chart-data.csv")
# change date to datetime type
soybeans_df['date'] = pd.to_datetime(soybeans_df['date'])
soybeans_df.set_index('date', inplace=True)

soybeans_df.head()


# In[141]:


soybeans_df.tail()


# In[142]:


# We have daily closing price (USD per bushel) of soybeans from 
# Dec. 05, 1968 to Mar. 24, 2020
# Note that only 5 trading days per week

soybeans_df.plot(figsize=(14,7))
plt.ylabel("Price (USD/bushel)")
plt.axvline("2008-12-31", linestyle="--", c="black")
plt.axvline("2011-12-31", linestyle="--", c="black")


# ### Split into Training and Testing data

# In[143]:


soy_train = soybeans_df[soybeans_df.index <= "2008-12-31"]
soy_train.shape


# In[144]:


soy_val = soybeans_df[(soybeans_df.index > "2008-12-31") & (soybeans_df.index <= "2011-12-31")]
soy_val.shape


# In[145]:


soy_test = soybeans_df[soybeans_df.index > "2011-12-31"]
soy_test.shape


# ### Scale the data

# In[146]:


scaler = StandardScaler()

# fit the scaler with training data
train_arr = scaler.fit_transform(soy_train)

# apply scaler to val and test
val_arr = scaler.transform(soy_val)
test_arr = scaler.transform(soy_test)


# In[147]:


test_arr.shape


# ### Transforming the data into sequences

# In[148]:


def sequentialize(arr, seq_len):
    """
    A function to return sliding sequences of 
    feature and target from an array
    """
    x, y = [], []
    for i in range(len(arr) - seq_len):
        x_i = arr[i : (i + seq_len)]
        y_i = arr[(i + 1) : (i + seq_len + 1)]
        x.append(x_i)
        y.append(y_i)
    x_arr = np.array(x).reshape(-1, seq_len)
    y_arr = np.array(y).reshape(-1, seq_len)
    x_var = Variable(torch.from_numpy(x_arr).float())
    y_var = Variable(torch.from_numpy(y_arr).float())
    
    return x_var, y_var


# In[149]:


# set the sequence length
# 60 trading days = 1 quarter
seq_len = 60

x_train, y_train = sequentialize(train_arr, seq_len)
x_val, y_val = sequentialize(val_arr, seq_len)
x_test, y_test = sequentialize(test_arr, seq_len)


# In[150]:


x_test.shape


# In[151]:


def plot_sequence(axes, i, x_train, y_train, seq_no):
    window = len(x_train[0])
    axes[i].set_title("Sequence %d" % (seq_no + 1))
    axes[i].set_xlabel("Time Bars")
    axes[i].set_ylabel("Scaled Price")
    axes[i].plot(range(window), x_train[seq_no].cpu().numpy(), color="r", label="Feature")
    axes[i].plot(range(1, (window + 1)), y_train[seq_no].cpu().numpy(), color="b", label="Target")
    axes[i].legend()


# In[152]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
plot_sequence(axes, 0, x_train, y_train, 0)
plot_sequence(axes, 1, x_train, y_train, 999)


# # LSTM Neural Network

# In[153]:


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, input, future=0, y=None):
        outputs = []
        
        # reset the state of LSTM
        # the state is kept till the end of the sequence
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32)
        
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
            
        for i in range(future):
            if y is not None and random.random() > 0.5:
                output = y[:, [i]] # teacher forcing
            h_t, c_t = self.lstm(output, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
            
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


# In[154]:


class Optimization:
    """ A helper class to train, test and diagnose the LSTM"""

    def __init__(self, model, loss_fn, optimizer, scheduler):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_losses = []
        self.val_losses = []
        self.futures = []

    @staticmethod
    def generate_batch_data(x, y, batch_size):
        for batch, i in enumerate(range(0, len(x) - batch_size, batch_size)):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            yield x_batch, y_batch, batch

    def train(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        batch_size=100,
        n_epochs=15,
        do_teacher_forcing=None,
    ):
        seq_len = x_train.shape[1]
        for epoch in range(n_epochs):
            start_time = time.time()
            self.futures = []

            train_loss = 0
            for x_batch, y_batch, batch in self.generate_batch_data(x_train, y_train, batch_size):
                y_pred = self._predict(x_batch, y_batch, seq_len, do_teacher_forcing)
                self.optimizer.zero_grad()
                loss = self.loss_fn(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            self.scheduler.step()
            train_loss /= batch
            self.train_losses.append(train_loss)

            self._validation(x_val, y_val, batch_size)

            elapsed = time.time() - start_time
            print(
                "Epoch %d Train loss: %.2f. Validation loss: %.2f. Avg future: %.2f. Elapsed time: %.2fs."
                % (epoch + 1, train_loss, self.val_losses[-1], np.average(self.futures), elapsed)
            )

    def _predict(self, x_batch, y_batch, seq_len, do_teacher_forcing):
        if do_teacher_forcing:
            future = random.randint(1, int(seq_len) / 2)
            limit = x_batch.size(1) - future
            y_pred = self.model(x_batch[:, :limit], future=future, y=y_batch[:, limit:])
        else:
            future = 0
            y_pred = self.model(x_batch)
        self.futures.append(future)
        return y_pred

    def _validation(self, x_val, y_val, batch_size):
        if x_val is None or y_val is None:
            return
        with torch.no_grad():
            val_loss = 0
            for x_batch, y_batch, batch in self.generate_batch_data(x_val, y_val, batch_size):
                y_pred = self.model(x_batch)
                loss = self.loss_fn(y_pred, y_batch)
                val_loss += loss.item()
            val_loss /= batch
            self.val_losses.append(val_loss)

    def evaluate(self, x_test, y_test, batch_size, future=1):
        with torch.no_grad():
            test_loss = 0
            actual, predicted = [], []
            for x_batch, y_batch, batch in self.generate_batch_data(x_test, y_test, batch_size):
                y_pred = self.model(x_batch, future=future)
                y_pred = (
                    y_pred[:, -len(y_batch) :] if y_pred.shape[1] > y_batch.shape[1] else y_pred
                )
                loss = self.loss_fn(y_pred, y_batch)
                test_loss += loss.item()
                actual += torch.squeeze(y_batch[:, -1]).data.cpu().numpy().tolist()
                predicted += torch.squeeze(y_pred[:, -1]).data.cpu().numpy().tolist()
            test_loss /= batch
            return actual, predicted, test_loss

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")


# In[155]:


def generate_sequence(scaler, model, x_sample, future=1000):
    """ Generate future values for x_sample with the model """
    y_pred_tensor = model(x_sample, future=future)
    y_pred = y_pred_tensor.cpu().tolist()
    y_pred = scaler.inverse_transform(y_pred)
    return y_pred

def to_dataframe(actual, predicted):
    return pd.DataFrame({"actual": actual, "predicted": predicted})


def inverse_transform(scalar, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df


# ## Training the LSTM

# In[156]:


lstm_1 = LSTM(input_size=1, hidden_size=30, output_size=1)

loss_fun = nn.MSELoss()

optimizer_1 = optim.Adam(lstm_1.parameters(), lr=1e-3)
scheduler_1 = optim.lr_scheduler.StepLR(optimizer_1, step_size=5, gamma=0.1)

optimization_1 = Optimization(lstm_1, loss_fun, optimizer_1, scheduler_1)


# In[157]:


optimization_1.train(x_train, 
                     y_train, 
                     x_val,
                     y_val,
                     do_teacher_forcing=True)


# In[158]:


optimization_1.plot_losses()


# ## Evaluation 

# In[165]:


actual_1, predicted_1, test_loss_1 = optimization_1.evaluate(x_test, y_test, future=0, batch_size=60)

df_result_1 = to_dataframe(actual_1, predicted_1) 
df_result_1 = inverse_transform(scaler, df_result_1, ['actual', 'predicted'])

df_result_1.plot(figsize=(14, 7))
print("Test loss %.4f" % test_loss_1)


# In[171]:


len(predicted_1)


# In[172]:


df_result_1.iloc[1500:2000].plot(figsize=(14, 7))


# In[41]:


df_result_1.shape


# In[42]:


df_result_1.head()


# # Soy Prophet

# In[173]:


# set up training dataframe
soy_df_train = soybeans_df[soybeans_df.index < "2011-12-31"]
soy_df_train = soy_df_train.reset_index()
soy_df_train.rename(columns= {'date':'ds', ' value':'y'}, inplace=True)

soy_df_train.shape


# ## Training the Prophet

# In[174]:


# fit a Prophet model
soy_proph = Prophet()
soy_proph.fit(soy_df_train)


# In[175]:


# dataframe with all dates
soy_future = soybeans_df.reset_index()[['date']].rename(columns={'date':'ds'})

soy_future.shape


# ## Predictions

# In[176]:


soy_forecast = soy_proph.predict(soy_future)

soy_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)


# In[178]:


# plot the forecast
fig2 = soy_proph.plot(soy_forecast, figsize=(14,7))
plt.plot(soybeans_df[soybeans_df.index >= "2011-12-31"], 
        c='r', linestyle='-', label='True Prices')
plt.ylabel("Price (USD/Bushel)")
plt.xlabel(None)
plt.legend(loc='upper center')


# ## Evaluation

# In[211]:


mean_squared_error(soy_test[[' value']], soy_forecast[soy_forecast['ds'] >= "2011-12-31"][['yhat']])


# In[218]:


soy_fc = soy_forecast[soy_forecast['ds'] >= "2011-12-31"][['yhat']]
soy_fc = soy_fc.set_index(soy_test.index)


# In[220]:


forecast_accuracy(soy_fc['yhat'], soy_test[' value'])


# # ARIMA 
# 

# In[226]:


from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from numpy import log


# In[221]:


# null hypothesis is that the series is not stationary
result = adfuller(soybeans_df)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


# Since the p-value for the ADF is less than 0.05, we reject the null hypothesis and conclude that the series is stationary. We should set d=0.

# In[224]:


# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse,
            'corr':corr, 'minmax':minmax})


# In[225]:


# TRAINING DATA for the model
arima_train = soy_train.append(soy_val)


# # Automatic Step-wise AIC Criterion Model Selection

# In[227]:


# fit the model
model = pm.auto_arima(arima_train, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=False,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())


# In[228]:


model.plot_diagnostics()
plt.show()


# In[232]:


# Forecast
n_periods = len(soy_test)
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = soy_test.index

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(soybeans_df, label="Actual", color='orange')
plt.plot(fc_series, color='purple', label="Forecasted")
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15, label="95% Confidence Interval")

plt.legend()
plt.ylabel("Prices (USD/Bushel)")
plt.show()


# In[231]:


forecast_accuracy(fc, soy_test[' value'])


# In[ ]:





# In[222]:


# PACF plot of series
fig, axes = plt.subplots(1, 2, figsize=(14,7))
soybeans_df.plot(ax=axes[0])
plot_pacf(soybeans_df, ax=axes[1])

plt.show()


# Partial Autocorrelation looks significant up until Lag 2, so we'll fix p=2.

# In[223]:


# ACF plot of series
fig, axes = plt.subplots(1, 2, figsize=(14,7))
soybeans_df.plot(ax=axes[0])
plot_acf(soybeans_df.dropna(), ax=axes[1])

plt.show()


# It looks like we need 40 MA terms to remove autocorrelation in the series. This makes me suspicious.
