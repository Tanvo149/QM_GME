import pandas as pd 
from pandas.plotting import register_matplotlib_converters
import numpy as np
import matplotlib.pyplot as plt 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import re
import pmdarima as pm


df = pd.read_csv('data/nasdaq/csv/GME.CSV')
df.Date = pd.to_datetime(df.Date, format='%d-%m-%Y')
df.Date = pd.DatetimeIndex(df.Date).to_period('B')
df.set_index('Date', inplace=True)
df_truncate = df['2019-01-01':]

df_series = df_truncate['Close']

rolling = df_series.rolling(window=2)
rolling_mean = rolling.mean()
rolling_mean_0 = rolling_mean.fillna(0)
rolling_mean_truncate = rolling_mean_0[1:]

print(rolling_mean_0.iloc[-30:])



log_return = np.log(rolling_mean_truncate/rolling_mean_truncate.shift(1)).dropna()

x = pd.DataFrame(rolling_mean_truncate/rolling_mean_truncate.shift(1))

# result_adf = adfuller(log_return)
# print(result_adf[0])
# print(result_adf[1]) 


df_vol = df_truncate['Volume']

rolling_vol = df_vol.rolling(window=2)
rolling_mean_vol = rolling_vol.mean()
rolling_mean_0_vol = rolling_mean_vol.fillna(0)
rolling_mean_truncate_vol = rolling_mean_0_vol[1:]


log_vol= np.log(rolling_mean_truncate_vol/rolling_mean_truncate_vol.shift(1)).dropna()
exog = log_vol.values.reshape(-1,1)


model = pm.auto_arima(
    log_return,  d=0, 
    start_p=0, 
    start_q=0, 
    max_p=9, max_q = 9, 
    information_criterion='aic',
    trace=True,
    error_action='ignore'
   )

summary_string = str(model.summary())
param = re.findall('SARIMAX\(([0-9]+), ([0-9]+), ([0-9]+)',summary_string)
print(param)
p,d,q = int(param[0][0]) , int(param[0][1]) , int(param[0][2])
print(p,d,q)

