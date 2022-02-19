import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import pandas_datareader as web
import streamlit as st
import yfinance as yf
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential

# Setting up the streamlit UI
st.title('Crypto Dashboard')
st.header('made by Anurag')

tickers = {'BTC', 'ETH'}
tickers2 = {'GBP', 'EUR'}

dropdown = st.selectbox('Pick your asset here ', tickers)
dropdownAgainst = st.selectbox('Pick your against asset here ', tickers2)

# Allowing user to pick the date from a dropdown selection
startx = st.date_input('Start', value=pd.to_datetime('2020-01-01'))
endx = st.date_input('End', value=pd.to_datetime('today'))

# Letting the user go as far back as 01/01/2016
start = dt.datetime(2016, 1, 1)
end = dt.datetime.now()

# Declaring the initials
crypto_currency = dropdown

# Declaring the actual currency so we can get the BTC price
against_currency = dropdownAgainst
yf.pdr_override()
df = pdr.get_data_yahoo(f'{crypto_currency}-{against_currency}', start, end)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df['Close'] = df['Close'].astype(float)
print(df.isna().any())
# Dropping all the rows with nan values
df.dropna(inplace=True)
print(df)

df['Close'].plot(figsize=(16, 6))
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Printing actual data
#st.line_chart(scaled_data)



# 60 timesteps and 1 output
prediction_days = 60
future_day = 0
x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data) - future_day):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x + future_day, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Create the network
model = Sequential()
model.add(BatchNormalization())
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation="sigmoid"))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Show a progress bar while the model fits to the data
with st.spinner("Please wait ..."):
    model.fit(x_train, y_train, epochs=2, batch_size=32)

# Create the test data
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

if (startx != test_start):
    test_start = startx
if (endx != end):
    test_end = endx

yf.pdr_override()
test_df = pdr.get_data_yahoo(f'{crypto_currency}-{against_currency}', test_start, test_end)

# print(df.columns)
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df['Close'] = test_df['Close'].astype(float)

# Dropping all the rows with nan values
test_df.dropna(inplace=True)

# Printing df
# test_df --> prints the whole test data
actual_prices = test_df['Close'].values
total_dataset = pd.concat((df['Close'], test_df['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_df) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)
# print(model_inputs)

x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
# print(model.predict(x_test))

prediction_prices = scaler.inverse_transform(prediction_prices)



actual_prices = actual_prices.T
actual_prices = actual_prices.flatten()
prediction_prices = prediction_prices.T
prediction_prices = prediction_prices.flatten()

# accuracy_score(y_true, y_pred, normalize=False)

# This is how I would calculate the mean squared error
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

# accuracy_score(actual_prices, prediction_prices, normalize=False)


error = mean_absolute_error(actual_prices, prediction_prices)

from sklearn.metrics import mean_squared_error

error2 = np.sqrt(mean_squared_error(actual_prices, prediction_prices))
# report error

errors = list()
TT = 0

for i in range(len(actual_prices)):
    err = abs((actual_prices[i] - prediction_prices[i]))
    errors.append(err)
    if err <= error:
        TT = TT + 1
    # print('>%.1f, %.1f = %.3f' % (actual_prices[i], prediction_prices[i], err))

# plot errors
plt.plot(errors)
plt.xticks(ticks=[i for i in range(len(errors))], labels=prediction_prices)
plt.xlabel('Predicted Value')
plt.ylabel('Mean Absolute Error')
# plt.show()

success = TT / len(actual_prices)

# print(TT/len(actual_prices))
df_actual = DataFrame(actual_prices)
df_predicted = DataFrame(prediction_prices)

# printing predicted and actual values in a line chart
dataset = pd.DataFrame()

dataset['actual prices'] = df_actual
dataset['predicted prices'] = df_predicted
#df_actual = df_actual.append([[] for _ in range(future_day)], ignore_index=True)#actual data appended as much as future days
#dataset['predicted prices'] = dataset['predicted prices'].shift(future_day)

# change index to date
idx = pd.date_range(startx, periods=len(dataset.index), freq='D')

dataset.set_index(idx, inplace=True)

# dataset

st.caption('Here is the result table ...')
st.line_chart(dataset)
float(format(success, '.2f'))
st.markdown(f'Models current succes rate is : ' + str(float(format(success, '.2f'))))

# Didn't think giving the RMSE was valid, success rate is enough :)

#st.markdown(f'Root Mean Squared Error : ' + str(float(format(error2, '.2f'))))
st.balloons()

#streamlit run C:\Users\anura\PycharmProjects\Playground\lstm-crypto-dashboard.py