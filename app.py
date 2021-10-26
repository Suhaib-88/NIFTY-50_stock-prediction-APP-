import streamlit as st
import time
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt 
from plotly import graph_objs as go
from plotly import express as px

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

st.title("NIFTY-50 STOCK Price Prediction")

stocks =('SHREECEM', 'SUNPHARMA', 'TATAMOTORS', 'TCS', 'SBIN', 'NESTLEIND',
'NTPC', 'M&M', 'MARUTI', 'ONGC', 'POWERGRID', 'JSWSTEEL', 'KOTAKBANK',
'LT', 'ICICIBANK', 'INDUSBANK', 'INFY', 'IOC', 'ITC', 'HEROMOTOCO',
'HINDALCO', 'HINDUNILVR', 'HCLTECH', 'HDFCBANK', 'HDFC', 'DRREDDYS',
'EICHERMOTOR', 'GRASIM', 'CIPLA', 'COALINDIA', 'BPCL', 'BRITANNIA',
'ADANIPORTS', 'BAJAJFINSERV', 'BAJAJFINANCE', 'BHARTIARTL', 'AXISBANK',
'BAJAJ-AUTO', 'ASIANPAINT', 'UPL')

selected_stock = st.selectbox('Select a stock for prediction', stocks)

min_date = datetime.date(2019,5,31)
max_date = datetime.date(2021,10,13)

pred_date=st.date_input("Select a date and model predicts stock price",min_value=min_date,value=max_date,max_value=max_date)

if pred_date.weekday()==6 or pred_date.weekday()==5:
	st.warning('Please select a weekday (Mon-Fri)')
else:
	pass

print()

@st.cache
def load_data(date,ticker):
    data=pd.read_csv('Final-50-stocks .csv')
    data.reset_index(inplace=True)
    return data[[date,ticker]]

	
data_load_state = st.text('Loading data...')
data = load_data("DATE",selected_stock)
data_load_state.text('Loading data... done!')

df=data.copy()

plot_df=data.loc[:2130,["DATE",selected_stock]]	
st.subheader('Raw data')
st.write(plot_df.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=plot_df['DATE'], y=plot_df[selected_stock], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

data=data[selected_stock].reset_index(drop=True)

scaler= StandardScaler()
scaled_data= scaler.fit_transform(np.array(data).reshape(-1,1))


# Predict 
train_size = int(len(data)*0.75)
df_train=scaled_data[:train_size]
df_test=scaled_data[train_size:]

def make_dataset(dataset, time_step):
    x_data, y_data = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   
        x_data.append(a)
        y_data.append(dataset[i + time_step, 0])
    return np.array(x_data), np.array(y_data)

X_train,y_train=make_dataset(df_train,100)
X_valid,y_valid=make_dataset(df_test,100)

X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_valid = X_valid.reshape(X_valid.shape[0],X_valid.shape[1] , 1)

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_absolute_error',optimizer='adam')



if st.button("Predict"):

	with st.spinner('Fitting the model...'):
		model.fit(X_train,y_train,validation_data=(X_valid,y_valid),epochs=5,batch_size=128,verbose=1)
	st.success('Successfully fitted the model!')
	
	train_predict=model.predict(X_train)
	test_predict=model.predict(X_valid)

	##Transformback to original form
	train_predict=scaler.inverse_transform(train_predict)
	test_predict=scaler.inverse_transform(test_predict)


	look_back=100
	trainPredictPlot = np.empty_like(scaled_data)
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
			# shift test predictions for plotting
	testPredictPlot = np.empty_like(scaled_data)
	testPredictPlot[:, :] = np.nan
	testPredictPlot[len(train_predict)+(look_back*2)+1:len(scaled_data)-1, :] = test_predict
			
	all_np=np.concatenate((scaler.inverse_transform(scaled_data),trainPredictPlot,testPredictPlot),axis=1)
	ds=pd.DataFrame()
	ds['Actual-full_dataset']=all_np[:,0]
	ds['Pred_on_train_set']=all_np[:,1]
	ds['Pred_on_test_set']=all_np[:,2]
	new_df=pd.concat([df['DATE'],ds],axis=1)
	predicted_df=new_df.set_index('DATE')

	pred_df=predicted_df.loc[predicted_df.index >"2019-05-30","Pred_on_test_set"].to_frame()
	pred_value= pred_df.loc[str(pred_date)].values


	def previous_day_price(current_date):
		pred_value=new_df.loc[2133:,["DATE","Pred_on_test_set"]]
		preds= str(pred_value.loc[pred_value["DATE"]== str(current_date),"DATE"].values[0])[:10]
		date=datetime.datetime.strptime(preds,'%Y-%m-%d')-datetime.timedelta(1)
		return datetime.datetime.strftime(date,'%Y-%m-%d')
	
	try:
		prev_date=previous_day_price(pred_date)
		prev_price= pred_df.loc[str(prev_date)].values
		st.write(f"Last Traded Price(LTP) on {prev_date} : {prev_price}")

	except:
		pass
		
	try:
		st.write(f'The predicted stock price on {pred_date}:',pred_value)
	
	except:
		err = KeyError('An error has occured..Make sure a correct date is selected')
		st.write(f'{err}')

	st.text(f'Taking you to the prediction graph for {selected_stock}...')
	fig= px.line(predicted_df)
	# plot baseline and predictions
	time.sleep(6)
	fig.show()
