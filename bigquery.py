import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
import requests
import pandas as pd
import numpy as np
#FBProphet 
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from fbprophet.diagnostics import performance_metrics 


#need update
# Create API client
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)

st.sidebar.title("Options")

option = st.sidebar.selectbox("Which Dashboard?", ('wallstreetbets','Prophet','Time Series','twitter','stockwits'))

if option == 'stockwits':
    symbol = st.sidebar.text_input("Symbol", max_chars=5)
    st.subheader('stockwits')

    r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json")

    data = r.json()

    for message in data['messages']:
        st.image(message['user']['avatar_url'])
        st.write(message['user']['username'])
        st.write(message['created_at'])
        st.write(message['body'])

# QUERY = (
#     'SELECT * FROM `tvv-airflow-tutorial-demo.ARK_ETF.history`'
#      'LIMIT 100'
# )



@st.cache(ttl=600)
def run_query(query):
    query_job = client.query(query)
    rows_raw = query_job.result()
    # Convert to list of dicts. Required for st.cache to hash the return value.
    rows = [dict(row) for row in rows_raw]
    return rows

if option == 'Prophet':
    
    QUERY = 'SELECT * FROM `tvv-airflow-tutorial-demo.GME.history`'
    
    rows = run_query(QUERY)

    df = pd.DataFrame(rows)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date', ascending=False)
    df.reset_index(drop=True)
    st.dataframe(df)
    mapping = {df.columns[0]: 'ds', df.columns[4] : 'y'}
    df.rename(columns=mapping, inplace=True)
    
    
    df_truncate = df[['ds','y']]
     
    st.write('Adjust Model Parameters')
    daily_seasonality = st.checkbox('Daily Seasonality')
    st.write('Daily Seasonaility:', daily_seasonality)
    
    weekly_seasonality = st.checkbox('Weekly Seasonality')
    st.write('Daily Seasonaility:', weekly_seasonality)
    
    yearly_seasonality = st.checkbox('Yearly Seasonality')
    st.write('Daily Seasonaility:', yearly_seasonality)
    
    changepoint_range = st.number_input('ChangePoint Range between 0 and 1', min_value=0.00, max_value=1.00, step=0.01, format="%.2f")
    
    model_param = {"daily_seasonality": daily_seasonality, 
                   'weekly_seasonality': weekly_seasonality, 
                   'yearly_seasonality': yearly_seasonality,
                    'changepoint_range': changepoint_range, 
                    'changepoint_prior_scale':0.75}

    m = Prophet(**model_param) 
    m.fit(df_truncate)
    
    period = st.number_input('Specified the number of days to predict:', min_value = 1, max_value = 999)
    future = m.make_future_dataframe(periods=int(period))
   
    forecast = m.predict(future)
    #st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    fig1 = m.plot(forecast, figsize=(8,4))
    st.plotly_chart(fig1)
    
    actual = np.array(df_truncate[['y']])
    
    #Error Rate 
    
    st.dataframe(actual[:period])
    
    
