import streamlit as st
import pandas as pd 
from datetime import datetime
import datetime
import pickle
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import io 
sns.set_theme()
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
cal = calendar()
st.set_option('deprecation.showPyplotGlobalUse', False)

import pickle



from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from category_encoders import TargetEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler




from pathlib import Path
csv_file = str(Path(__file__).parents[1]) +"/data/before_eda.csv"

model_file = str(Path(__file__).parents[1]) +"/models/gbdt_model.sav"

def app():

    df = pd.read_csv(csv_file)
        
    df.created_at = pd.to_datetime(df.created_at)
    df.actual_delivery_time = pd.to_datetime(df.actual_delivery_time)

    loaded_model = pickle.load(open(model_file, 'rb'))
    
    with st.form(key="form1"):
        col1, col2 = st.columns([2,15])
        with col1:
            col1.text("Customer Details:")
            market_id     = st.selectbox("Market ID" ,df['market_id'].unique() )
            tkt_created_time  = st.time_input("Time").strftime("%H")
            created_at    = st.date_input("Created Date",datetime.date(2015,1,1))
            order_protocol =st.selectbox("order protocol",df["order_protocol"].unique()  )
            total_items    = st.number_input("Total items")
            subtotal       = st.number_input("Subtotal",0)
            num_distinct_items  = st.number_input(" Number of Distinct Items",0)
            minimun_price = st.slider('Minimum Price',0,10)
            maximum_price = st.slider('Maximum Price',0,20000)
            
        with col2:
            col2.text("Store Details")
            store_id      = st.selectbox("Store ID",  df['store_id'].unique() )
            store_primary_category = st.selectbox("Store Primary Category",  df["store_primary_category"].unique()   )
            total_onshift_partners =st.number_input("Onshift partners",0)
            total_busy_partners  =st.number_input("Busy pateners",0)
            total_outstanding_orders=st.number_input("Outstanding orders",0)
            
        submit_button  =  col2.form_submit_button(label='Predict')
        if(submit_button):
            test_data = {
                    'created_at':created_at,
                    'market_id': str(market_id),
                    'tkt_created_time':int(tkt_created_time),
                    'store_id':str(store_id),
                    'order_protocol':str(order_protocol),
                    'store_primary_category':str(store_primary_category),
                    'total_items':int(total_items),
                    'subtotal':int(subtotal),
                    'num_distinct_items':str(num_distinct_items),
                    'min_item_price':int(minimun_price),     
                    'max_item_price': int(maximum_price),
                    'total_onshift_partners':int(total_onshift_partners),
                    'total_busy_partners':int(total_busy_partners),
                    'total_outstanding_orders':int(total_outstanding_orders)
                }
            test_data = pd.DataFrame(test_data,index=[0])
            test_data.created_at = pd.to_datetime(test_data.created_at)
            test_data['tkt_created_month'] = test_data.created_at.dt.month
            test_data['tkt_create_day']    = test_data.created_at.dt.day
            test_data['creat_day_is_weekend']=test_data.created_at.dt.dayofweek>4
            test_data.drop('created_at',axis=1,inplace=True)

            

            df = df.drop([
                'created_at',
                'actual_delivery_time',
                'store_id',
                'tkt_created_year',
                'tkt_created_month',
                'tkt_deliver_day','tkt_deliver_month','tkt_deliver_time','deliver_day_is_wekend','tkt_deliver_year'],axis=1)

            X = df.drop('delivery_duration',axis=1)
            y = df['delivery_duration']

            # Train-Test-Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
            encoder = TargetEncoder()
            encoder.fit(X_train[['market_id','store_primary_category','order_protocol','num_distinct_items']],y_train)

            test_data[['market_id','store_primary_category','order_protocol','num_distinct_items']] = encoder.transform(test_data[['market_id','store_primary_category','order_protocol','num_distinct_items']])

            test_data.drop(['total_busy_partners','store_id','total_outstanding_orders','tkt_created_month'],axis=1,inplace=True)


            # st.write(test_data)
            pred_time = loaded_model.predict(test_data)
            
            col2.header(f"Delivers in {pred_time} minutes.  point-estimate")

            std_dev=29.678452825040438
            z_critical=1.959963984540054

            worst_case = pred_time+(z_critical*std_dev)

            col2.header(f"In worst case ETA be {worst_case} minutes.  95% confidence-interval")

            
            plt.figure(figsize=(9,3))
            pd.Series(loaded_model.feature_importances_).plot(kind="bar")
            plt.xticks(range(loaded_model.feature_importances_.shape[0]),labels=test_data.columns)                                                                                   
            st.pyplot()