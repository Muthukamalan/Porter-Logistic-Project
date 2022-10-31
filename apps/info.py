import streamlit as st
import pandas as pd 
import datetime
import pickle
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import io 
sns.set_theme()
import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore")
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
cal = calendar()
st.set_option('deprecation.showPyplotGlobalUse', False)

from pathlib import Path
csv_file = str(Path(__file__).parents[1]) +"/data/cleaned_df.csv"



def app():
    st.title('Welcom to Porter')

    st.write("This is home page")
    st.write("""


    
                                            # Porter Delivery Management System

        ### **About Porter**

            Porter is India's Largest Marketplace for Intra-City Logistics. Leader in the country's $40 billion intra-city logistics market, Porter strives to improve the lives of 1,50,000+ driver-partners by providing them with consistent earning & independence. Currently, the company has serviced 5+ million customers

            Porter works with a wide range of restaurants for delivering their items directly to the people.

            Porter has a number of delivery partners available for delivering the food, from various restaurants and wants to get an estimated delivery time that it can provide the customers on the basis of what they are ordering, from where and also the delivery partners.


        ### **Problem Statement** 


            Determining the delivery time of the order depending on the time of order, deliver partners available and etc.


    """)

    df = pd.read_csv(csv_file).sort_values(by='created_at')
    df.created_at = pd.to_datetime(df.created_at)
    df.actual_delivery_time = pd.to_datetime(df.actual_delivery_time)

    st.dataframe(df)

    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    
    st.write("""
            ### Missing value HEATMAP 
    """)
    

    st.write("Number of Missing Value")
    _ = plt.figure(figsize=(24,6));sns.heatmap(df.sample(100).isna(), linewidths=.2,cmap="YlGnBu"); 
    plt.xlabel("features")
    st.pyplot()

    st.write("""
            WOW!!! Zero null values
        [Wanna more about how I clean dataset ](https://github.com/Muthukamalan/Porter-Projetc)
    """)

    
    df['tkt_created_year'] = df.created_at.dt.year
    df['tkt_deliver_year']  = df.actual_delivery_time.dt.year


    df['tkt_created_month']= df.created_at.dt.month
    df['tkt_deliver_month'] = df.actual_delivery_time.dt.month


    df['tkt_create_day']   = df.created_at.dt.day
    df['tkt_deliver_day']   = df.actual_delivery_time.dt.day


    df['tkt_created_time']  = df.created_at.dt.hour
    df['tkt_deliver_time']  = df.actual_delivery_time.dt.hour


    df['creat_day_is_weekend'] = df.created_at.dt.dayofweek>4
    df['deliver_day_is_wekend']= df.actual_delivery_time.dt.dayofweek>4




