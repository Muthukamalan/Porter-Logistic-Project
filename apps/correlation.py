import streamlit as st
import pandas as pd 
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

from pathlib import Path
csv_file = str(Path(__file__).parents[1]) +"/data/cleaned_df.csv"


def app():
    st.write("""
            # Correlation plot

        *Note        : I purposely reduce size whie doing pair plot because to render fast
    """)




    df = pd.read_csv(csv_file).sample(200)
    df.created_at = pd.to_datetime(df.created_at)
    df.actual_delivery_time = pd.to_datetime(df.actual_delivery_time)
    
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



    correlation = df.drop(['market_id','order_protocol','tkt_created_year','creat_day_is_weekend' , 'deliver_day_is_wekend','tkt_created_time','tkt_deliver_time','tkt_created_month','actual_delivery_time','created_at','store_id', 'tkt_create_day', 'store_primary_category','tkt_deliver_day','tkt_deliver_month', 'tkt_deliver_year'],axis=1).corr()
    correlation = correlation.where(np.triu(np.ones(correlation.shape)).astype(np.bool)).style.background_gradient(cmap="Blues")
    st.dataframe(correlation)

    correlation = df.drop(['market_id','order_protocol','tkt_created_year','creat_day_is_weekend' , 'deliver_day_is_wekend','tkt_created_time','tkt_deliver_time','tkt_created_month','actual_delivery_time','created_at','store_id', 'tkt_create_day', 'store_primary_category','tkt_deliver_day','tkt_deliver_month', 'tkt_deliver_year'],axis=1).corr()
    plt.tight_layout()
    _ = sns.heatmap(correlation.where(np.triu(np.ones(correlation.shape)).astype(np.bool)),annot=True, linewidths=0.3, cmap='RdYlGn', fmt=".1f");
    st.pyplot()
