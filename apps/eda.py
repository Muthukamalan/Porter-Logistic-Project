import streamlit as st
import pandas as pd 
import datetime
import pickle
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

import matplotlib as mpl
plt.rcParams.update({'figure.max_open_warning': 0})

import io 
sns.set_theme()
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
cal = calendar()
st.set_option('deprecation.showPyplotGlobalUse', False)

from pathlib import Path
csv_file = str(Path(__file__).parents[1]) +"/data/cleaned_df.csv"


def app():
    st.title('EDA')
    st.write("note: It take some time to render and dataset also huge!! please bare with it and have a coffee!! ☕ ")
    # st.write("It's still under construction and need to add more functionality and details 🥹")
    df = pd.read_csv(csv_file)
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

    st.subheader("Univariate Analysis")

    for col in ['market_id','store_id','store_primary_category','order_protocol']:
        
        xidx = df[col].value_counts().shape[0]
        if(xidx>10):
            xidx=10
        fig, (ax1,ax2) = plt.subplots(1,2);
        _ = df[col].value_counts()[:xidx].plot(kind='bar',ax=ax1)
        _ = ax1.set_title(f'count plot of {col}')
        
        _ = plt.pie( 
                df[col].value_counts(dropna=False)[:xidx],
                autopct='%0.2f%%',
                labels=None,
                #labels=df[col].value_counts(dropna=False).index[:xidx]
        )
        _ = ax2.set_title(f"distribution of {col} Feature",fontsize='small')
        _ = ax2.legend(bbox_to_anchor=(1.25, 1.05),labels=df[col].value_counts(dropna=False).index[:xidx])
        st.pyplot()



    fig,[[ax1,ax2],[ax3,ax4],[ax5,ax6],[ax7,ax8]] = plt.subplots(4,2,figsize=(26,20));
    for col,axes in zip(df.select_dtypes('number').columns, [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]):
        _ = sns.boxplot(data=df,y=col,ax=axes);
    fig.suptitle("\n".join(["Numerical Univariate Analysis - Box Plot"]), y=0.98)
    st.pyplot()

    plt.figure(figsize=(6,9))
    _ = df.groupby('num_distinct_items')['store_primary_category'].get_group(2).value_counts()[:10].plot(kind='bar');
    plt.title("Item Vs count")
    st.pyplot()
    st.write("""
        `Inference`: Top 2 selling Items are American and pizza
    """)


    fig,ax = plt.subplots(1,1, figsize = (5,4))
    ax = sns.kdeplot( np.log(df.delivery_duration))
    plt.xlabel('log of delivery_duration')
    plt.ylabel('count')
    st.pyplot()
    st.write(" Suprisingly it follows log-NORMAL distribution with mean 3.79 and std 0.363 as business context 44.57 mins ")


    st.subheader("Bi-variate Analysis")

    st.write("Number of records a.k.a calls by each portocol")
    _ = pd.pivot_table(
        index=df.tkt_create_day,
        columns=df.order_protocol,
        data=df,
        aggfunc=np.count_nonzero
    )['market_id'].plot(kind='box',figsize=(12,6))
    plt.xlabel("order protocol")
    plt.ylabel("# calls")
    st.pyplot()

    st.write("Number of records a.k.a entires by each order_protocol by month")
    _ = pd.pivot_table(
        index=df.tkt_created_month,
        columns=df.order_protocol,
        data=df,
        aggfunc=np.count_nonzero
    )['created_at'].plot(kind='bar')
    st.pyplot()

    _ = pd.pivot_table(
        index=df.tkt_created_month,
        columns=df.order_protocol,
        data=df,
        aggfunc=np.count_nonzero
    )['created_at'].T.plot(kind="bar",figsize=(24,7))
    plt.xlabel("order protocol")
    plt.ylabel("# calls ")
    st.pyplot()

    st.write("""
        `Inference`: We seen significant increase in entries compare to previous month
    """)

    st.write("Number of records a.k.a entires by each order_protocol by date")
    _ = pd.pivot_table(
        index=df.tkt_create_day,
        columns=df.order_protocol,
        data=df,
        aggfunc=np.count_nonzero
    )['created_at'].T.plot(kind="box",figsize=(24,7))
    plt.xlabel("date")
    plt.ylabel("# calls ")
    st.pyplot()

    st.write("""
        `Inference`: Calls in the date range of 17 to 21 comparetively low.
    """)


    st.write("Number of records a.k.a entires by each order_protocol by day")
    weekly_pattern = pd.pivot_table(
        index=df.created_at.dt.day_name(),
        columns=df.order_protocol,
        data=df,
        aggfunc=np.count_nonzero
    )['created_at'].T
    weekly_pattern[['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']].plot(kind='box')
    plt.xticks(rotation=90)
    plt.xlabel('day')
    plt.ylabel('# calls')
    st.pyplot()

    st.write("""
        `Inference`: Calls in the wednesday is comparetively low with others

        `Business Insight`: If 17 to 21 is comes in mid of week run business with limited employee without business impact
    """)

    st.subheader("Multivariate Analysis")
    st.write('market_id vs delivery_duration by order_protocol')
    _ = sns.catplot(
        data=df,
        y='delivery_duration',
        x='market_id',
        hue='creat_day_is_weekend',
        kind="box",
        col='order_protocol',
        col_wrap=3
    )
    plt.ylim(0,100)
    st.pyplot()

    st.write("""
        `Inference`: Protocol 6 and 7 has so many fluxation 
    """)



    st.write('market_id vs number_of_partners by order_protocol')
    _ =sns.catplot(
        data=df,
        y='total_onshift_partners',
        x='market_id',
        hue='creat_day_is_weekend',
        kind="box",
        col='order_protocol',
        col_wrap=3
    )
    st.pyplot()



    _ = sns.catplot(
        data=df,
        y='delivery_duration',
        x='creat_day_is_weekend',
        kind="box",
        col='order_protocol',
        col_wrap=3
    )
    plt.ylim(0,100)
    st.pyplot()
    st.write("""
        `Inference`: every market is mainating same amount of workforce even is weekend and weekdays expect market_id=2
    """)

    st.subheader("Monitering via Creating Date")

    plt.figure(figsize=(40,9))
    _ = sns.boxplot(data=df,x='tkt_create_day',y='total_busy_partners',hue='market_id')
    st.pyplot()



    plt.figure(figsize=(40,9))
    _ = sns.boxplot(data=df,x='tkt_create_day',y='total_busy_partners',hue='market_id')
    st.pyplot()

    plt.figure(figsize=(40,9))
    _ = sns.boxplot(data=df,x='tkt_create_day',y='total_onshift_partners',hue='market_id')
    st.pyplot()


    plt.figure(figsize=(40,9))
    _ = sns.boxplot(data=df,x='tkt_create_day',y='delivery_duration',hue='market_id')
    plt.ylim(0,200)
    st.pyplot()


    st.subheader("Monitering via Created Time")

    plt.figure(figsize=(40,9))
    _ = sns.boxplot(data=df.loc[df.tkt_created_month==1],x='tkt_created_time',y='delivery_duration',hue='market_id')
    plt.ylim(0,200)
    st.pyplot()

    plt.figure(figsize=(40,9))
    _ = sns.boxplot(data=df,x='tkt_deliver_time',y='delivery_duration',hue='market_id')
    plt.ylim(0,200)
    st.pyplot()




    st.write("""
                    **Checking any co-relation btw parnters onshift and busy**
    """)
    plt.figure(figsize=(24,8))
    _= sns.jointplot(data=df, y='total_onshift_partners', x ='total_busy_partners',height=10,kind='scatter', hue='market_id')
    st.pyplot()


    st.write("""
                    **Checking via Delivery Date**
    """)
    plt.figure(figsize=(40,9))
    _ = sns.boxplot(data=df,x='tkt_deliver_day',y='total_busy_partners',hue='market_id')
    plt.title('ppls who are all busy when delivering')
    st.pyplot()


    st.write("""
                    **ppls who are all busy when delivering**
    """)
    plt.figure(figsize=(40,9))
    _ = sns.boxplot(data=df,x='tkt_create_day',y='delivery_duration',hue='creat_day_is_weekend')
    plt.ylim(0,100)
    st.pyplot()


    st.write("""
                    **Plot**
    """)
    _ = df.groupby(['market_id','order_protocol'])['subtotal'].median().reset_index()
    _ = sns.barplot(
        data=_,
        hue="order_protocol",
        x="market_id",
        y='subtotal'
    )
    st.pyplot()
    st.write("""    
                `Inference`: on 50% of Order protocol 7 in market_id 2 has more subtotal than others
    """)


    st.write("""
                    **Plot**
    """)
    _ = sns.barplot(
        df.groupby(['market_id','order_protocol'])['subtotal'].count().T.reset_index(),
        x='order_protocol',
        hue='market_id',
        y='subtotal'
    )
    st.pyplot()
    st.write("""    
                `Inference`: checking market_id  in terms of order protocol on basis count of subtotal
    """)

    st.write("""
                    **Plot**
    """)
    _ = sns.barplot(
        df.groupby(['market_id','order_protocol'])['subtotal'].median().T.reset_index(),
        x='order_protocol',
        hue='market_id',
        y='subtotal'
    )
    st.pyplot()

    st.write("""
                   `Inference`: checking market_id  in terms of order protocol on basis Meidan of subtotal
    """)
