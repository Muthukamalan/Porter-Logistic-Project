import streamlit as st
st.set_page_config(
        page_title="Porter Logistics",
        layout='wide'
)
from multiapp import MultiApp
from apps import eda, data_stats,info,pair_plot,correlation # import your app modules here
import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore")
	

app = MultiApp()



# Add all your application here

app.add_app("Problem Statement",info.app)
app.add_app("EDA", eda.app)
app.add_app("correlation", correlation.app)
app.add_app("PairPLot", pair_plot.app)
app.add_app("Internal Portal", data_stats.app)


# The main app
app.run()