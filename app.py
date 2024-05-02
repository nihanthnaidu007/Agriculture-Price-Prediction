# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 19:55:28 2023

@author: sandeep
"""

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(page_title="Agriculture Price Prediction", page_icon="🌾")

# Load image
image = Image.open("pic_1.jpeg")


@st.cache(suppress_st_warning=True)
def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value
def get_value(val,my_dict):
    for key,value in my_dict.items(): 
        if val == key:
            return value
app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction']) #two pages


if app_mode=='Home':
    st.title("AGRICULTURE PRICE PREDICTION DASHBOARD")
    st.write("Agriculture is an important sector in India. It is indispensible for the sustenance and growth of the Indian economy. On an average, about 70% of the households and 10% of the urban population is dependent on agriculture as their source of livelihood. Today, India is a major supplier of several agricultural commodities like tea, coffee, rice, spices, oil meals, fresh fruits, fresh vegetables, meat and its preparations and marine products to the international market. India is a large producer of several agricultural products. In terms of quantity of production, India is the top producer in the world in milk, and second largest in wheat and rice. ")
    st.write("**Here are some visualizations regarding our project these will provide some analysis and price spent on each place.**")
    df=pd.read_csv("total.csv")
    
    #SCATTER PLOT
    st.header("SCATTER PLOT")
    x_axis=st.selectbox("Select X-axis feature",df.columns)
    y_axis=st.selectbox("Select Y-axis feature",df.columns)
    fig=px.scatter(df,x=x_axis,y=y_axis,hover_data=[x_axis,y_axis])
    st.plotly_chart(fig,use_container_width=True)
    
    
    #LINE CHART
    st.header("LINE CHART")
    x_axis=st.selectbox("Select the x-axis feature",df.columns)
    y_axis=st.selectbox("Select the y-axis feature",df.columns)
    fig=px.line(df,x=x_axis,y=y_axis,hover_data=[x_axis,y_axis])
    st.plotly_chart(fig, use_container_width=True)
    
    
    # Histogram
    st.subheader("Histogram")
    feature = st.selectbox("Select feature", df.columns)
    fig = px.histogram(df, x=feature)
    st.plotly_chart(fig, use_container_width=True)
    
    # BAR CHART
    mean_price = df.groupby("commodity")["modal_price"].mean()
    
    # Create a bar chart
    fig = px.bar(mean_price, x=mean_price.index, y=mean_price.values)
    
    # Add chart title and axes labels
    fig.update_layout(
        title="BAR CHART",
        xaxis_title="Category",
        yaxis_title="Mean Price"
)
    # Display the chart in Streamlit
    st.plotly_chart(fig)
    
    
    
    #SCATTER CHART
    st.header("3D-scatter plot")
    x=st.selectbox("Select the x_axis feature",df.columns)
    y=st.selectbox("Select the y_axis feature", df.columns)
    z=st.selectbox("Select the z_axis Feature", df.columns)
    color=st.selectbox("Select the color",df.columns)
    fig=px.scatter_3d(df,x=x,y=y,z=z,color=color)
    st.plotly_chart(fig,use_container_width=True)
    
    
    
    
    
   
    
    
elif app_mode=="Prediction":
    # Define the Streamlit app
    st.title('Agriculture Price Prediction')
    st.image(image, caption='My Image', use_column_width=True)
    # Load the model and label encoders
    model = pickle.load(open("agriculture_price_prediction_model.pkl", 'rb'))
    state_encoder = pickle.load(open("state_encoder.pkl", 'rb'))
    market_encoder = pickle.load(open("market_encoder.pkl", 'rb'))
    commodity_encoder = pickle.load(open("commodity_encoder.pkl", 'rb'))
    
    
    # Define a function to encode the input values
    def encode_input(state, market, commodity):
        state_encoded = state_encoder.transform([state])[0]
        market_encoded = market_encoder.transform([market])[0]
        commodity_encoded = commodity_encoder.transform([commodity])[0]
        return state_encoded, market_encoded, commodity_encoded
    
    
   
    # Define the input fields
    state = st.selectbox('Select the state', state_encoder.classes_)
    market = st.selectbox('Select the market', market_encoder.classes_)
    commodity = st.selectbox('Select the commodity', commodity_encoder.classes_)
    
    
    
        # Define the predict button
    if st.button('Predict'):
        # Encode the input values
        state_encoded, market_encoded, commodity_encoded = encode_input(state, market, commodity)
    
        # Create a DataFrame with the encoded input values
        input_df = pd.DataFrame({
            
           'state': [state_encoded],
           'market': [market_encoded],
           'commodity': [commodity_encoded]
       })
        # Make the prediction using the trained model
        prediction = model.predict(input_df)
        # Display the predicted price
        st.success(f'The predicted price is {prediction[0]:.2f}')
           
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
