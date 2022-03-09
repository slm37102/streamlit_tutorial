import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.container()
dataset = st.container()
feature = st.container()
model_training = st.container()

# CSS
st.markdown(
    """"
    <style>
    .main {
        background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache
def get_data():
    return pd.read_csv('data/yellow_tripdata_2021-07.csv', low_memory=False)

with header:
    st.title('Hello World!')
    st.text('text')

with dataset:
    st.header('Dataset')
    st.text('test')
    taxi_data = get_data()
    st.write(taxi_data.head())

    st.subheader('Pick-up location ID distribution')
    pulocation_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts()).head(50)
    st.bar_chart(pulocation_dist)

with feature:
    st.header('feature')
    st.markdown('* **first feature:** I create')
    st.markdown('* **second feature:** I create')

with model_training:
    st.header('model')
    st.text('parm')
    
    sel_col, disp_col = st.columns(2)
    
    max_depth = sel_col.slider('max_depth', min_value=10, max_value=100, value=20, step=10)
    n_estimators = sel_col.selectbox('number of tree', options=[100,200,300,'No Limit'])
    input_feature = sel_col.text_input('feature', 'PULocationID')
    
    sel_col.text('list of feature:')
    sel_col.write(taxi_data.columns)
    
    if n_estimators == 'No Limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    
    X = taxi_data[[input_feature]]
    y = taxi_data[['trip_distance']]
    
    regr.fit(X,y)
    prediction = regr.predict(y)
    
    disp_col.subheader('MAE:')
    disp_col.write(mean_absolute_error(y, prediction))
    disp_col.subheader('MSE')
    disp_col.write(mean_squared_error(y, prediction))
    disp_col.subheader('R2')
    disp_col.write(r2_score(y, prediction))
    


