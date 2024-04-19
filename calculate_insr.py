import pandas as pd
import streamlit as st
from sklearn.metrics import r2_score

df = pd.read_csv("insurance.csv")

df['Hospital charges']=df['charges']
df = df.drop(columns=['charges'])
df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
df.replace({'smoker': {'no': 0, 'yes': 1}}, inplace=True)
df.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)
#initialising the x and y data ... 
x = df.iloc[:,:6]
y = df.loc[:,['Hospital charges']]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=0)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
#object initialise
log = LinearRegression()
dcs = DecisionTreeRegressor()
#fitting the data inside and performing the training
log.fit(x_train,y_train)
dcs.fit(x_train,y_train)

st.header("Insurance Predictor")
st.sidebar.header("User Input")
age = st.sidebar.slider("Age",1,65,17)
sex = st.sidebar.selectbox("Sex",("Male","Female"))
bmi = st.sidebar.slider("BMI",10.0,40.0,0.1)
children = st.sidebar.slider("No of Children",0,5,0)
smoker = st.sidebar.selectbox("Smoker ?",("Yes","No"))
region = st.sidebar.selectbox("Region",("Northwest","Southwest"))

sex_encoded = 0 if sex == "Male" else 1
smoker_encoded = 1 if smoker == "Yes" else 0
region_encoded = 0 if region == "Northwest" else 1


pred1 = log.predict([[age,sex_encoded,bmi,children,smoker_encoded,region_encoded]])
pred3 = log.predict(x_test)

pred2 = dcs.predict([[age,sex_encoded,bmi,children,smoker_encoded,region_encoded]])
pred4 = dcs.predict(x_test)


st.subheader("Predicted Insurance by Linear Regression")
st.write(pred1)
st.subheader("Accuracy")
st.write(r2_score(y_test,pred3))

st.subheader("Predicted Insurance by Decision Tree")
st.write(pred2)
st.subheader("Accuracy")
st.write(r2_score(y_test,pred4))





