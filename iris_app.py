import streamlit as st 
from KNN import KNN
import numpy as np 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 

iris=load_iris()

X, y = iris.data, iris.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


model = KNN(k=5)
model.fit(X_train, y_train)  

st.title = (" 🌸 Iris Flower Classifier ")
st.write("Enter the flower's measurements: ")

sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)


input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data) 

species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

st.subheader("Prediction: ")
st.success(f"The model predicts this is {species_map[prediction[0]]}** 🌼")