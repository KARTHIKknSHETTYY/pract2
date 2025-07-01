import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np
iris = load_iris()
X, y = iris.data, iris.target
model = LogisticRegression(max_iter=200).fit(X, y)
st.title("Iris Flower Species Prediction")
inputs = [st.slider(label,min_value=val[0],max_value = val[1],value = val[2])for label, val in zip([
    'Sepal Length','Sepal Width','Petal Length','Petal Width'],[(4.0, 8.0, 5.0), (2.0, 5.0, 3.0), (1.0, 7.0, 4.5), (0.1, 2.5, 1.5)])]
if st.button("Predict"):
    result = model.predict([inputs])
    st.success(f"The predicted species is: {iris.target_names[result][0]}")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 