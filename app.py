import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("linreg.pkl", "rb"))

st.title("Brain Weight Prediction")

genders = ["Male", "Female"]
ages = ["x > 18", "x < 18"]

gender = st.selectbox("Gender", genders)
age = st.selectbox("Age Range", ages)
head_size = st.number_input("Head Size (cm3)")

if st.button("Predict"):
	gender = genders.index(gender) + 1
	age = ages.index(age) + 1
	test = np.array([[gender, age, head_size]])
	res = model.predict(test)
	print(res)
	st.success("Predicted: " + str(res[0]))
