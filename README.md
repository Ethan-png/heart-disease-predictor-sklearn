# Machine Learning Model for Heart Disease Prediction

This repository contains a machine learning model implemented in Python using TensorFlow and Keras for predicting heart disease. The model is trained on preprocessed data and can be used to predict the presence or absence of heart disease based on various input features.

## Usage

To use this project, follow these steps:

1. Clone the repository to your local machine:
git clone https://github.com/your-username/heart-disease-prediction.git


2. Navigate to the project directory:
cd heart-disease-prediction


3. Install the required dependencies:
```python
pip install tensorflow pandas numpy matplotlib keras scikit-learn
```
4. Import the required libraries in your Python script:

```python
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
```
5. Load the preprocessed data:
```python
df = pd.read_csv("preprocessed_data.csv")
```
6. Split the data into input features (X) and target variable (y):
```python
x = df.iloc[:, :13].values
y = df["target"].values
```
7. Split the data into training and testing sets:
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)
```
8. Scale the input features using the StandardScaler:
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
```
9. Define and train the machine learning model:
```python
model = Sequential()
model.add(Dense(activation="relu", input_dim=13, units=8, kernel_initializer="uniform"))
model.add(Dense(activation="relu", units=14, kernel_initializer="uniform"))
model.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=8, epochs=100)
```
10. Evaluate performance:
```python
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
```
11. Make predictions using the trained model:
```
result = model.predict(np.array([x_test[3]]))
print(result)
```

# Contributing
If you wish to contribute to this project, you can fork the repository and create a pull request with your changes. Please ensure that your code follows the existing coding style and includes appropriate comments and documentation.

# License
This project is licensed under the MIT License. You are free to use, modify, and distribute this code for personal or commercial purposes.


