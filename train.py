import pandas as pd 
import numpy as np
import matplotlib as plt
import pickle
from sklearn.model_selection import train_test_split

data=pd.read_csv("data.csv")

data.dropna(inplace=True)

data.dropna(inplace=True)
x=data.drop(['bedrooms', 'bathrooms', 'sqft_living', 'city'], axis=1)
y=data['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

y_train_df = y_train.to_frame(name='target_price')

train_data = pd.concat([x_train, y_train_df], axis=1)

#training the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)

# Save model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)