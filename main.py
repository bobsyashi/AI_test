import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

data = pd.read_csv('num.csv')

data['YN'] = data['YN'].map({'yes': 0, 'no': 1})

data = data.apply(pd.to_numeric, errors='coerce')

x = data.drop(columns = ['number'])

y = data['number']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(f"x_train: {x_train.shape}")
print(f"y_train: {y_train.shape}")

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape, activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1,  activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=500)

model.evaluate(x_test, y_test)

