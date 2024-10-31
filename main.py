import pandas as pd
import keras
from sklearn.model_selection import train_test_split

data = pd.read_csv['cancer.csv']

x = data.drop(columns = ['diagnosis(1=m, 0=b)'])

y = data['diagnosis(1=m, 0=b)']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(f"x_train: {x_train.shape}")
print(f"y_train: {y_train.shape}")