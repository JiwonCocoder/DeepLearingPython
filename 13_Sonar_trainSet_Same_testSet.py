from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy
import tensorflow as tf

#seed값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

#데이터입력
df = pd.read_csv('../dataset/sonar.csv',header=None)
print(df.head(5))

#data를 X,Y로 나누기
dataset = df.values
X = dataset[:, 0:60]
Y_obj = dataset[:,60]

#문자열 변환
e = LabelEncoder()
e.fit(Y_obj)
Y=e.transform(Y_obj)

#모델 설정
model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=200, batch_size=5)

print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))