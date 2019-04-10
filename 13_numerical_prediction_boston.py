from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy
import pandas as pd
import tensorflow as tf

#seed값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv("../dataset/housing.csv", delim_whitespace=True, header=None)
#칼럼수 14:속성은 13개, 1개의 클래스.
dataset = df.values
X = dataset[:,0:13]
Y = dataset[:,13]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)
print(df.info())
print(df.head())

#선형 회귀 실행
'''
선형회귀 데이터는 마지막에 참과 거짓을 구분할 필요가 없다.
출력층에 활성화 함수를 지정할 필요도 없다.
'''
model = Sequential()
model.add(Dense(30, input_dim=13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train,Y_train, epochs=200, batch_size=10)

#예측 값과 실제값의 비교
'''
flatten()은 데이터 배열이 몇 차원이든 모두 1차원으로 바꿔 읽기 쉽게 해주는 함수입니다.
'''
print(model.predict(X_test))
Y_prediction = model.predict(X_test).flatten()
print(Y_prediction)
for i in range(10):
    label = Y_test[i]
    prediction = Y_prediction[i]
    print("실제가격: {:.3f}, 예상가격: {:.3f}".format(label,prediction))