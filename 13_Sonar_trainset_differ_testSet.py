from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy
import tensorflow as tf
##13_Sonar_trainSet_differ_testSet에 추가된 부분
from sklearn.model_selection import train_test_split
#model 저장을 위해서
from keras.models import load_model

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

#13_Sonar_trainSet_differ_testSet에 추가된 부분: 학습셋과 테스트셋의 구분 by sklearn.model_selection
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)
#모델 설정
model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

#13_Sonar_trainSet_differ_testSet에서 달라진 부분: 만들어진 model에 대한 accuracy test는 따로 testData를 이용
model.fit(X_train, Y_train, epochs=130, batch_size=5)
#모델을 저장하는 부분
model.save('my_model.h5')

del model
model = load_model('my_model.h5')
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))


