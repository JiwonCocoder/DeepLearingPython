from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf

#seed 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

#데이터 입력:string이 있으므로 numpy보다는 panda로 데이터를 불러오는 상황
df = pd.read_csv('../dataset/iris.csv', names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

#그래프로 상관관계를 확인:seaborn libaray의 pairplot함수를 써서 속성별 상관도를, matplotlib로 그래프를 그리고
sns.pairplot(df, hue='species');
plt.show()

#불러온 데이터를 X,Y로 나눠준다.
dataset = df.values
X = dataset[:,0:4].astype(float)
Y_obj = dataset[:,4]

#클래스 이름을 숫자 형태로 바꿔줘야. sklearn라이브러리의 LabelEncoder():array([1,2,3])
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

#원-핫 인코딩 keras.util의 np_utils.categorical()함수를 적용해서, 각각의 Y값이 숫자 0,1로 이루어지게 : array([[1., 0., 0.].....])
Y_encoded = np_utils.to_categorical(Y)

######모델 생성 ######
model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
#소프트맥스란, 클래스 값의 총합을 1인 향태로 바꿔서 계산해주는 것. 이럴경우, 큰 값은 더 크게, 작은 값은 더 작게 나타나게됨
#->따라서 원-핫 인코딩 값으로 전환시킬 수 있게 됨.
model.add(Dense(3, activation='softmax'))

#모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#모델 실행
model.fit(X, Y_encoded, epochs=50, batch_size=1)

#결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y_encoded)[1]))