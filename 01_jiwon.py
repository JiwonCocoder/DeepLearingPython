from keras.models import Sequential
from keras.layers import Dense

import numpy
import tensorflow as tf

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)
#데이터를 불러와서 사용할 수 있게 만들어 주는 부분
'''
numpy라는 라이브러리를 이용했으며,
그 numpy 라이브러리 안에 있는 loadtxt()라는 함수를 사용해, 외부 데이터셋을 콤마로 분리하여, 
Data_set이라는 임시 저장소에 저장시킴
'''
Data_set = numpy.loadtxt("../dataset/ThoraricSurgery.csv", delimiter=",")
'''
X:속성 데이터셋
Y:클래스 데이터셋
속성끼리, 클래스끼리 데이터셋을 만들어줘야함.
'''
X = Data_set[:,0:17]
Y = Data_set[:,17]

#2.딥러닝 실행
'''
케라스를 사용해 딥러닝을 실행시킴.
케라스가 구동되려면 텐서플로 라이브러리가 미리 설치되어 있어야 함.
텐서플로를 케라스를 이용해 쉽게 접근할 수 있다는 것 같다.
Keras의 Sequential함수를 이용해 딥러닝의 구조를 한 층 한 층 필요한 만큼 쉽게 쌓아올릴 수 있다. by model.add()
Dense()함수를 이용해, 각 층이 제각각 어떤 특성을 가질지 옵션을 설정하는 역할을 함.
'''
model = Sequential()
'''
units:output space의 dimensionality를 의미.
output arrays of shape(*,30)...2D input에 대한 output
input arrays of shape(*,17)...2D input...2D tensor임을 말한다.
'''
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
'''
딥러닝 구조와 층별 옵션을 정하고 나면 compile()함수를 이용해 이를 싱행시킴.
loss: 한 번 신경망이 실행될 때마다 오차 값을 추척하는 함수
optimizer :오차를 어떻게 줄여 나갈지 정하는 함수
activation: 다음 층으로 어떻게 넘어갈지를 결정하는 부분. relu, sigmoid함수를 사용
'''
model.compile(loss='mean_squared_error', optimizer='adam',  metrics=['accuracy'])
model.fit(X, Y, epochs=30, batch_size=10)

#3. 결과 출력
'''
model.evaluate()함수를 이용해 딥러닝 모델의 정확도를 점검
현재는 기존의 데이터셋을 이용해, 새 환자인 것으로 가정하고 테스트한 것.
'''
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))
