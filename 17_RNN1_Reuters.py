'''
LSTEM을 이용한 로이터 뉴스 카테고리 분류하기
과거에 입력된 데이터와 나중에 입력된 데이터 사이의 관계를 고려해야할 경우.
앞서 입력받은 데이터가 얼마나 중요한지를 판단하여 별도의 가중치를 줘서 다음 데이터로 넘어감.
모든 입력 값에 이 작업을 순서대로 실행하므로 다음 층으로 넘어가기 전에 같은 층을 맴도는 것처럼 보임.
순환이 되는 가운데 앞서 나온 입력에 대한 결과가 뒤에 나오는 입력 값에 영향을 주게 된다.
LSTM은 반복되기 직전에 다음 층으로 기억된 값을 넘길지 안넘길지를 관리하는 단계를 하나 더 추가하는 것.
'''
#로이터 뉴스 데이터셋 불러오기
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils

import numpy
import tensorflow as tf
import matplotlib.pyplot as plt

#seed값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

#불러온 데이터를 학습셋과 테스트셋으로 나누기
'''
num_words의 의미:단어빈도 1~1000만 사용하겠다.
X_train[0]은 첫번째 데이터의 값을 나타내는데 단어가 아니라, 그 단어가 해당 데이터안에 몇번 들어 있는지를 의미하는 것.
이러한 작업을 위해 tokenizer()같은 함수를 사용하는데, 케라스는 이 작업을 이미 마친 데이터를 불러올 수 있다.
또한, 데이터적처리 함수 sequence를 사용하여 단어수를 maxlen=100으로 맞추게 한다. 부족하면 0으로 채움. 넘치면 100까지만 사용
'''
(X_train, Y_train), (X_test, Y_test) = reuters.load_data(num_words=1000, test_split=0.2)

#데이터 확인하기
category = numpy.max(Y_train) + 1
print(category, '카테고리')
print(len(X_train), '학습용 뉴스 기사')
print(len(X_test),'텟흐트용 뉴스 기사')
print(X_train[0])

#데이터 전처리: sequence함수를 사용해, 데이터당 단어수를 100으로 맞춤.
x_train  =sequence.pad_sequences(X_train, maxlen=100)
x_test = sequence.pad_sequences(X_test, maxlen=100)
#데이터 전처리:원-핫 인코딩 처리를 하여 정답값이 0,1값으로 표현되도록 바꿔줌.
y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(Y_test)

#모델의 설정
model = Sequential()
'''
Embedding층은 데이터 전처리 과정을 통해 입력된 값을 받아 다음 층이 알아들을 수 있는 형태로 전환하는 역할.
Embedding('불러온 단어의 총 개수', '기사당 단어 수')형식으로 사용하며, 모델 설정 부분의 맨 처음에 있어야 함.
LSTM은 RNN에서 기억 값에 대한 가중치를 제어.
LSTM('기사당 단어수', '기타옵션')
'''
model.add(Embedding(1000,100))
model.add(LSTM(100, activation='tanh'))
model.add(Dense(46, activation='softmax'))

#모델의 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#모델의 실행
history = model.fit(x_train, y_train, batch_size=100, epochs=20, validation_data=(x_test, y_test))

#테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))

#테스트셋의 오차
y_vloss = history.history['val_loss']
#학습셋의 오차
y_loss = history.history['loss']
#그래프로 표현
x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red',label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label="Trainset_loss")
#그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('eploch')
plt.ylabel('loss')
plt.show()

