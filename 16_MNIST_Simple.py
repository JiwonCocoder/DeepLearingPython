from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint,EarlyStopping

import matplotlib.pyplot as plt
import numpy
import os
import tensorflow as tf

#seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)
############데이터 전처리###################
#MNIST 데이터 불러오기
(X_train, Y_class_train), (X_test, Y_class_test) = mnist.load_data()
#각각 데이터의 갯수
print("학습셋 이미지 수: %d" % (X_train.shape[0]))
print("테스트셋 이미지 수: %d" %(X_test.shape[0]))

#하나의 이미지만 불러와서 확인:imshow()를 이용.첫번째 이미지를 흑백으로 출력
plt.imshow(X_train[0], cmap='Greys')
plt.show()
#하나의 class값만 부른 상황
print("class: %d" %(Y_class_train[0]))

'''
그리고 이 이미지 데이터는 숫자의 집합으로 바뀌어 학습셋으로 사용됨. 따라서 28X28개의 속성을 가진 데이터인것.
reshape(총샘플수, 1차원 속성의 수)함수를 사용. 28X28인 2차원배열을 784개의 1차원 배열로 바꿈. 
또한 케라스는 데이터를 0에서 1사이의 값을 변환해야 잘 작동.
따라서 0~255사이의 값을 0~1사이의 값으로 바꿔줘야 함. 
normalization:이렇게 데이터 폭이 클 때 적절한 값으로 분산의 정도를 바꾸는 과정 
'''
X_train = X_train.reshape(X_train.shape[0], 784).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 784).astype('float32') / 255
'''
np_utils.to_categorical(클래스, 클래스의 개수)함수를 이용
그런데 딥러닝의 분류 문제를 해결하려면 원-핫 인코딩 방식을 적용해야 한다.
0~9까지의 정수값을 갖는 현재 상태에서 0똔느 1로만 이루어진 벡터로 값을 수정해야 한다.

'''
Y_train= np_utils.to_categorical(Y_class_train, 10)
Y_test = np_utils.to_categorical(Y_class_test, 10)
print(Y_train[0])

######딥러닝 기본 프레임 만들기#########
#모델 프레임 설정
model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))

#모델 실행 환경 설정
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#모델 최적화 설정:자동 중단
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = "./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss',verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

#모델의 실행: 200개씩*30번 = 60000 이걸 30번 반복해야. 따라서 30 * 30 = 900번 수행됨
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback,checkpointer])

#테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))

#테스트셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']

#그래프로 표현
x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.',c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.',c="blue", label='Trainset_loss')

#그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('eploch')
plt.ylabel('loss')
plt.show()