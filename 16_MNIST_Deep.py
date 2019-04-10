from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping

import matplotlib.pyplot as plt
import numpy
import os
import tensorflow as tf

#seed값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

#데이터 불러오기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
#같은 표현인지: X_train = X_train.reshape(X_train.shape[0], 784).astype('float32') / 255
print(X_train.shape)
print(X_train.shape[0])
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

#컨볼루션 신경망 설정
'''
입력된 이미지에 다시 한번 특징을 추출하기 위해 마스크(필터, 윈도 또는 커널)을 도입하는 기법
마스크가 적용되어 새롭게 만들어진 층을 컨볼루션(합성곱)이라고 함.
컨볼루션을 만들면 입력 데이터로부터 더욱 정교한 특징을 추출가능.
이렇게 마스크를 여러 개 만들 경우 여러 개의 컨볼루션이 만들어짐.
케라스에서 컨볼루션 층을 추가하는 함수는 
Conv2D(마스크 몇개, 마스크크기, 맨처음 층에는 입력되는 값을 알려줘야
.input_shape=(행, 열, 색상 또는 흑백), activation함수 정의)
'''
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),input_shape=(28, 28, 1), activation = 'relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
'''
여기에 맥스 풀링 층을 추가.
컨볼루션 층을 통해 이미지 특징을 도출. 그러나 그 결과가 여전히 크고 복잡하면 다시 한번 축소해야.
ex. maxPooling2D(pool_size=2:전체 크기가 절반으로 줄어듬): 
정해진 구역 안에서 가장 큰 값만 다음 층으로 ㄹ넘기고 나머지는 버림)
'''
model.add(MaxPooling2D(pool_size=2))
'''
과적합을 피하기 위한 기법:drop out
은닉층에 배치된 노드 중 일부를 랜덤하게 꺼주는 것.
'''
model.add(Dropout(0.25))
'''
convolution층 or maxpolling은 주어진 이미지를 2차원 배열인 채로 다루게 된다
을 거친 값을 기본층에 연결하려면 1차원으로 바꿔주야 하고, Flatten()
'''
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

#모델 최적화 설정
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss',patience=10)

#모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback, checkpointer])

#테스터ㅡ정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))

#테스트셋의 오차
y_vloss = history.history['val_loss']

#학습셋의 오차
y_loss = history.history['loss']

#테스트 정확도 출력
print("\n Test Accuracy: %4f" %(model.evaluate(X_test, Y_test)[1]))
#그래프로 표현
x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

#그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()