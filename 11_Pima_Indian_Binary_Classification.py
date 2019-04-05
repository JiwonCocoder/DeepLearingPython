from keras.models import Sequential
from keras.layers import Dense
import numpy
import tensorflow as tf

#seed값 생성
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

#데이터 로드:숫자로만 이루어져 있으니까 nunmy를 이용해서 데이터를 불러오고.
dataset = numpy.loadtxt("../dataset/pima-indians-diabetes.csv", delimiter = ",")
X = dataset[:,0:8]
Y = dataset[:, 8]

#모델의 설정
model = Sequential()
#은닉층 1: node 12 input차원 8, activation 함수 =relu
model.add(Dense(12, input_dim = 8, activation ='relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#모델 컴파일
model.compile(loss = 'binary_crossentropy', optimizer ='adam', metrics = ['accuracy'])

#모델 실행
model.fit(X, Y, epochs =200, batch_size = 10)

#결과 출력
print("\n Accuracy : %.4f" % (model.evaluate(X,Y)[1]))