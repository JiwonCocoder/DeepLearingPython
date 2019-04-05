#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

#실행할 때마다 같은 결과를 출력하기 위한 seed값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)
#데이터 입력
x_data = np.array([[2, 3], [4, 3], [6, 4], [8, 6], [10, 7], [12, 8], [14, 9]])
y_data = np.array([0, 0, 0, 1, 1, 1, 1]).reshape(7, 1)

#텐서플로에서 데이터를 담는 플레이스 홀더를 정해준다. 행에는 제한이 없고, 열의 경우 2, 1로 고정이 되어 있는 것.
X = tf.placeholder(tf.float64, shape=[None, 2])
Y = tf.placeholder(tf.float64, shape=[None, 1])
#a,b를 임의로 정한다.
#입력값 :2 , 나가는 값:1 개라는 뜻.
a = tf.Variable(tf.random_uniform([2, 1], dtype=tf.float64))
b = tf.Variable(tf.random_uniform([1], dtype=tf.float64))

#시그모이드 함수의 방정식: 행렬곱을 이용
#y = 1/(1 + np.e**(a*x_data + b))
y = tf.sigmoid(tf.matmul(X, a) + b)
#loss를 구하는 함수
loss = -tf.reduce_mean(Y*tf.log(y) + (1-Y) * tf.log(1 - y))

#학습률 값
learning_rate = 0.1

#loss를 최소하는 값 찾기-by gradient decent
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

predicted = tf.cast(y > 0.5, dtype=tf.float64)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float64))
#학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(3001):
        a_, b_, loss_, _ = sess.run([a, b, loss, gradient_decent], feed_dict={X: x_data, Y: y_data})
        if (i+1) % 300 == 0:
            print("step=%d,  loss=%.4f, a1=%.4f, a2=%.4f, b=%.4f" % (i + 1, loss_, a_[0], a_[1], b_))

#new_x = np.array([7, 6.]).reshape(1,2)
    new_x = np.array([7, 6.]).reshape(1, 2)  # [7, 6]은 각각 공부 시간과 과외 수업수.

#new_y = sess.run(y, feed_dict={X: new_x})
    new_y = sess.run(y, feed_dict={X: new_x})

    print("study time: %d, extra study time: %d " % (new_x[:, 0], new_x[:, 1]))
    print("pass possibility: %6.2f %%" %(new_y*100))
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#
 #   for i in range(3001):
  #      a_, b_, loss_, _ = sess.run([a, b, loss, gradient_decent], feed_dict={X: x_data, Y: y_data})
   #     if (i + 1) % 300 == 0:
    #        print("step=%d, a1=%.4f, a2=%.4f, b=%.4f, loss=%.4f" % (i + 1, a_[0], a_[1], b_, loss_))