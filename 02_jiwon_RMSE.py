import numpy as np
#기울기 a와 y절편
ab = [3, 76]
#x,y의 데이터 값
data = [[2, 81],[4, 93],[6,91],[8,97]]
x = [i[0] for i in data]
y = [i[1] for i in data]
#y = ax+b에 a와 b값을 대입하여 결과를 출력하는 함수
def predict(x):
    return ab[0]*x + ab[1]

#RMSE 함수
def RMSE(predict_y, y):
    return np.sqrt(((predict_y - y)**2).mean())

#RMSE함수를 각 y값에 대입하여 최종 값을 구하는 함수
def RMSE_val(predict_result,y):
    return RMSE(np.array(predict_result), np.array(y))
#에측 값이 들어갈 빈 리스트
predict_result = []
#모든 x값을 한 번씩 대입하여
for i in range(len(x)) :
    #predict_result 리스트를 완성한 다. 
    predict_result.append(predict(x[i]))


#최종 RMSE출력
print("rmse 최종값:" + str(RMSE(np.array(predict_result),np.array(y))))