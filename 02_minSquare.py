import numpy as np

x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

mx = np.mean(x)
my = np.mean(y)

#기울기 공식의 분모
divisor = np.sum([(mx - i)**2 for i in x])
#기울기 공식의 분자
def top(x, mx, y, my) :
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx)*(y[i] - my)
    return d
dividend = top(x, mx, y, my)

#기울기와 y절편 구하기
a = dividend / divisor
b = my - a*mx
#출력으로 확인
print("기울기:", a)
print("y절편", b)
