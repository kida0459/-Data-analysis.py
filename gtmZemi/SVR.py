# coding:utf-8
# import
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# 変数の初期化
SST = 0  # ２乗和の総和
SSR = 0  # ２乗回帰の総和
totalR2 = 0
totalCV = np.poly1d([])
linear = []
path = "/Users/kidashuhei/testdata.dat"

# データを読み込みx,yをそれぞれに格納
x = np.array([], dtype=np.float64)
y = np.array([], dtype=np.float64)
xy = np.array([], dtype=np.float64)

with open(path) as f:
    data = f.readline()
    while data:
        data = data.replace("\n", "")
        dataxy = data.split('\t')
        x = np.append(x, float(dataxy[0]))
        y = np.append(y, float(dataxy[1]))
        data = f.readline()


xy = np.stack([x, y])  # x,yを結合した行列

# xとyの平均値
x_avr = np.mean(x)
y_avr = np.mean(y)

# x,yの分散、共分散
s_xy = np.cov(xy, rowvar=1, bias=1)

# w1とw0の計算し、回帰式を定義
w1 = s_xy[0][1] / s_xy[0][0]
w0 = y_avr - w1 * x_avr
p = np.poly1d([w1, w0])  # w1 * x + w0

# 線形回帰分析
for i in range(0, len(x)):
    linear.append(float(p(x[i])))

# SSTとSSR、R2の計算
SST = ((y - y_avr) ** 2).sum()
SSR = ((y - linear) ** 2).sum()
R2 = 1 - (SSR / SST)

# グラフを表示する
plt.plot(x, y, 'o', x, p(x), '-')
plt.title("Linear Regression")
plt.show()
plt.cla()

print("------------------線形回帰------------------")
print("SST:", SST)
print("SSR:", SSR)

print("R^2:", R2)
print("回帰式:", p)

print("------------------SVRwith交差検証------------------")
# SVRwith交差検証
for i in range(1, 6):
    # 元データを分割し格納
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True)

    # SVR
    svr_model = SVR(kernel='linear', gamma=0.1, epsilon=0.001)
    svr_model.fit(x_train.reshape(-1, 1), y_train)

    # 解析結果
    svr_y = svr_model.predict(x_test.reshape(-1, 1))

    # xとyの平均値
    x_avr = np.mean(x_test)
    svr_y_avr = np.mean(svr_y)

    xy = np.stack([x_test, svr_y])  # x_test,svr_yを結合した行列

    # x,yの分散、共分散
    s_xy = np.cov(xy, rowvar=1, bias=1)

    # w1とw0の計算し、回帰式を定義
    w1 = s_xy[0][1] / s_xy[0][0]
    w0 = svr_y_avr - w1 * x_avr
    p = np.poly1d([w1, w0])  # w1 * x + w0

    # SSTとSSR、R2の計算を行う
    SST = ((y_test - np.mean(y_test)) ** 2).sum()
    SSR = ((y_test - svr_y) ** 2).sum()
    R2 = 1 - (SSR / SST)

    print("--------", i, "回目-------")
    print("回帰式:", p)
    print("R^2:", R2)
    totalR2 += R2
    totalCV += p

totalCV = totalCV / 5

print('平均の回帰式:', format(totalCV))
print('平均のR^2:', (totalR2 / 5))

plt.plot(x, y, 'o', x, totalCV(x), '-')
plt.title("SVR")
plt.show()
