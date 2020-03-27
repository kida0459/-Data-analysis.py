# -*- coding: utf-8
import numpy as np
import pylab as plt
import random as rand


def activation_func(pr):
    if pr >= 0:
        a = 1
    else:
        a = 0
    return a


# 識別関数の本体：y=w'xを計算
def predict(wvec, xvec):
    out = np.dot(wvec, xvec)
    res = activation_func(out)
    return res


# 学習部：識別関数に学習データを順繰りに入れて、重みベクトルを更新
def train(wvec, xvec, label):
    res = predict(wvec, xvec)
    c = 0.2 # 学習係数
    wtmp = wvec + c * (label - res) * xvec
    return wtmp


if __name__ == '__main__':
    TRAIN_NUM = 400  # 学習データ
    TRAIN_NUM_H = int(TRAIN_NUM / 2)
    LOOP = 1000  # ループ回数
    wvec = [rand.uniform(-1, 1), rand.uniform(-1, 1), rand.uniform(-1, 1)]  # 重みベクトルの初期値

    # データ読み込み
    data = np.loadtxt("/Users/kidashuhei/SLP_data.dat", skiprows=1)
    np.set_printoptions(threshold=10000)

    # データ正規化
    x_max = max(max(data[:, 0]), max(data[:, 1]))
    x_min = min(min(data[:, 0]), min(data[:, 1]))
    for i in range(TRAIN_NUM):
        data[i, 0] = (data[i, 0] - x_min) / (x_max - x_min)
        data[i, 1] = (data[i, 1] - x_min) / (x_max - x_min)

    # 初期グラフの設定
    x_initial = np.arange(0, 1, 0.1)
    y_initial = [-(wvec[0] / wvec[1]) * xi - (wvec[2] / wvec[1]) for xi in x_initial]

    # 教師ラベルを1 or 0で振る
    s_labels = np.ones(TRAIN_NUM)
    s_labels[298:TRAIN_NUM] = 0

    data[:, 2] = np.ones(TRAIN_NUM)
    # print(data[:, 2])

    # グラフ作成
    plt.figure("Simple-Perceptron")

    # ループ回数の分だけ繰り返しつつ、重みベクトルを学習させる
    # 終了条件：学習が終了した時
    for j in range(LOOP):
        count = 0
        for i in range(TRAIN_NUM):
            tmp = wvec
            wvec = train(wvec, data[i, :], s_labels[i])
            if (tmp == wvec).all():
                count += 1
        if TRAIN_NUM == count:
            print(j) # 総学習回数
            break

    print(wvec)
    # 分離直線を引く
    x_fig = np.arange(0, 1, 0.1)
    y_fig = [-(wvec[0] / wvec[1]) * xi - (wvec[2] / wvec[1]) for xi in x_fig]

    # 分離対象と分離直線をグラフに表示
    plt.xlim(0, 1)
    plt.ylim(-1, 1.5)
    plt.scatter(data[0:297, 0], data[0:297, 1], c='b', label="x1", s=5)
    plt.scatter(data[298:TRAIN_NUM, 0], data[298:TRAIN_NUM, 1], c='r', label="x2", s=5)
    plt.plot(x_fig, y_fig, c='g', label="After train")
    plt.plot(x_initial, y_initial, c='black', label="Before train")

    plt.legend()
    plt.grid()

    plt.show()
# -*- coding: utf-8
import numpy as np
import pylab as plt
import random as rand


def activation_func(pr):
    if pr >= 0:
        a = 1
    else:
        a = 0
    return a


# 識別関数の本体：y=w'xを計算
def predict(wvec, xvec):
    out = np.dot(wvec, xvec)
    res = activation_func(out)
    return res


# 学習部：識別関数に学習データを順繰りに入れて、重みベクトルを更新
def train(wvec, xvec, label):
    res = predict(wvec, xvec)
    c = 0.2 # 学習係数
    wtmp = wvec + c * (label - res) * xvec
    return wtmp


if __name__ == '__main__':
    TRAIN_NUM = 400  # 学習データ
    TRAIN_NUM_H = int(TRAIN_NUM / 2)
    LOOP = 1000  # ループ回数
    wvec = [rand.uniform(-1, 1), rand.uniform(-1, 1), rand.uniform(-1, 1)]  # 重みベクトルの初期値

    # データ読み込み
    data = np.loadtxt("/Users/kidashuhei/SLP_data.dat", skiprows=1)
    np.set_printoptions(threshold=10000)

    # データ正規化
    x_max = max(max(data[:, 0]), max(data[:, 1]))
    x_min = min(min(data[:, 0]), min(data[:, 1]))
    for i in range(TRAIN_NUM):
        data[i, 0] = (data[i, 0] - x_min) / (x_max - x_min)
        data[i, 1] = (data[i, 1] - x_min) / (x_max - x_min)

    # 初期グラフの設定
    x_initial = np.arange(0, 1, 0.1)
    y_initial = [-(wvec[0] / wvec[1]) * xi - (wvec[2] / wvec[1]) for xi in x_initial]

    # 教師ラベルを1 or 0で振る
    s_labels = np.ones(TRAIN_NUM)
    s_labels[298:TRAIN_NUM] = 0

    data[:, 2] = np.ones(TRAIN_NUM)
    # print(data[:, 2])

    # グラフ作成
    plt.figure("Simple-Perceptron")

    # ループ回数の分だけ繰り返しつつ、重みベクトルを学習させる
    # 終了条件：学習が終了した時
    for j in range(LOOP):
        count = 0
        for i in range(TRAIN_NUM):
            tmp = wvec
            wvec = train(wvec, data[i, :], s_labels[i])
            if (tmp == wvec).all():
                count += 1
        if TRAIN_NUM == count:
            print(j) # 総学習回数
            break

    print(wvec)
    # 分離直線を引く
    x_fig = np.arange(0, 1, 0.1)
    y_fig = [-(wvec[0] / wvec[1]) * xi - (wvec[2] / wvec[1]) for xi in x_fig]

    # 分離対象と分離直線をグラフに表示
    plt.xlim(0, 1)
    plt.ylim(-1, 1.5)
    plt.scatter(data[0:297, 0], data[0:297, 1], c='b', label="x1", s=5)
    plt.scatter(data[298:TRAIN_NUM, 0], data[298:TRAIN_NUM, 1], c='r', label="x2", s=5)
    plt.plot(x_fig, y_fig, c='g', label="After train")
    plt.plot(x_initial, y_initial, c='black', label="Before train")

    plt.legend()
    plt.grid()

    plt.show()
