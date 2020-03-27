# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso

# データの読み込み
df = pd.read_csv("/Users/kidashuhei/deleted_NaN_risk_factors_cervical_cancer.csv", index_col=0)
# データのAgeからSchillerまで範囲指定
# 目的変数をCitologyに設定
X = df.loc[:, 'Age':'Schiller']
y = df.loc[:, "Citology"]
# 配列に格納
data = list(df.columns)

# Lasso回帰で交差検証を実行
# alphasで制約を調整
lasso = LassoCV(cv=5, alphas=10 ** np.arange(-6, 1, 0.1))
lasso.fit(X, y)

# 平均二乗誤差(MSE)の値が最小となる正則項ラムダ
minlam = np.argmin(lasso.mse_path_.mean(axis=-1))
minmse = np.amin(lasso.mse_path_.mean(axis=-1))
print(lasso.alphas_.min())
print(minmse)

# プロット
plt.figure()
plt.xlim(plt.xlim()[::-1])
plt.semilogx(lasso.alphas_, lasso.mse_path_.mean(axis=-1), "r.")
plt.semilogx(lasso.alphas_[minlam], minmse, "g o")
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.title('Cross-validation')
plt.axis('tight')
plt.show()

coefs = []

# 特徴量を配列に格納
for l in lasso.alphas_:
    result = Lasso(alpha=l)
    result.fit(X, y)
    coefs.append(result.coef_)

# プロット
plt.figure()
plt.semilogx(lasso.alphas_, coefs)
plt.legend(data, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=1,)

# 最小MSEの時のラムダの値に線を引く
plt.vlines(lasso.alphas_[minlam],-0.5,0.5)

plt.xlim(plt.xlim()[::-1])
plt.xlabel('Lambda')
plt.ylabel('coefs')
plt.show()
