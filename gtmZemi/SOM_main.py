import matplotlib.pyplot as plt
from gtmZemi.SOM import SOM
import pandas as pd

def main():

    # データ数
    f_samples = 300
    # input file
    df = pd.read_table('/Users/kidashuhei/SOM.dat', sep='\t')
    print(df.columns)
    # 配列に格納
    f_data = df.loc[:, "第一F":"第三F"]
    c_data = df.loc[:, '分類']
    # print(f_data)

    # learning from SOM class
    som = SOM()
    som.fit(f_data)

    # plot result
    for k in range(f_samples):
        plt.scatter(som.zeta[:,0],som.zeta[:,1], s=50) # 散布図
        plt.scatter(som.z[:,0], som.z[:,1], s=5, color='black')
        plt.annotate(c_data[k],xy=(som.z[k,0],som.z[k,1]),size=10) # 各要素にラベルを表示
        plt.title("SOM")
    plt.show()


if __name__ == '__main__':
    main()