import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

# バッチ学習のSOM
class SOM(object):
    def __init__(self,
                 n_components=2, # ノード
                 resolution=20,
                 sigma_max=1.0,
                 sigma_min=0.2,
                 tau=20,
                 train_num=1000): # 学習回数
        if n_components != 2:
            raise (NotImplementedError())
        self.n_components = n_components
        self.resolution = resolution
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.tau = tau
        self.train_num = train_num

        self.y = None
        self.z = None
        self.k_star = None
        self.zeta = None

    def fit(self, data):
        self._initialize(data)
        for t in range(self.train_num):
            self._e_step(data)
            self._m_step(data, t)

    # 初期値をPCAにより求める
    def _initialize(self, data):
        self.zeta = create_zeta(self.resolution, self.n_components)
        pca = PCA(n_components=self.n_components)
        pca.fit(data)
        self.y = pca.inverse_transform(np.sqrt(pca.explained_variance_)[None, :] * self.zeta)

    # 現在の参照ベクトル集合の情報をもとに，勝者ノードのノード番号を推定
    # それに合わせて潜在変数の推定値も求める
    def _e_step(self, data):
        self.k_star = np.argmin(cdist(data, self.y, 'sqeuclidean'), axis=1)
        self.z = self.zeta[self.k_star, :]

    # 現在の潜在変数の情報をもとに，参照ベクトル集合を推定
    def _m_step(self, data, t):
        r = np.exp(-0.5 * cdist(self.zeta, self.z, 'sqeuclidean') / (self._sigma(t) ** 2))
        self.y = np.dot(r, data) / np.sum(r, axis=1)[:, None]

    # return sigma
    def _sigma(self, epoch):
        return self.sigma_min + (self.sigma_max - self.sigma_min) * np.exp(- epoch / self.tau)


def create_zeta(resolution, n_components):
    if n_components != 2:
        raise (NotImplementedError())
    mesh1d, step = np.linspace(-1, 1, resolution, endpoint=False, retstep=True)
    mesh1d += step / 2
    meshgrid = np.meshgrid(mesh1d, mesh1d)
    return np.dstack(meshgrid).reshape(-1, 2)

'''
---参考文献---
https://qiita.com/tohru-iwasaki/items/e51864269767ccc07254#fn1
'''