# coding: utf-8


import sys
from python_environment_check import check_packages
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
from distutils.version import LooseVersion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# # PyTorchとScikit-Learnによる機械学習  
# # -- コード例

# ## パッケージバージョンチェック

# check_packages.pyスクリプトから読み込むためにフォルダをパスに追加:



sys.path.insert(0, '..')


# 推奨パッケージバージョンをチェック:





d = {
    'numpy': '1.21.2',
    'matplotlib': '3.4.3',
    'sklearn': '1.0',
    'pandas': '1.3.2'
}
check_packages(d)


# # 第3章 - Scikit-Learnを使った機械学習分類器の概観

# ### 概要

# - [分類アルゴリズムの選択](#Choosing-a-classification-algorithm)
# - [scikit-learnでの最初のステップ](#First-steps-with-scikit-learn)
#     - [scikit-learnでのパーセプトロンの訓練](#Training-a-perceptron-via-scikit-learn)
# - [ロジスティック回帰によるクラス確率のモデリング](#Modeling-class-probabilities-via-logistic-regression)
#     - [ロジスティック回帰の直感と条件付き確率](#Logistic-regression-intuition-and-conditional-probabilities)
#     - [ロジスティック損失関数の重みの学習](#Learning-the-weights-of-the-logistic-loss-function)
#     - [scikit-learnでのロジスティック回帰モデルの訓練](#Training-a-logistic-regression-model-with-scikit-learn)
#     - [正則化による過学習への対処](#Tackling-overfitting-via-regularization)
# - [サポートベクターマシンによる最大マージン分類](#Maximum-margin-classification-with-support-vector-machines)
#     - [最大マージンの直感](#Maximum-margin-intuition)
#     - [スラック変数を使った非線形分離不可能ケースへの対処](#Dealing-with-the-nonlinearly-separable-case-using-slack-variables)
#     - [scikit-learnでの代替実装](#Alternative-implementations-in-scikit-learn)
# - [カーネルSVMを使った非線形問題の解決](#Solving-nonlinear-problems-using-a-kernel-SVM)
#     - [カーネルトリックを使った高次元空間での分離超平面の発見](#Using-the-kernel-trick-to-find-separating-hyperplanes-in-higher-dimensional-space)
# - [決定木学習](#Decision-tree-learning)
#     - [情報利得の最大化 – 最大の効果を得る](#Maximizing-information-gain-–-getting-the-most-bang-for-the-buck)
#     - [決定木の構築](#Building-a-decision-tree)
#     - [ランダムフォレストによる弱学習器から強学習器への結合](#Combining-weak-to-strong-learners-via-random-forests)
# - [k近傍法 – 怠惰学習アルゴリズム](#K-nearest-neighbors-–-a-lazy-learning-algorithm)
# - [まとめ](#Summary)







# # 分類アルゴリズムの選択

# ...

# # scikit-learnでの最初のステップ

# scikit-learnからIrisデータセットを読み込みます。ここで、3列目は花びらの長さ、4列目は花びらの幅を表します。クラスはすでに整数ラベルに変換されており、0=Iris-Setosa、1=Iris-Versicolor、2=Iris-Virginicaとなっています。




# Iris データセットをロード
iris = datasets.load_iris()
# 3,4列目の特徴量を抽出
X = iris.data[:, [2, 3]]
# クラスラベルを取得
y = iris.target

# 一意なクラスラベルを出力
print('Class labels:', np.unique(y))


# データを訓練70%、テスト30%に分割:




# データセットを70%トレーニングセットに30%をテストセットに分割
# train_test_split 関数はランダムにデータを分割する
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)




print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))


# 特徴量の標準化:




sc = StandardScaler()
# 訓練データの平均と標準偏差を計算
sc.fit(X_train)
# 平均と標準偏差を用いてデータを標準化
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)



# ## scikit-learnでのパーセプトロンの訓練




# 学習率を0.1に設定し、パーセプトロンのインスタンスを作成
ppn = Perceptron(eta0=0.1, random_state=1)
# 訓練データを用いてパーセプトロンを学習
ppn.fit(X_train_std, y_train)




# テストデータで予測を行う
y_pred = ppn.predict(X_test_std)
# 誤分類したデータ点の数を出力
print('Misclassified examples: %d' % (y_test != y_pred).sum())




# 精度を計算(分類の正解率)
# y_test: 正解ラベル, y_pred: 予測ラベル
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))




# 精度を計算するための別の方法
# ppn.score メソッドを使用
# X_test_std: テストデータ, y_test: 正解ラベル
print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))





# matplotlib の最新バージョンとの互換性をチェック


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # マーカージェネレーターとカラーマップを設定
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 決定領域をプロット
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # grid pointsを生成
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # 各特徴量を１次元配列に変換し、分類器で予測
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # 予測結果を元のグリッドポイントのデータサイズに変換
    lab = lab.reshape(xx1.shape)
    # グリッドポイントの等高線をプロット
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    # 軸の範囲の設定
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # クラス例をプロット
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')

    # テストデータ点を目立たせる（点を丸で表示）
    if test_idx:
        # 全てのデータ点をプロット
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='none',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='Test set')


# 標準化された訓練データを使ってパーセプトロンモデルを訓練:



# 訓練データとテストデータの特徴量を行方向に結合
X_combined_std = np.vstack((X_train_std, X_test_std))
# 訓練データとテストデータのクラスラベルを結合
y_combined = np.hstack((y_train, y_test))

# 決定領域をプロット
plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))
# 軸ラベルを設定
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
# 凡例を設定(左上に配置)
plt.legend(loc='upper left')

# グラフを表示
plt.tight_layout()
#plt.savefig('figures/03_01.png', dpi=300)
plt.show()



# # ロジスティック回帰によるクラス確率のモデリング

# ...

# ### ロジスティック回帰の直感と条件付き確率




# シグモイド関数を定義
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# 0.1間隔で-7から7未満のデータを生成
z = np.arange(-7, 7, 0.1)
# 生成したデータでシグモイド関数を実行
sigma_z = sigmoid(z)

# 元のデータとシグモイド関数の出力をプロット
plt.plot(z, sigma_z)
# 垂直線を追加(z=0の位置に)
plt.axvline(0.0, color='k')
# y軸の上限と下限を設定
plt.ylim(-0.1, 1.1)
# 軸ラベルを設定
plt.xlabel('z')
plt.ylabel('$\sigma (z)$')

# y軸の目盛りとグリッド線
plt.yticks([0.0, 0.5, 1.0])
# Axesクラスへのobjectの取得
ax = plt.gca()
# y軸の目盛りに合わせて水平グリッド線を追加
ax.yaxis.grid(True)

# グラフを表示
plt.tight_layout()
#plt.savefig('figures/03_02.png', dpi=300)
plt.show()











# ### ロジスティック損失関数の重みの学習



# y=1の損失値を計算する関数
def loss_1(z):
    return - np.log(sigmoid(z))

# y=0の損失値を計算する関数
def loss_0(z):
    return - np.log(1 - sigmoid(z))

# 0.1間隔で-10から10未満のデータを生成
z = np.arange(-10, 10, 0.1)
# シグモイド関数を実行
sigma_z = sigmoid(z)

# y=1の損失値を計算する関数を実行
c1 = [loss_1(x) for x in z]
# 結果をプロット
plt.plot(sigma_z, c1, label='L(w, b) if y=1')

# y=0の損失値を計算する関数を実行
c0 = [loss_0(x) for x in z]
# 結果をプロット
plt.plot(sigma_z, c0, linestyle='--', label='L(w, b) if y=0')

# x軸とy軸の上限、下限を設定
plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
# 軸のラベルを設定
plt.xlabel('$\sigma(z)$')
plt.ylabel('L(w, b)')
# 凡例を設定
plt.legend(loc='best')
# グラフを表示
plt.tight_layout()
#plt.savefig('figures/03_04.png', dpi=300)
plt.show()




class LogisticRegressionGD:
    """勾配降下ベースのロジスティック回帰分類器。

    Parameters（パラメータ）
    ------------
    eta : float
      学習率（0.0から1.0の間）
    n_iter : int
      訓練データセットに対する繰り返し回数
    random_state : int
      重みの初期化のための乱数生成器のシード


    Attributes（属性）
    -----------
    w_ : 1d-array
      訓練後の重み
    b_ : Scalar（スカラー）
      学習後のバイアス項
    losses_ : list
       各エポックでの対数損失関数値

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        # 学習率の初期化、訓練回数の初期化、乱数生成器のシードを設定
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ 訓練データに適合させる

        Parameters（パラメータ）
        ----------
        X : {配列のような構造}, shape = [n_examples, n_features]
          訓練ベクトル、n_examplesはデータ点の数、n_featuresは特徴量の数
        y : 配列のようなデータ構造, shape = [n_examples]
          目標値

        Returns（戻り値）
        -------
        self : LogisticRegressionGDのインスタンス

        """
        rgen = np.random.RandomState(self.random_state) # 乱数生成器のインスタンスを作成
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1]) # 重みを正規分布から初期化
        self.b_ = np.float_(0.) # バイアス項を初期化
        self.losses_ = [] # 損失値を格納するリストを初期化

        # 訓練回数分まで訓練データを反復処理
        for i in range(self.n_iter):
            net_input = self.net_input(X) # 総入力を計算
            output = self.activation(net_input) # シグモイド関数を適用
            errors = (y - output) # 出力と目標値の差を計算
            self.w_ += self.eta * X.T.dot(errors) / X.shape[0] # 重みを更新
            self.b_ += self.eta * errors.mean() # バイアス項を更新
            loss = (-y.dot(np.log(output)) - (1 - y).dot(np.log(1 - output))) / X.shape[0] # 損失値を計算
            self.losses_.append(loss) # 損失値をリストに追加
        return self

    def net_input(self, X):
        """総入力を計算"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, z):
        """ロジスティックシグモイド活性化関数を計算"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """1ステップ後のクラスラベルを返す"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)





# Irisデータセットのクラスラベル0と1のサブセットを作成
X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

# ロジスティック回帰のインスタンスを作成
lrgd = LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)
# 訓練データを用いてロジスティック回帰を学習
lrgd.fit(X_train_01_subset,
         y_train_01_subset)

# 決定領域をプロット
plot_decision_regions(X=X_train_01_subset, 
                      y=y_train_01_subset,
                      classifier=lrgd)

# 軸ラベルを設定
plt.xlabel('Petal length [standardized]')
# y軸のラベルを設定
plt.ylabel('Petal width [standardized]')
# 凡例を設定
plt.legend(loc='upper left')

# グラフを表示
plt.tight_layout()
#plt.savefig('figures/03_05.png', dpi=300)
plt.show()


# ### scikit-learnでのロジスティック回帰モデルの訓練




# ロジスティック回帰のインスタンスを生成
lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')
# 訓練データをモデルに適合させる
lr.fit(X_train_std, y_train)

# 決定領域をプロット
plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]') # 軸ラベルを設定
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left') # 凡例を設定
plt.tight_layout() # グラフを表示
#plt.savefig('figures/03_06.png', dpi=300)
plt.show()




# テストセットの最初の3つのデータ点の確率を予測
lr.predict_proba(X_test_std[:3, :])




lr.predict_proba(X_test_std[:3, :]).sum(axis=1)




# 各行で最も大きい列を特定
lr.predict_proba(X_test_std[:3, :]).argmax(axis=1)




# predictメソッドを直接呼び出す方が便利
lr.predict(X_test_std[:3, :])




# 1行のデータを2次元配列に変換する方法、reshapeメソッドで新しい次元を追加
lr.predict(X_test_std[0, :].reshape(1, -1))



# ### 正則化による過学習への対処







weights, params = [], [] # 重み係数と逆正則化パラメータのリストを生成
# 10個の逆正則化パラメータに対応するロジスティック回帰モデルを処理
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10.**c,
                            multi_class='ovr')
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1]) # 重み係数を格納
    params.append(10.**c) # 逆正則化パラメータを格納

weights = np.array(weights) # 重み係数をNumPy配列に変換
# 横軸に逆正則化パラメータ、縦軸に重み係数をプロット
plt.plot(params, weights[:, 0],
         label='Petal length')
plt.plot(params, weights[:, 1], linestyle='--',
         label='Petal width')
plt.ylabel('Weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log') # x軸を対数スケールに設定
#plt.savefig('figures/03_08.png', dpi=300)
plt.show()



# # サポートベクターマシンによる最大マージン分類





# ## 最大マージンの直感

# ...

# ## スラック変数を使った非線形分離不可能ケースへの対処








svm = SVC(kernel='linear', C=1.0, random_state=1) # 線形SVMのインスタンスを生成
svm.fit(X_train_std, y_train) # 線形SVMモデルを訓練

# 決定領域をプロット
plot_decision_regions(X_combined_std, 
                      y_combined,
                      classifier=svm, 
                      test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('figures/03_11.png', dpi=300)
plt.show()


# ## scikit-learnでの代替実装




ppn = SGDClassifier(loss='perceptron') # SGDバージョンのパーセプトロン
lr = SGDClassifier(loss='log') # SGDバージョンのロジスティック回帰
svm = SGDClassifier(loss='hinge') # SGDバージョンのSVM（損失関数=ヒンジ損失）



# # カーネルSVMを使った非線形問題の解決




np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, 0)

plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='royalblue',
            marker='s',
            label='Class 1')
plt.scatter(X_xor[y_xor == 0, 0],
            X_xor[y_xor == 0, 1],
            c='tomato',
            marker='o',
            label='Class 0')

plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.legend(loc='best')
plt.tight_layout()
#plt.savefig('figures/03_12.png', dpi=300)
plt.show()







# ## カーネルトリックを使った高次元空間での分離超平面の発見



svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor,
                      classifier=svm)

plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('figures/03_14.png', dpi=300)
plt.show()





svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('figures/03_15.png', dpi=300)
plt.show()




svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, 
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('figures/03_16.png', dpi=300)
plt.show()



# # 決定木学習







def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))

x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None 
       for p in x]

plt.ylabel('Entropy')
plt.xlabel('Class-membership probability p(i=1)')
plt.plot(x, ent)
#plt.savefig('figures/03_26.png', dpi=300)
plt.show()







# ## 情報利得の最大化 - 最大の効果を得る





def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))


def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))


def error(p):
    return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)

ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent]
err = [error(i) for i in x]

fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], 
                          ['Entropy', 'Entropy (scaled)', 
                           'Gini impurity', 'Misclassification error'],
                          ['-', '-', '--', '-.'],
                          ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=5, fancybox=True, shadow=False)

ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity index')
#plt.savefig('figures/03_19.png', dpi=300, bbox_inches='tight')
plt.show()



# ## 決定木の構築




tree_model = DecisionTreeClassifier(criterion='gini', 
                                    max_depth=4, 
                                    random_state=1)
tree_model.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, 
                      classifier=tree_model,
                      test_idx=range(105, 150))

plt.xlabel('Petal length [cm]')
plt.ylabel('Petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('figures/03_20.png', dpi=300)
plt.show()





feature_names = ['Sepal length', 'Sepal width',
                 'Petal length', 'Petal width']
tree.plot_tree(tree_model,
               feature_names=feature_names,
               filled=True)

#plt.savefig('figures/03_21_1.pdf')
plt.show()




# ## ランダムフォレストによる弱学習器から強学習器への結合




forest = RandomForestClassifier(n_estimators=25, 
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined, 
                      classifier=forest, test_idx=range(105, 150))

plt.xlabel('Petal length [cm]')
plt.ylabel('Petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('figures/03_2.png', dpi=300)
plt.show()



# # k近傍法 - 怠惰学習アルゴリズム








knn = KNeighborsClassifier(n_neighbors=5, 
                           p=2, 
                           metric='minkowski')
knn.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, 
                      classifier=knn, test_idx=range(105, 150))

plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('figures/03_24_figures.png', dpi=300)
plt.show()



# # まとめ

# ...

# ---
# 
# 読者は以下のセルを無視してください。









