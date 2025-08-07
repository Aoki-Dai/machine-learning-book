# coding: utf-8


import sys
from python_environment_check import check_packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects

# # PyTorchとScikit-Learnによる機械学習  
# # -- コード例

# ## パッケージのバージョンチェック

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


# # 第5章 - 次元削減によるデータ圧縮


# ### 概要

# - [主成分分析による教師なし次元削減](#Unsupervised-dimensionality-reduction-via-principal-component-analysis)
#   - [主成分分析の背後にある主要なステップ](#The-main-steps-behind-principal-component-analysis)
#   - [主成分をステップバイステップで抽出](#Extracting-the-principal-components-step-by-step)
#   - [全分散と寄与分散](#Total-and-explained-variance)
#   - [特徴変換](#Feature-transformation)
#   - [scikit-learnでの主成分分析](#Principal-component-analysis-in-scikit-learn)
#   - [特徴の寄与の評価](#Assessing-feature-contributions)
# - [線形判別分析による教師ありデータ圧縮](#Supervised-data-compression-via-linear-discriminant-analysis)
#   - [主成分分析と線形判別分析の比較](#Principal-component-analysis-versus-linear-discriminant-analysis)
#   - [線形判別分析の内部動作](#The-inner-workings-of-linear-discriminant-analysis)
#   - [散布行列の計算](#Computing-the-scatter-matrices)
#   - [新しい特徴部分空間のための線形判別の選択](#Selecting-linear-discriminants-for-the-new-feature-subspace)
#   - [新しい特徴空間への例の投影](#Projecting-examples-onto-the-new-feature-space)
#   - [scikit-learnによるLDA](#LDA-via-scikit-learn)
# - [非線形次元削減技術](#Nonlinear-dimensionality-reduction-techniques)
#   - [t-分散確率的近傍埋め込みによるデータ可視化](#Visualizing-data-via-t-distributed-stochastic-neighbor-embedding)
# - [まとめ](#Summary)






# # 主成分分析による教師なし次元削減

# ## 主成分分析の背後にある主要なステップ





# ## 主成分をステップバイステップで抽出




df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

# Wineデータセットが一時的にUCI機械学習リポジトリから
# 利用できない場合は、以下のコード行のコメントを外して
# ローカルパスからデータセットを読み込んでください:

# df_wine = pd.read_csv('wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

df_wine.head()



# データを70%の訓練用と30%のテスト用サブセットに分割。



# 2列目のデータをXに、1列目のデータをyに格納
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# 訓練データ70%とテストデータ30%に分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, 
    stratify=y,
    random_state=0
)


# データの標準化。



# 特徴量の標準化
# 平均0、分散1に変換
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# ---
# 
# **注意**
# 
# 誤って`X_test_std = sc.fit_transform(X_test)`を`X_test_std = sc.transform(X_test)`の代わりに書いてしまいました。この場合、テストセットの平均と標準偏差は訓練セットと（かなり）似ているはずなので、大きな違いはないでしょう。しかし、第3章で覚えているように、何らかの変換を行う場合の正しい方法は、訓練セットからのパラメータを再利用することです -- テストセットは基本的に「新しい、未知の」データを表すべきです。
# 
# 私の最初のタイポは、一部の人が*これらのパラメータをモデル訓練/構築から再利用せず*、新しいデータを「ゼロから」標準化するという一般的な間違いを反映しています。これがなぜ問題なのかを説明する簡単な例を示します。
# 
# 1つの特徴（これを「長さ」と呼びましょう）を持つ3つの例からなる簡単な訓練セットがあると仮定しましょう：
# 
# - train_1: 10 cm -> class_2
# - train_2: 20 cm -> class_2
# - train_3: 30 cm -> class_1
# 
# 平均: 20, 標準偏差: 8.2
# 
# 標準化後、変換された特徴値は次のようになります：
# 
# - train_std_1: -1.21 -> class_2
# - train_std_2: 0 -> class_2
# - train_std_3: 1.21 -> class_1
# 
# 次に、モデルが標準化された長さの値 < 0.6をclass_2として分類することを学習したと仮定しましょう（そうでなければclass_1）。ここまでは問題ありません。今度は、分類したい3つのラベルなしデータポイントがあるとしましょう：
# 
# - new_4: 5 cm -> class ?
# - new_5: 6 cm -> class ?
# - new_6: 7 cm -> class ?
# 
# 訓練データセットの「標準化されていない長さ」の値を見ると、これらの例はすべてclass_2に属する可能性が高いと直感的に言えます。しかし、標準偏差と平均を再計算して標準化すると、訓練セットで以前と同様の値が得られ、分類器は（おそらく誤って）例4と5をclass_2として分類するでしょう。
# 
# - new_std_4: -1.21 -> class_2
# - new_std_5: 0 -> class_2
# - new_std_6: 1.21 -> class_1
# 
# しかし、「訓練セット標準化」からのパラメータを使用すると、次の値が得られます：
# 
# - example5: -18.37 -> class_2
# - example6: -17.15 -> class_2
# - example7: -15.92 -> class_2
# 
# 値5 cm、6 cm、7 cmは、以前に訓練セットで見たものよりもはるかに低いものです。したがって、「新しい例」の標準化された特徴が訓練セットのすべての標準化された特徴よりもはるかに低いことは理にかなっています。
# 
# ---

# 共分散行列の固有値分解。



# 共分散行列を作成
cov_mat = np.cov(X_train_std.T)
# 共分散行列の固有値と固有ベクトルを計算
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print('\nEigenvalues \n', eigen_vals)


# **注意**: 
# 
# 上記では、対称共分散行列をその固有値と固有ベクトルに分解するために[`numpy.linalg.eig`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html)関数を使用しました。
#     <pre>>>> eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)</pre>
#     これは本当に「間違い」ではありませんが、おそらく最適ではありません。このような場合には[`numpy.linalg.eigh`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigh.html)を使用する方が良いでしょう。これは[エルミート行列](https://en.wikipedia.org/wiki/Hermitian_matrix)用に設計されています。後者は常に実固有値を返しますが、数値的に安定性の低い`np.linalg.eig`は非対称正方行列を分解できるため、特定の場合には複素固有値を返すことがあります。(S.R.)


# ## 全分散と寄与分散



# 固有値を合計
tot = sum(eigen_vals)
# 分散説明率を計算
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
# 分散説明率の累積和を計算
cum_var_exp = np.cumsum(var_exp)





# 分散説明率の棒グラフを作成
plt.bar(range(1, 14), var_exp, align='center',
        label='Individual explained variance')

# 分散説明率の累積和の階段グラフを作成
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='Cumulative explained variance')

plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('figures/05_02.png', dpi=300)
plt.show()



# ## 特徴変換



# (固有値, 固有ベクトル)のタプルのリストを作成
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# (固有値, 固有ベクトル)のタプルを大きいものから小さいものへソート
eigen_pairs.sort(key=lambda k: k[0], reverse=True)




w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)


# **注意**
# 使用しているNumPyとLAPACKのバージョンによって、行列Wの符号が反転して得られる場合があります。これは問題ではないことに注意してください：$v$が行列$\Sigma$の固有ベクトルの場合、
# 
# $$\Sigma v = \lambda v,$$
# 
# ここで$\lambda$は固有値です。
# 
# 
# すると$-v$も同じ固有値を持つ固有ベクトルになります：
# $$\Sigma \cdot (-v) = -\Sigma v = -\lambda v = \lambda \cdot (-v).$$



X_train_std[0].dot(w)




X_train_pca = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['o', 's', '^']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=f'Class {l}', marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('figures/05_03.png', dpi=300)
plt.show()



# ## scikit-learnでの主成分分析

# **注意**
# 
# 以下の4つのコードセルは、本の内容に加えて追加されたもので、scikit-learnで独自のPCA実装からの結果を再現する方法を説明するためのものです：




pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_




plt.bar(range(1, 14), pca.explained_variance_ratio_, align='center')
plt.step(range(1, 14), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')

plt.show()




pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)




plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()





def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # マーカー生成器とカラーマップの設定
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 決定境界面をプロット
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # クラスの例をプロット
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')


# 最初の2つの主成分を使用してロジスティック回帰分類器を訓練。




pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
lr = lr.fit(X_train_pca, y_train)




plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('figures/05_04.png', dpi=300)
plt.show()




plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('figures/05_05.png', dpi=300)
plt.show()




pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_


# ## 特徴の寄与の評価



loadings = eigen_vecs * np.sqrt(eigen_vals)

fig, ax = plt.subplots()

ax.bar(range(13), loadings[:, 0], align='center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(13))
ax.set_xticklabels(df_wine.columns[1:], rotation=90)

plt.ylim([-1, 1])
plt.tight_layout()
plt.savefig('figures/05_05_02.png', dpi=300)
plt.show()




loadings[:, 0]




sklearn_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

fig, ax = plt.subplots()

ax.bar(range(13), sklearn_loadings[:, 0], align='center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(13))
ax.set_xticklabels(df_wine.columns[1:], rotation=90)

plt.ylim([-1, 1])
plt.tight_layout()
plt.savefig('figures/05_05_03.png', dpi=300)
plt.show()



# # 線形判別分析による教師ありデータ圧縮

# ## 主成分分析と線形判別分析の比較





# ## 線形判別分析の内部動作


# ## 散布行列の計算

# 各クラスの平均ベクトルを計算:



np.set_printoptions(precision=4)

mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print(f'MV {label}: {mean_vecs[label - 1]}\n')


# クラス内散布行列を計算:



d = 13 # 特徴数
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))  # 各クラスの散布行列
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)  # 列ベクトルに変換
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter                          # クラス散布行列の和

print('Within-class scatter matrix: '
      f'{S_W.shape[0]}x{S_W.shape[1]}')


# より良い方法: クラスが等しく分布していないため共分散行列を使用:



print('Class label distribution:',  
      np.bincount(y_train)[1:])




d = 13  # 特徴数
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
    
print('Scaled within-class scatter matrix: '
      f'{S_W.shape[0]}x{S_W.shape[1]}')


# クラス間散布行列を計算:



mean_overall = np.mean(X_train_std, axis=0)
mean_overall = mean_overall.reshape(d, 1)  # 列ベクトルに変換

d = 13  # 特徴数
S_B = np.zeros((d, d))

for i, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)  # 列ベクトルに変換
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

print('Between-class scatter matrix: '
      f'{S_B.shape[0]}x{S_B.shape[1]}')



# ## 新しい特徴部分空間のための線形判別の選択

# 行列$S_W^{-1}S_B$の一般化固有値問題を解く:



eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))


# **注意**:
#     
# 上記では、対称共分散行列をその固有値と固有ベクトルに分解するために[`numpy.linalg.eig`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html)関数を使用しました。
#     <pre>>>> eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)</pre>
#     これは本当に「間違い」ではありませんが、おそらく最適ではありません。このような場合には[`numpy.linalg.eigh`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigh.html)を使用する方が良いでしょう。これは[エルミート行列](https://en.wikipedia.org/wiki/Hermitian_matrix)用に設計されています。後者は常に実固有値を返しますが、数値的に安定性の低い`np.linalg.eig`は非対称正方行列を分解できるため、特定の場合には複素固有値を返すことがあります。(S.R.)

# 固有値の降順で固有ベクトルをソート:



# (固有値, 固有ベクトル)のタプルのリストを作成
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# (固有値, 固有ベクトル)のタプルを大きいものから小さいものへソート
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# リストが固有値の降順で正しくソートされていることを視覚的に確認

print('Eigenvalues in descending order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])




tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)

plt.bar(range(1, 14), discr, align='center',
        label='Individual discriminability')
plt.step(range(1, 14), cum_discr, where='mid',
         label='Cumulative discriminability')
plt.ylabel('Discriminability ratio')
plt.xlabel('Linear discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
#plt.savefig('figures/05_07.png', dpi=300)
plt.show()




w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
              eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)


# ## 新しい特徴空間への例の投影

# ## Projecting examples onto the new feature space



X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['o', 's', '^']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train == l, 0],
                X_train_lda[y_train == l, 1] * (-1),
                c=c, label=f'Class {l}', marker=m)

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('figures/05_08.png', dpi=300)
plt.show()



# ## scikit-learnによるLDA




lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)





lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
lr = lr.fit(X_train_lda, y_train)

plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('figures/05_09.png', dpi=300)
plt.show()




X_test_lda = lda.transform(X_test_std)

plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('figures/05_10.png', dpi=300)
plt.show()



# # 非線形次元削減技術





# ### t-分散確率的近傍埋め込みによるデータ可視化




digits = load_digits()

fig, ax = plt.subplots(1, 4)

for i in range(4):
    ax[i].imshow(digits.images[i], cmap='Greys')
    
# plt.savefig('figures/05_12.png', dpi=300)
plt.show() 




digits.data.shape




y_digits = digits.target
X_digits = digits.data






tsne = TSNE(n_components=2,
            init='pca',
            random_state=123)
X_digits_tsne = tsne.fit_transform(X_digits)






def plot_projection(x, colors):
    
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    for i in range(10):
        plt.scatter(x[colors == i, 0],
                    x[colors == i, 1])

    for i in range(10):

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        
plot_projection(X_digits_tsne, y_digits)
# plt.savefig('figures/05_13.png', dpi=300)
plt.show()



# # まとめ

# ...

# ---
# 
# 読者は次のセルを無視してください。




