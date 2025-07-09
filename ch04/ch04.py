# coding: utf-8


import sys
from python_environment_check import check_packages
import pandas as pd
from io import StringIO
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# # PyTorchとScikit-Learnによる機械学習
# # -- コード例集

# ## パッケージバージョンチェック

# check_packages.pyスクリプトから読み込むためにフォルダをパスに追加：



sys.path.insert(0, '..')


# 推奨パッケージバージョンをチェック：





d = {
    'numpy': '1.21.2',
    'matplotlib': '3.4.3',
    'sklearn': '1.0',
    'pandas': '1.3.2'
}
check_packages(d)


# # 第4章 - 良い訓練データセットの構築 – データ前処理


# ### 概要

# - [欠損データの処理](#欠損データの処理)
#   - [表形式データでの欠損値の識別](#表形式データでの欠損値の識別)
#   - [欠損値を持つ訓練例や特徴量の除去](#欠損値を持つ訓練例や特徴量の除去)
#   - [欠損値の補完](#欠損値の補完)
#   - [scikit-learn推定器APIの理解](#scikit-learn推定器APIの理解)
# - [カテゴリカルデータの処理](#カテゴリカルデータの処理)
#   - [名義特徴量と順序特徴量](#名義特徴量と順序特徴量)
#   - [順序特徴量のマッピング](#順序特徴量のマッピング)
#   - [クラスラベルのエンコーディング](#クラスラベルのエンコーディング)
#   - [名義特徴量でのワンホットエンコーディング](#名義特徴量でのワンホットエンコーディング)
# - [データセットを別々の訓練セットとテストセットに分割](#データセットを別々の訓練セットとテストセットに分割)
# - [特徴量を同じスケールに合わせる](#特徴量を同じスケールに合わせる)
# - [意味のある特徴量の選択](#意味のある特徴量の選択)
#   - [モデル複雑さに対するペナルティとしてのL1・L2正則化](#モデル複雑さに対するペナルティとしてのL1・L2正則化)
#   - [L2正則化の幾何学的解釈](#L2正則化の幾何学的解釈)
#   - [L1正則化によるスパース解](#L1正則化によるスパース解)
#   - [逐次特徴選択アルゴリズム](#逐次特徴選択アルゴリズム)
# - [ランダムフォレストによる特徴量重要度の評価](#ランダムフォレストによる特徴量重要度の評価)
# - [まとめ](#まとめ)






# # 欠損データの処理

# ## 表形式データでの欠損値の識別



# サンプルデータを作成
csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

# Python 2.7を使用している場合は、
# 文字列をunicodeに変換する必要があります：

if (sys.version_info < (3, 0)):
    csv_data = unicode(csv_data)

# サンプルデータを読み込む
df = pd.read_csv(StringIO(csv_data))
df




df.isnull().sum()




# 基礎となるNumPy配列にアクセス
# `values`属性を使用
df.values



# ## 欠損値を持つ訓練例や特徴量の除去



# 欠損値を含む行を除去

df.dropna(axis=0)




# 引数を1にして欠損値を含む列を除去

df.dropna(axis=1)




# すべての列がNaNの行のみを除去
# この場合は値がNaNである行は無いため、配列全体が返される

df.dropna(how='all')




# 実際の値が4個未満の行を除去

df.dropna(thresh=4)




# 特定の列（ここでは'C'）でNaNが出現する行のみを除去

df.dropna(subset=['C'])



# ## 欠損値の補完



# 再度：元の配列
df.values




# 列の平均値による欠損値の補完

# 欠損値補完のインスタンスを作成（平均値補完）
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
# imrをデータに適合させる
imr = imr.fit(df.values)
# 補完を実行
imputed_data = imr.transform(df.values)
imputed_data





# pandasのfillna()メソッドを使用して、列の平均値で欠損値を補完することもできます
df.fillna(df.mean())


# ## scikit-learn推定器APIの理解










# # カテゴリカルデータの処理

# ## 名義特徴量と順序特徴量



# サンプルデータを生成（Tシャツの色、サイズ、価格、クラスラベル）
df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', 13.5, 'class1'],
                   ['blue', 'XL', 15.3, 'class2']])
# 列名を設定
df.columns = ['color', 'size', 'price', 'classlabel']
df



# ## 順序特徴量のマッピング



# Tシャツのサイズと整数に対応させるディクショナリを作成
size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}
# ディクショナリを使用してサイズを整数に変換
df['size'] = df['size'].map(size_mapping)
df




# 整数値を元のサイズに戻すためのディクショナリを作成
inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'].map(inv_size_mapping)



# ## クラスラベルのエンコーディング




# クラスラベルを文字列から整数に変換するためのマッピング辞書を作成
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
class_mapping




# クラスラベルを文字列から整数に変換
df['classlabel'] = df['classlabel'].map(class_mapping)
df




# 整数とクラスラベルを対応させるディクショナリを作成
inv_class_mapping = {v: k for k, v in class_mapping.items()}
# 整数から元のクラスラベルに戻す
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
df





# scikit-learnのLabelEncoderによるラベルエンコーディング
class_le = LabelEncoder()
# クラスラベルから整数に変換
y = class_le.fit_transform(df['classlabel'].values)
y




# inverse_transformを使用して整数から元のクラスラベルに戻す
class_le.inverse_transform(y)



# ## 名義特徴量でのワンホットエンコーディング



X = df[['color', 'size', 'price']].values # Tシャツの色、サイズ、価格の列を取得
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
X
# blue→0, green→1, red→2




# ダミー特徴量


X = df[['color', 'size', 'price']].values
# one-hotエンコーダを生成
color_ohe = OneHotEncoder()
# one-hotエンコーティングを実行
color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()





X = df[['color', 'size', 'price']].values
# 列0をone-hotエンコードし、列1と列2はそのまま
c_transf = ColumnTransformer([ ('onehot', OneHotEncoder(), [0]),
                               ('nothing', 'passthrough', [1, 2])])
# エンコードされたデータをfloat型に変換
c_transf.fit_transform(X).astype(float)




# pandasによるワンホットエンコーディング
# 文字列値を持つ列だけを変換

pd.get_dummies(df[['price', 'color', 'size']])




# get_dummiesでの多重共線性対策

pd.get_dummies(df[['price', 'color', 'size']], drop_first=True)




# OneHotEncoderでの多重共線性対策

color_ohe = OneHotEncoder(categories='auto', drop='first')
c_transf = ColumnTransformer([ ('onehot', color_ohe, [0]),
                               ('nothing', 'passthrough', [1, 2])])
c_transf.fit_transform(X).astype(float)



# ## オプション：順序特徴量のエンコーディング

# 順序特徴量のカテゴリ間の数値的な違いが不明な場合、または2つの順序値間の差が定義されていない場合、0/1値を使用した閾値エンコーディングでエンコードすることもできます。例えば、M、L、XLの値を持つ「size」特徴量を「x > M」と「x > L」の2つの新しい特徴量に分割することができます。元のDataFrameを考えてみましょう：



df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', 13.5, 'class1'],
                   ['blue', 'XL', 15.3, 'class2']])

df.columns = ['color', 'size', 'price', 'classlabel']
df


# pandasのDataFrameの`apply`メソッドを使用して、値閾値アプローチを使用してこれらの変数をエンコードするために、カスタムラムダ式を記述できます：



df['x > M'] = df['size'].apply(lambda x: 1 if x in {'L', 'XL'} else 0)
df['x > L'] = df['size'].apply(lambda x: 1 if x == 'XL' else 0)

del df['size']
df



# # データセットを別々の訓練セットとテストセットに分割



df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)

# UCI機械学習リポジトリからWineデータセットが一時的に利用できない場合は、
# 以下のコード行のコメントを外してローカルパスからデータセットを読み込んでください：

# df_wine = pd.read_csv('wine.data', header=None)


df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

print('Class labels', np.unique(df_wine['Class label']))
df_wine.head()





X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test =\
    train_test_split(X, y, 
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)



# # 特徴量を同じスケールに合わせる




mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)





stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


# 視覚的な例：



ex = np.array([0, 1, 2, 3, 4, 5])

print('standardized:', (ex - ex.mean()) / ex.std())

# pandasはデフォルトでddof=1（標本標準偏差）を使用することに注意してください。
# 一方、NumPyのstdメソッドとStandardScalerはddof=0（母集団標準偏差）を使用します。

# 正規化
print('normalized:', (ex - ex.min()) / (ex.max() - ex.min()))



# # 意味のある特徴量の選択

# ...

# ## モデル複雑さに対するペナルティとしてのL1・L2正則化

# ## L2正則化の幾何学的解釈









# ## L1正則化によるスパース解





# L1正則化をサポートするscikit-learnの正則化モデルでは、`penalty`パラメータを`'l1'`に設定するだけで、スパース解を得ることができます：




LogisticRegression(penalty='l1')


# 標準化されたWineデータに適用...




lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')
# C=1.0がデフォルトであることに注意してください。この値を増加または減少させることで、
# それぞれ正則化効果を弱くしたり強くしたりできます。
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))




lr.intercept_




np.set_printoptions(8)




lr.coef_[lr.coef_!=0].shape




lr.coef_





fig = plt.figure()
ax = plt.subplot(111)
    
colors = ['blue', 'green', 'red', 'cyan', 
          'magenta', 'yellow', 'black', 
          'pink', 'lightgreen', 'lightblue', 
          'gray', 'indigo', 'orange']

weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c, solver='liblinear', 
                            multi_class='ovr', random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1],
             color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('Weight coefficient')
plt.xlabel('C (inverse regularization strength)')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', 
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)

#plt.savefig('figures/04_08.png', dpi=300, 
#            bbox_inches='tight', pad_inches=0.2)

plt.show()



# ## 逐次特徴選択アルゴリズム





class SBS:
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, 
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, 
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score





knn = KNeighborsClassifier(n_neighbors=5)

# 特徴量の選択
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# 特徴量サブセットのパフォーマンスをプロット
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
# plt.savefig('figures/04_09.png', dpi=300)
plt.show()




k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])




knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))




knn.fit(X_train_std[:, k3], y_train)
print('Training accuracy:', knn.score(X_train_std[:, k3], y_train))
print('Test accuracy:', knn.score(X_test_std[:, k3], y_test))



# # ランダムフォレストによる特徴量重要度の評価




feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature importance')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(X_train.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
# plt.savefig('figures/04_10.png', dpi=300)
plt.show()





sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
print('Number of features that meet this threshold criterion:', 
      X_selected.shape[1])


# ここで、先ほど設定した特徴選択の閾値基準を満たした3つの特徴量を出力してみましょう（注意：このコードスニペットは実際の書籍には載っていませんが、説明目的で後からこのノートブックに追加されました）：



for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))



# # まとめ

# ...

# ---
# 
# 読者は次のセルを無視してください。




