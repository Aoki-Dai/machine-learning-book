# coding: utf-8


import sys
from python_environment_check import check_packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc
from numpy import interp
from sklearn.utils import resample

# # PyTorch と Scikit-Learn で学ぶ機械学習
# # -- コード例

# ## パッケージのバージョン確認

# check_packages.py スクリプトを読み込むためにフォルダをパスに追加します:



sys.path.insert(0, '..')


# 推奨パッケージのバージョンを確認します:





d = {
    'numpy': '1.21.2',
    'matplotlib': '3.4.3',
    'sklearn': '1.0',
    'pandas': '1.3.2'
}
check_packages(d)


# # 第6章 - モデル評価とハイパーパラメータチューニングのベストプラクティス


# ### 概要

# - [パイプラインでワークフローを合理化する](#Streamlining-workflows-with-pipelines)
#   - [Breast Cancer Wisconsin データセットの読み込み](#Loading-the-Breast-Cancer-Wisconsin-dataset)
#   - [トランスフォーマと推定器をパイプラインで結合する](#Combining-transformers-and-estimators-in-a-pipeline)
# - [k 分割交差検証でモデル性能を評価する](#Using-k-fold-cross-validation-to-assess-model-performance)
#   - [ホールドアウト法](#The-holdout-method)
#   - [k 分割交差検証](#K-fold-cross-validation)
# - [学習曲線と検証曲線でアルゴリズムをデバッグする](#Debugging-algorithms-with-learning-and-validation-curves)
#   - [学習曲線でバイアスとバリアンスを診断する](#Diagnosing-bias-and-variance-problems-with-learning-curves)
#   - [検証曲線で過学習と学習不足に対処する](#Addressing-overfitting-and-underfitting-with-validation-curves)
# - [グリッドサーチで機械学習モデルを微調整する](#Fine-tuning-machine-learning-models-via-grid-search)
#   - [グリッドサーチによるハイパーパラメータの調整](#Tuning-hyperparameters-via-grid-search)
#   - [ランダムサーチでハイパーパラメータ空間を広く探索する](#Exploring-hyperparameter-configurations-more-widely-with-randomized-search)
#   - [逐次ハルビングによるより資源効率的なハイパーパラメータ探索](#More-resource-efficient-hyperparameter-search-with-successive-halving)
#   - [ネスト化交差検証によるアルゴリズム選択](#Algorithm-selection-with-nested-cross-validation)
# - [さまざまな評価指標を見てみる](#Looking-at-different-performance-evaluation-metrics)
#   - [混同行列の読み方](#Reading-a-confusion-matrix)
#   - [分類モデルの適合率と再現率の最適化](#Optimizing-the-precision-and-recall-of-a-classification-model)
#   - [ROC 曲線のプロット](#Plotting-a-receiver-operating-characteristic)
#   - [多クラス分類のスコアリング指標](#The-scoring-metrics-for-multiclass-classification)
# - [クラス不均衡への対処](#Dealing-with-class-imbalance)
# - [まとめ](#Summary)






# # パイプラインでワークフローを合理化する <a id="Streamlining-workflows-with-pipelines"></a>

# ...

# ## Breast Cancer Wisconsin データセットの読み込み <a id="Loading-the-Breast-Cancer-Wisconsin-dataset"></a>




df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases'
                 '/breast-cancer-wisconsin/wdbc.data', header=None)

# if the Breast Cancer dataset is temporarily unavailable from the
# UCI machine learning repository, un-comment the following line
# of code to load the dataset from a local path:

# df = pd.read_csv('wdbc.data', header=None)

df.head()




df.shape






X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_




le.transform(['M', 'B'])





X_train, X_test, y_train, y_test = \
    train_test_split(X, y, 
                     test_size=0.20,
                     stratify=y,
                     random_state=1)



# ## トランスフォーマと推定器をパイプラインで結合する <a id="Combining-transformers-and-estimators-in-a-pipeline"></a>




pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression())

pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
test_acc = pipe_lr.score(X_test, y_test)
print(f'Test accuracy: {test_acc:.3f}')







# # k 分割交差検証でモデル性能を評価する <a id="Using-k-fold-cross-validation-to-assess-model-performance"></a>

# ...

# ## ホールドアウト法 <a id="The-holdout-method"></a>






# ## k 分割交差検証 <a id="K-fold-cross-validation"></a>







    

kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)

scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)

    print(f'Fold: {k+1:02d}, '
          f'Class distr.: {np.bincount(y_train[train])}, '
          f'Acc.: {score:.3f}')
    
mean_acc = np.mean(scores)
std_acc = np.std(scores)
print(f'\nCV accuracy: {mean_acc:.3f} +/- {std_acc:.3f}')





scores = cross_val_score(estimator=pipe_lr,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print(f'CV accuracy scores: {scores}')
print(f'CV accuracy: {np.mean(scores):.3f} '
      f'+/- {np.std(scores):.3f}')



# # 学習曲線でアルゴリズムをデバッグする <a id="Debugging-algorithms-with-learning-and-validation-curves"></a>


# ## 学習曲線でバイアスとバリアンスを診断する <a id="Diagnosing-bias-and-variance-problems-with-learning-curves"></a>









pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty='l2', max_iter=10000))

train_sizes, train_scores, test_scores =\
                learning_curve(estimator=pipe_lr,
                               X=X_train,
                               y=y_train,
                               train_sizes=np.linspace(0.1, 1.0, 10),
                               cv=10,
                               n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='Training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
plt.tight_layout()
# plt.savefig('figures/06_05.png', dpi=300)
plt.show()



# ## 検証曲線で過学習と学習不足に対処する <a id="Addressing-overfitting-and-underfitting-with-validation-curves"></a>





param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(
                estimator=pipe_lr, 
                X=X_train, 
                y=y_train, 
                param_name='logisticregression__C', 
                param_range=param_range,
                cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, 
         color='blue', marker='o', 
         markersize=5, label='Training accuracy')

plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

plt.plot(param_range, test_mean, 
         color='green', linestyle='--', 
         marker='s', markersize=5, 
         label='Validation accuracy')

plt.fill_between(param_range, 
                 test_mean + test_std,
                 test_mean - test_std, 
                 alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.tight_layout()
# plt.savefig('figures/06_06.png', dpi=300)
plt.show()



# # グリッドサーチで機械学習モデルを微調整する <a id="Fine-tuning-machine-learning-models-via-grid-search"></a>


# ## グリッドサーチによるハイパーパラメータの調整 <a id="Tuning-hyperparameters-via-grid-search"></a>




pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'svc__C': param_range, 
               'svc__kernel': ['linear']},
              {'svc__C': param_range, 
               'svc__gamma': param_range, 
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  refit=True,
                  cv=10)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)




clf = gs.best_estimator_

# clf.fit(X_train, y_train) 
# note that we do not need to refit the classifier
# because this is done automatically via refit=True.

print(f'Test accuracy: {clf.score(X_test, y_test):.3f}')






pipe_svc = make_pipeline(
    StandardScaler(),
    SVC(random_state=1))

param_grid = [{'svc__C': param_range,
               'svc__kernel': ['linear']},
              {'svc__C': param_range,
               'svc__gamma': param_range,
               'svc__kernel': ['rbf']}]


rs = RandomizedSearchCV(estimator=pipe_svc,
                        param_distributions=param_grid,
                        scoring='accuracy',
                        refit=True,
                        n_iter=20,
                        cv=10,
                        random_state=1,
                        n_jobs=-1)




rs = rs.fit(X_train, y_train)
print(rs.best_score_)




print(rs.best_params_)


# ## ランダムサーチでハイパーパラメータ空間を広く探索する <a id="Exploring-hyperparameter-configurations-more-widely-with-randomized-search"></a>









param_range = [0.0001, 0.001, 0.01, 0.1,
               1.0, 10.0, 100.0, 1000.0]

param_range = scipy.stats.loguniform(0.0001, 1000.0)

np.random.seed(1)
param_range.rvs(10)


# ## 逐次ハルビングによるより資源効率的なハイパーパラメータ探索 <a id="More-resource-efficient-hyperparameter-search-with-successive-halving"></a>







hs = HalvingRandomSearchCV(
    pipe_svc,
    param_distributions=param_grid,
    n_candidates='exhaust',
    resource='n_samples',
    factor=1.5,
    random_state=1,
    n_jobs=-1)




hs = hs.fit(X_train, y_train)
print(hs.best_score_)
print(hs.best_params_)




clf = hs.best_estimator_
print(f'Test accuracy: {hs.score(X_test, y_test):.3f}')



# ## ネスト化交差検証によるアルゴリズム選択 <a id="Algorithm-selection-with-nested-cross-validation"></a>







gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2)

scores = cross_val_score(gs, X_train, y_train, 
                         scoring='accuracy', cv=5)
print(f'CV accuracy: {np.mean(scores):.3f} '
      f'+/- {np.std(scores):.3f}')





gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                  scoring='accuracy',
                  cv=2)

scores = cross_val_score(gs, X_train, y_train, 
                         scoring='accuracy', cv=5)
print(f'CV accuracy: {np.mean(scores):.3f} '
      f'+/- {np.std(scores):.3f}')



# # さまざまな性能評価指標を見てみる <a id="Looking-at-different-performance-evaluation-metrics"></a>

# ...

# ## 混同行列の読み方 <a id="Reading-a-confusion-matrix"></a>








pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)




fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
ax.xaxis.set_ticks_position('bottom')

plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.tight_layout()
#plt.savefig('figures/06_09.png', dpi=300)
plt.show()


# ### 補足

# 先ほどクラスラベルをエンコードし、悪性（malignant）を「陽性」クラス（1）、良性（benign）を「陰性」クラス（0）に設定したことを思い出してください:



le.transform(['M', 'B'])




confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)


# 次に、混同行列を以下のように表示しました:



confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)


# 注意: クラス0（真のラベル0）を正しくクラス0と予測したもの（真陰性）は、行列の左上（インデックス 0,0）にあります。真陰性を右下（インデックス 1,1）、真陽性を左上にしたい場合は、以下のように `labels` 引数で順序を変更できます:



confmat = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[1, 0])
print(confmat)


# 結論:
# 
# > この例ではクラス1（悪性）を陽性クラスとすると、モデルはクラス0に属するサンプルを71件（真陰性）、クラス1に属するサンプルを40件（真陽性）正しく分類しました。一方で、クラス0のサンプルを1件クラス1と誤分類（偽陽性）し、悪性であるにもかかわらず2件を良性と予測しました（偽陰性）。


# ## 分類モデルの適合率と再現率の最適化 <a id="Optimizing-the-precision-and-recall-of-a-classification-model"></a>




pre_val = precision_score(y_true=y_test, y_pred=y_pred)
print(f'Precision: {pre_val:.3f}')

rec_val = recall_score(y_true=y_test, y_pred=y_pred)
print(f'Recall: {rec_val:.3f}')

f1_val = f1_score(y_true=y_test, y_pred=y_pred)
print(f'F1: {f1_val:.3f}')

mcc_val = matthews_corrcoef(y_true=y_test, y_pred=y_pred)
print(f'MCC: {mcc_val:.3f}')





scorer = make_scorer(f1_score, pos_label=0)

c_gamma_range = [0.01, 0.1, 1.0, 10.0]

param_grid = [{'svc__C': c_gamma_range,
               'svc__kernel': ['linear']},
              {'svc__C': c_gamma_range,
               'svc__gamma': c_gamma_range,
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring=scorer,
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)



# ## ROC 曲線のプロット <a id="Plotting-a-receiver-operating-characteristic"></a>





pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(penalty='l2', 
                                           random_state=1,
                                           solver='lbfgs',
                                           C=100.0))

X_train2 = X_train[:, [4, 14]]
    

cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train))

fig = plt.figure(figsize=(7, 5))

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train],
                         y_train[train]).predict_proba(X_train2[test])

    fpr, tpr, thresholds = roc_curve(y_train[test],
                                     probas[:, 1],
                                     pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr,
             tpr,
             label=f'ROC fold {i+1} (area = {roc_auc:.2f})')

plt.plot([0, 1],
         [0, 1],
         linestyle='--',
         color=(0.6, 0.6, 0.6),
         label='Random guessing (area = 0.5)')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label=f'Mean ROC (area = {mean_auc:.2f})', lw=2)
plt.plot([0, 0, 1],
         [0, 1, 1],
         linestyle=':',
         color='black',
         label='Perfect performance (area = 1.0)')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='lower right')

plt.tight_layout()
# plt.savefig('figures/06_10.png', dpi=300)
plt.show()



# ## 多クラス分類のスコアリング指標 <a id="The-scoring-metrics-for-multiclass-classification"></a>



pre_scorer = make_scorer(score_func=precision_score, 
                         pos_label=1, 
                         greater_is_better=True, 
                         average='micro')


# ## クラス不均衡への対処 <a id="Dealing-with-class-imbalance"></a>



X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
y_imb = np.hstack((y[y == 0], y[y == 1][:40]))




y_pred = np.zeros(y_imb.shape[0])
np.mean(y_pred == y_imb) * 100





print('Number of class 1 examples before:', X_imb[y_imb == 1].shape[0])

X_upsampled, y_upsampled = resample(X_imb[y_imb == 1],
                                    y_imb[y_imb == 1],
                                    replace=True,
                                    n_samples=X_imb[y_imb == 0].shape[0],
                                    random_state=123)

print('Number of class 1 examples after:', X_upsampled.shape[0])




X_bal = np.vstack((X[y == 0], X_upsampled))
y_bal = np.hstack((y[y == 0], y_upsampled))




y_pred = np.zeros(y_bal.shape[0])
np.mean(y_pred == y_bal) * 100



# # まとめ <a id="Summary"></a>

# ...

# ---
# 
# この次のセルは読者は無視して構いません。




