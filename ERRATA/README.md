# 現在の正誤表

&nbsp;
## 第1章

正誤表ではありませんが、改善提案です。現在、「記法の規則」セクションに以下の段落があります：


ベクトルを表すために小文字の太字文字を使用し（$\mathbf{x} \in \mathbb{R}^{n \times 1}$）、行列を表すために大文字の太字文字を使用します（$\mathbf{X} \in \mathbb{R}^{n \timeこのページの下部で、以下のように記載されています：

> そしてモデルは訓練データセットで100パーセントの精度に達します。検証データセットの精度は95パーセントで、モデルが若干過学習していることを示しています。

しかし、値は訓練精度90%、検証精度85%であるべきです。

**432ページ**

`import sklearn.model_selection` の行は冗長で削除できます。ルまたは行列の単一要素を参照するときは、文字をイタリック体で書きます（それぞれ $x^{(n)}$ または $x_m^{(n)}$）。


これは以下のように改善できます：


ベクトルを表すために小文字の太字文字を使用し（列ベクトルの場合は $\mathbf{x} \in \mathbb{R}^{n \times 1}$、行ベクトルの場合は $\mathbf{x} \in \mathbb{R}^{1 \times m}$）、行列を表すために大文字の太字文字を使用します（$\mathbf{X} \in \mathbb{R}^{n \times m}$）。ベクトルまたは行列の単一要素を参照するときは、文字をイタリック体で書きます（それぞれ $x^{(n)}$ または $x_m^{(n)}$）。

また、同じボックス内で、

$$
\mathbf{X}^{(i)} = \left[ x_1^{(i)} \, x_2^{(i)} \, x_3^{(i)} \, x_4^{(i)} \right]
$$

は以下のように変更できます：

$$
\mathbf{x}^{(i)} = \left[ x_1^{(i)} \, x_2^{(i)} \, x_3^{(i)} \, x_4^{(i)} \right], \quad 1 \leq i \leq n.
$$

そして

$$
\mathbf{x}_j = \left[ \begin{array}{c}
x_j^{(1)} \\
x_j^{(2)} \\
\vdots \\
x_j^{(150)}
\end{array} \right]
$$

は以下のように変更できます：

$$
\mathbf{x}_j = \left[ \begin{array}{c}
x_j^{(1)} \\
x_j^{(2)} \\
\vdots \\
x_j^{(150)}
\end{array} \right], \quad i \leq j \leq m.
$$


&nbsp;
## 第3章

**66ページ**

小さなスタイル上の問題：現在、下部の和記号は

```math
\sum_{i=1}
```

となっていますが、以下のいずれかであるべきです：

 ```math
 \sum_{i}
 ```

または

```math
\sum_{i=1}^{n}
```

**69ページ**

LogisticRegressionGD分類器のdocstringで「平均二乗誤差損失」と記載されていますが、これはコピー・ペーストエラーで、「対数損失」であるべきです。

**87ページ**

gammaの値が大きいほど、半径は小さく（大きくではなく）なります。

&nbsp;
## 第4章

コードブロックで、次のように記載されています：

```python
# Note that C=1.0 is the default. You can increase
# or decrease it to make the regulariztion effect
# stronger or weaker, respectively.
```

しかし、「stronger or weaker」ではなく「weaker or stronger」であるべきです

&nbsp;
## 第6章

**185ページ**

図6.6のキャプションで、現在「...SVM hyperparameter C」となっていますが、「...logistic regression hyperparameter C」であるべきです。

&nbsp;
## 第10章

**313ページ**

「...歪みが最も急速に増加し始めるkの値を特定する...」という文で、図を左から右に読むため（右から左ではなく）、*減少* がより正確な用語です。

&nbsp;
## 第11章

**341ページ**

ネット入力にバイアス項を追加。

**354ページ**

MSEは `mse = mse/i` で正規化されていますが、代わりに `mse = mse/(i+1)` で正規化されるべきです。（ただし、これは以下に示される結果には影響しません。MSEは依然として0.3です。）

**366ページ**

以下のように記載されています：

```math
\frac{\partial L}{\partial w_{1,1}^{(\text {out })}} = ...
```

しかし、以下であるべきです：

 ```math
\frac{\partial L}{\partial w_{1,1}^{(\text {h })}}
 ```

上の図と下のテキストに合わせるため。

**361ページ**

```math
\frac{\partial}{\partial w_{j, l}^{(l)}}=L(\boldsymbol{W}, \boldsymbol{b}) 
```

は以下であるべきです：

```math
\frac{\partial L}{\partial w_{j, l}^{(l)}}
```


&nbsp;
## 第12章

**376ページ**

テキストで以下のように記載されています：

> このため、PyTorchは便利な torch.chunk() 関数を提供しており、入力テンソルを等しいサイズのテンソルのリストに分割します。[...] テンソルサイズがchunksの値で割り切れない場合、最後のチャンクは小さくなります。

しかし、これは必ずしも正しくありません。ディスカッション [#203](https://github.com/rasbt/machine-learning-book/discussions/203) で提案されているように、より良い表現は以下の通りです：

> テンソルサイズがchunksの値で割り切れない場合、結果として得られるチャンクの数は意図した数より少なくなる可能性があり、および/または最後のチャンクが他のチャンクより小さくなる可能性があります。


**380ページ**

カスタムの `JointDataset` を定義したにも関わらず、`TensorDataset` を使用しています

**393ページ**

`y_pred = model(X_test_norm).detach().numpy()` の行は、単に `y_pred = model(X_test_norm)` に変更すべきです。二重にdetachすることを避けるためで、これはPyTorch 2.xでエラーが発生することが知られています。


**431ページ**

Torchmetrics 0.8.0以降を使用する場合、以下の行：

```python
self.train_acc = Accuracy()
self.valid_acc = Accuracy()
self.test_acc = Accuracy()
```

は以下のように変更する必要があります：

```python
self.train_acc = Accuracy(task="multiclass", num_classes=10)
self.valid_acc = Accuracy(task="multiclass", num_classes=10)
self.test_acc = Accuracy(task="multiclass", num_classes=10)
```


&nbsp;
## 第13章


**423ページ**

At the bottom of this page, it says

> and the
model reaches 100 percent accuracy on the training dataset. The validation dataset’s accuracy is 95 percent, which indicates that the model is slightly overfitting.

But the values should be 90% training accuracy and 85% validation accuracy.

**Page 432**

The line `import sklearn.model_selection` is redundant and can be removed.

&nbsp;
## 第14章


**459ページ**

459ページの `conv1d()` と `conv2d()` 関数は、[@JaGeo](https://github.com/JaGeo) による親切な[プルリクエスト](https://github.com/rasbt/machine-learning-book/pull/168)によって改善され、(1,1)以外のストライドのケースを処理できるようになりました。


**489ページ**

エラーではありませんが、可読性のために、以下を変更すると良いでしょう： 

```python
for j in range(num_epochs):
    img_batch, label_batch = next(iter(data_loader))
```

を

```python
for img_batch, label_batch in data_loader:
```

に。

&nbsp;
## 第15章

**505ページ**

方程式は技術的には正しいですが、文字oの代わりに文字0（ゼロ）が使われているように見えます：

```math
\mathbf{o}^{\left( t \right)} = \sigma_{0}\left( \mathbf{W}_{ho}\mathbf{h}^{\left( t \right)}+\mathbf{b}_{0} \right)
```

は以下であるべきです：

```math
\mathbf{o}^{\left( t \right)} = \sigma_{o}\left( \mathbf{W}_{ho}\mathbf{h}^{\left( t \right)}+\mathbf{b}_{o} \right)
```

**530ページ**

`from torch.utils.data import Dataset` の行が2回出現しています。

&nbsp;
## 第16章

**547ページ**

エラーではありませんが、`attention_weights.sum(dim=1)` を介して列の合計を求めている箇所で、この行列は対称であり、行の合計を求めても同じ結果が得られることを言及できます。

---

&nbsp;
## 第17章

**626ページ**

KL情報ボックスに間違いがあるようです。マイナス記号を削除するか、P(x)/Q(x) を Q(x)/P(x) に変更するべきです。さらに、log記号が欠けているようです。正しい公式は以下の通りです：

```math
KL(P \| Q) = -\sum_{i} P(x_i) \log \frac{Q(x_i)}{P(x_i)}
```

または

```math
KL(P \| Q) = \sum_{i} P(x_i) \log \frac{P(x_i)}{Q(x_i)}
```


2022年11月16日以前に印刷された書籍については、[旧正誤表](old-errata)を参照してください。



