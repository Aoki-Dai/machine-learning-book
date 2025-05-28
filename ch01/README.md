Python機械学習 - コード例


##  第1章: コンピュータにデータから学習する能力を与える


---



## 既存のCondaユーザー向け: 新しいConda環境の作成（オプション）



すでにcondaユーザーの場合は、次のセクションをスキップして、このリポジトリのメインフォルダから


```
make -f Makefile
```

を実行することで、新しいconda環境（"pyml-book"という名前）で推奨されるすべてのパッケージバージョンを取得できます。



## Python環境の手動設定

この章にはコード例は含まれていませんが、次の章に進む前にPythonの設定と確認を行うことをお勧めします。

より詳細な設定手順については、第1章の***Python Package IndexからのPythonとパッケージのインストール***のセクションを参照してください。



**Conda**

condaを使用している場合（[Miniforge](https://github.com/conda-forge/miniforge)経由でのcondaインストールを推奨）、次のように新しい環境を作成できます:

```bash
conda create -n "pyml-book" python=3.9 numpy=1.21.2 scipy=1.7.0 scikit-learn=1.0 matplotlib=3.4.3 pandas=1.3.2
```

この環境を作成した後、次のようにアクティベートできます:

```bash
conda activate "pyml-book"
```



**PipとVirtualenv**

`pip`を使用することを好む場合は、次のように必要なパッケージをインストールできます:

```bash
pip install numpy==1.21.2 scipy==1.7.0 scikit-learn==1.0 matplotlib==3.4.3 pandas==1.3.2
```

ただし、この書籍用に新しい仮想環境を作成することを追加で推奨します。
[virtualenv](https://virtualenv.pypa.io/en/latest/)を使用して、特定のPythonバージョンで新しい仮想環境を作成できます:

```bash
pip install virtualenv
cd /path/to/where/you/want/your/environment
virtualenv pyml-book
source pyml-book/bin/activate 
```

環境をアクティベートした後、次のように必要なパッケージをインストールできます:

```bash
pip install numpy==1.21.2 scipy==1.7.0 scikit-learn==1.0 matplotlib==3.4.3 pandas==1.3.2
```







## Python環境の確認

以下の章に向けてPython環境が正しく設定されていることを確認するために、このリポジトリのメインフォルダにある[`../python_environment_check.py`](../python_environment_check.py)スクリプトを実行することをお勧めします。

`python_environment_check.py`スクリプトは次のように実行できます:

    python python_environment_check.py

以下は出力例です:

```python
(base) sebastian@MacBook-Air ~/Desktop/Python-Machine-Learning-PyTorch-Edition/ch01 % python ../python_environment_check.py
[OK] Your Python version is 3.9.6 | packaged by conda-forge | (default, Jul 11 2021, 03:35:11)
[Clang 11.1.0 ]
[OK] numpy 1.21.2
[OK] scipy 1.7.0
[OK] matplotlib 3.4.3
[OK] sklearn 1.0
[OK] pandas 1.3.2
```


## Jupyter Notebook

このリポジトリに含まれるコードファイルの.ipynb拡張子について疑問に思った読者もいるでしょう -- これらはJupyter notebook（以前はIPython notebookと呼ばれていました）です。

通常の.pyスクリプトと比較して、Jupyter notebookでは以下のすべてを一箇所にまとめることができます:

- コード
- コード実行結果
- データのプロット
- 数学記法の入力と表示に便利なMarkdownとLaTeX構文をサポートするドキュメント

最新のインストール手順については、https://jupyter.org/install ウェブサイトをご覧ください。

Jupyter notebookを開くことができる公式アプリケーションは2つあります: オリジナルのJupyter Notebookアプリと新しいJupyter Labアプリ（VS CodeもJupyter notebookサポートがあります）。このリポジトリで提供されているnotebookは両方と互換性があります。

Jupyter Labは次のようにインストールすることをお勧めします:

```bash
conda install -c conda-forge jupyterlab
```

または

```bash
pip install jupyterlab
```

最後に、このリポジトリで提供されているJupyter notebookはオプションですが、強く推奨することに注意してください。この書籍にあるすべてのコード例は.pyスクリプトファイルでも利用可能です（Jupyter notebookから変換されており、同一のコードが含まれていることを保証しています）。
