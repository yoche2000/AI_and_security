# scikit-learn 套件

http://scikit-learn.org/stable/datasets/

## 應用領域：
```
監督式學習（Supervised learning）
  分類（Classification）
  迴歸（Regression）
非監督式學習（Unsupervised learning）
  分群（Clustering）
降維（Dimensionality reduction）
模型選擇（Model selection）
預處理（Preprocessing）
```
## 學習主題：
```
[第 21 天] 機器學習 玩具資料與線性迴歸
[第 22 天] 機器學習（2）複迴歸與 Logistic 迴歸
[第 23 天] 機器學習（3）決策樹與 k-NN 分類器
[第 24 天] 機器學習（4）分群演算法
[第 25 天] 機器學習（5）整體學習
[第 26 天] 機器學習（6）隨機森林與支持向量機
```
##  讀入資料

### 使用 sklearn 的 datasets 物件的 load_iris() 方法來讀入鳶尾花資料
```
from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
print(type(iris.data)) # 資料是儲存為 ndarray
print(iris.feature_names) # 變數名稱可以利用 feature_names 屬性取得

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names) # 轉換為 data frame
iris_df.ix[:, "species"] = iris.target # 將品種加入 data frame
iris_df.head() # 觀察前五個觀測值

```
### 作業:

```
波士頓房地產資料可以透過 load_boston() 方法讀入
糖尿病病患資料可以透過 load_diabetes() 方法讀入
scikit-learn comes with a few small standard datasets that 
do not require to download any file from some external website.

load_boston([return_X_y])	Load and return the boston house-prices dataset (regression).
load_iris([return_X_y])	Load and return the iris dataset (classification).
load_diabetes([return_X_y])	Load and return the diabetes dataset (regression).
load_digits([n_class, return_X_y])	Load and return the digits dataset (classification).
load_linnerud([return_X_y])	Load and return the linnerud dataset (multivariate regression).
load_wine([return_X_y])	Load and return the wine dataset (classification).
load_breast_cancer([return_X_y])	Load and return the breast cancer wisconsin dataset (classification).
```

