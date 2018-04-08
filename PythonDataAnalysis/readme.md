# 開發環境

```
anaconda===open data platform(conda)
conda==package and env management

lab1:完成安裝anaconda==>裝在D:\anaconda3
建議安裝時要設路徑

測試是否成功:D:\anaconda3\scripts\conda
可以看到很多指令參數

查看版本==>D:\anaconda3\scripts\conda -V

使用jupyter
在你的目錄底下啟動jupyter==>jupyter notebook

使用http://nbviewer.jupyter.org/

lab2:完成jupyter程式開發與程式執行範例

```

### 推薦
```
Python 資料運算與分析實戰：一次搞懂 NumPy•SciPy•Matplotlib•pandas 最強套件
作者： 中久喜健司  譯者： 莊永裕 出版社：旗標   出版日期：2018/02/05
```
```
Python3 for Data Science 
https://www.udemy.com/python-for-data-science/?couponCode=PYTHON-DATA-SCI10
https://github.com/udemy-course/python-data-science-intro
線上學習程式碼
http://nbviewer.jupyter.org/github/udemy-course/python-data-science-intro/tree/master/
```
### 自我學習 
```
R 語言使用者的 Python 學習筆記
https://github.com/yaojenkuo/learn_python_for_a_r_user
```
### 核心觀念:善用第三方套件+資料型態及其運算
```
Python基本資料型態: list,Tuple,dictionary,
Numpy資料型態:ndarray
Pandas資料型態:series, dataframe,Panel

資料視覺化
matplotlib==>mplot3d
Seaborn
Bokeh

scipy
scikit-learn(機器學習)
tensorflow(深度學習)
```
# Python基本資料型態: list,Tuple,dictionary,set
```
```
# Numpy資料型態:ndarray
```
Array
lab1:建立1D array
lab2:建立2D array3*3
lab3:array屬性shape/size/dtype
list_1 = [1,2,3,4]
list_2 = [2,2,6,4]
array_1 = np.array([list_1,list_2])
array屬性shape/size/dtype===>
array_1.shape
array_1.size
array_1.dtype

lab4:
array_4 = 
np.zeros(5)
lab5:slice運算
c = np.array([[1,2,3],[4,5,6],[7,8,9]])
c
c[1:,0:2]
c[1:2,]
c[:,1:2]

Array and matrix manipulation
lab1:快速創建array(下列何者為誤??)
    np.random.randint(10,size = 20)==>產生大小20的array
    np.random.randint(10,size = 20).reshape(4,5)
    np.random.randint(10,size = 20).reshape[4,5]
    randdn和randint有何差異??
    
lab2:array運算a{+-*/}b
lab3:建立matrix:np.mat(a)====> array to matrix
     a = np.mat(np.random.randint(10,size = 20).reshape[4,5])
     b = np.mat(np.random.randint(10,size = 20).reshape[5,4])
     a*b==>?*?的matrix

lab4:matrix運算
lab5:array常用函式

universal function
nda = np.arange(12).reshape(2,6)
nda
np.square(nda)

broadcasting機制
nda = np.array([1,2,3])
nda+1
```
```
https://docs.scipy.org/doc/numpy/reference/index.html
```
# matplotlib
```
# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('../dataset/lena.png') 
plt.imshow(img)

plt.show()
#https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch01/img_show.py
# 更多參數說明,請參閱底下無範例
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
```
```
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1) 
y = np.sin(x)

plt.plot(x, y)
plt.show()
```
```
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 6, 0.1) 
y1 = np.sin(x)
y2 = np.cos(x)


plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle = "--", label="cos")
plt.xlabel("x") 
plt.ylabel("y") 
plt.title('sin & cos')
plt.legend()
plt.show()
```

```
import numpy as np
import matplotlib.pyplot as plt

a = [1, 2, 3]
b = [4, 5, 6]
plt.plot(a, b)
plt.plot(a, b, '*')
```

#### 3. Histogram and KDE Plot
```

```

```
```


# Pandas資料型態:series, dataframe,Panel
```
lab1:建立series的三種方法
lab2:series的運算
lab3:建立dataframe的種方法
lab4:dataframe的運算

```
#### lab1:建立series的三種方法
```
#先建立一個list再用Series轉成series

from pandas import Series

a = [1, 2, 3, 4]
s = Series(a)
s.index
s.values
s1 = Series(a, index=['A','B','C','D'])
```

```
import numpy as np
s2 = Series(np.arange(5))
```

```
d = {'A':1,'B':2,'C':3,'D':4}
s3 = Series(d)
s3.to_dict()
```

https://pandas.pydata.org/pandas-docs/stable/io.html


# 科學運算的python:scipy

```
功能:
內插法
統計分析
Fast fourier transform
```
### 統計分析之rayleigh機率分布

```
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# (1)統計分佈函式的設定（預先Freeze）
rv = sp.stats.rayleigh(loc=1)

# (2)以上述統計分佈函式生成的亂數變數
r = rv.rvs(size=3000)

# (3)機率密度函式繪製用的百分點資料列
x = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 100)

# 將取樣資料的分佈與原本的機率密度函數一同繪製
plt.figure(1)
plt.clf()
plt.plot(x, rv.pdf(x), 'k-', lw=2, label='機率密度函數')
plt.hist(r, normed=True, histtype='barstacked', alpha=0.5)
plt.xlabel('值')
plt.ylabel('分佈度')
plt.show()

```
### 內插法

```

import numpy as np
import matplotlib.pyplot as plt

# (1)由於名稱略長附加別名
import scipy.interpolate as ipl


# (2)原本的函數定義
def f(x):
    return (x-7) * (x-2) * (x+0.2) * (x-4)

# (3)生成原始資料（正解的值）
x = np.linspace(0, 8, 81)
y = np.array(list(map(f, x)))

# (4)補值前的寬刻度資料
x0 = np.arange(9)
y0 = np.array(list(map(f, x0)))

# (5)設定補值函式（線性補值）
#  設定補值函式（線性補值／3次樣條）
f_linear = ipl.interp1d(x0, y0, bounds_error=False)
f_cubic = ipl.interp1d(x0, y0, kind='cubic', bounds_error=False)
#  補值處理的執行
y1 = f_linear(x)  # 線性補值
y2 = f_cubic(x)  # 3次樣條補值

# (6)補值資料與原始資料的比較繪製
plt.figure(1)
plt.clf()
plt.plot(x, y, 'k-', label='原始函數')
plt.plot(x0, y0, 'ko', label='補值前資料', markersize=10)
plt.plot(x, y1, 'k:', label='線性補值', linewidth=4)
plt.plot(x, y2, 'k--', label='3次樣條補值', linewidth=4, alpha=0.7)
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.grid('on')
plt.show()
```
