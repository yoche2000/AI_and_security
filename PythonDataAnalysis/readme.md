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



```
```

# Pandas資料型態:series, dataframe,Panel
```
lab1:建立series的三種方法
lab2:series的運算
lab3:建立dataframe的三種方法
lab4:dataframe的運算

```
