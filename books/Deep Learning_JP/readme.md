# Deep Learning：用Python進行深度學習的基礎理論實作
```
https://github.com/oreilly-japan/deep-learning-from-scratch
作者： 斎藤康毅  譯者： 吳嘉芳
出版社：歐萊禮  出版日期：2017/08/17
第一章 Python入門   第二章 感知器 
第三章 神經網路   第四章 神經網路的學習  第五章 誤差反向傳播法 
第六章 與學習有關的技巧  第七章 卷積神經網路  第八章 深度學習 
附錄A Softmax-with-Loss層的計算圖 
```
## 第一章 Python入門

### 使用Matplotlib畫圖===>資料視覺化

```
pyplot提供顯示影像用的方法imshow()
如果要載入影像，可以利用matplotlib.image模組的 imread()
```

```
# coding: utf-8
# img_show.py

import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('../dataset/lena.png') 
plt.imshow(img)

plt.show()

```


```
# coding: utf-8
# man.py

class Man:
    """サンプルクラス"""

    def __init__(self, name):
        self.name = name
        print("Initilized!")

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("Good-bye " + self.name + "!")

m = Man("David")
m.hello()
m.goodbye()
```
如果要繪製圖表，必須利用 matplotlib 的 pyplot 模組

繪製 sin 函數

```
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1) 
y = np.sin(x)

plt.plot(x, y)
plt.show()
```
加上標題與 x 軸標籤名稱
```
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# データの作成
x = np.arange(0, 6, 0.1) # 0から6まで0.1刻みで生成
y1 = np.sin(x)
y2 = np.cos(x)

# グラフの描画
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle = "--", label="cos")
plt.xlabel("x") 
```


## 第二章 感知器 Perceptron
```
>* 1957年，Rosenblatt發明瞭感知機（或稱感知器，Perceptron），
>* 是神經網路的雛形，同時也是支援向量機的基礎，在當時引起了不小的轟動。
>* 感知機是二類分類的線性分類模型，其輸入為例項的特徵向量，輸出為例項的類別，取+1和-1二值。
>* 感知機對應於輸入空間（特徵空間）中將例項劃分為正負兩類的分離超平面，屬於判別模型。
>* 感知機學習旨在求出將訓練資料進行線性劃分的分離超平面。
>* https://en.wikipedia.org/wiki/Frank_Rosenblatt
>* https://en.wikipedia.org/wiki/Perceptron
```
```
>* 1969年，Minskey提出了著名的XOR問題
>* "The Perceptron Controversy"或"The XOR Affair"
>* 感知器在類似XOR問題的線性不可分資料的無力

https://www.readhouse.net/articles/161164446/
http://www.ifuun.com/a201664130138/
```
### 感知器 Perceptrond可解AND
```
# coding: utf-8
import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = AND(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
```
### 感知器 Perceptrond可解OR

```
# coding: utf-8
import numpy as np


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = OR(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
```
### 感知器 Perceptrond可解NAND

```
# coding: utf-8
import numpy as np

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = NAND(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
```
### 要解XOR==>多層

```
# coding: utf-8
from and_gate import AND
from or_gate import OR
from nand_gate import NAND


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = XOR(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
```


## 第三章 神經網路   

```
1986年七月，Geoffrey Hinton 和 David Rumelhart 合作在自然雜誌上發表論文
"Learning Representations by Back-propagating errors"， 
第一次系統簡潔地闡述，反向傳播演算法在神經網路模型上的應用。
使用了反向傳播演算法的神經網路，在做諸如形狀識別之類的簡單工作時，效率比感知器大大提高。
八十年代末電腦的運行速度，也比二十年前高了幾個數量級。
神經網路的研究開始復蘇。
```

### 活化函數(activation function)
```
神經網路架構::輸入層==>中間層(隱藏層hidden layer)==>輸出層
活化函數(activation function)==>以臨界值為分界來轉換輸出的函數{階梯函數step function 步階函數}
https://en.wikipedia.org/wiki/Activation_function
https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
```
### 活化函數(activation function):step_function.py

```
# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int)

X = np.arange(-5.0, 5.0, 0.1)
Y = step_function(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1) 
plt.show()

```

### 活化函數(activation function):sigmoid
```
# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

X = np.arange(-5.0, 5.0, 0.1)
Y = sigmoid(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.show()

```


### 活化函數(activation function)比較:比較 sigmoid 函數與階梯函數
```
# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

def step_function(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = step_function(x)

plt.plot(x, y1)
plt.plot(x, y2, 'k--')
plt.ylim(-0.1, 1.1) 
plt.show()
```


### 活化函數(activation function):ReL U ( Rectified Linear Unit ） 函數
```
# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1.0, 5.5)
plt.show()

```
多維陣列的運算
矩陣的乘積


### 輸出層的設計
```
神經網路可以用來解決分類問題與迴歸問題。 
解決分類問題或迴歸問題時，必須改變輸出層的活化函數。 
一般而言，迴歸問題要使用恆等函數， 而分類問題使用的是 softmax 函數。

機器學習的問題大致可以分成 分類問題與迴歸問題 

分類問題是指資料屬於哪種類別的問題 
例如，從拍緝的人像中 ，要分類那個人是 男性還是女性，這就是分類問題。

迴歸問題是從輸入資料中預測(連續性)數值的問題。
例如，從拍攝的人像中預測那個人的體重

恆等函數與 softmax 函數
```

### 分類手寫數字影像
```
step1:學習:使用訓練資料(學習資料)進行權重參數學習===>model
step2:推論:使用學習完的參數(model)執行神經網路的推論。 
神經網路的正向傳播(forward propagation)
```

#### MNIST資料集
```
http://yann.lecun.com/exdb/mnist/
MNIST 的手寫數字影像集是機器學習領域中是最有名的資料集之一，
運用於各種場合包括簡單的實驗或當作論文發表的研究。 

MNIST 資料集是由 0~9 的數字影像構成
MNIST 資料集分成三個部分
55,000 筆的 training data (mnist.train)
10,000 筆的 test data (mnist.test)
5,000 筆的 validation data (mnist.validation)
MNIST 的圖片是 28 像素 x 28 像素，每一張圖片就可以用 28 x 28 = 784 個數字來紀錄
使用這些影像可以進行學習與推論
在一般的 MNIST 資料集用法中，通常會使用訓練影像進行學習，再利用學習後的模型，預測能否正確分類測試影像。

```
```
下載並熟悉 MNIST data
https://ithelp.ithome.com.tw/articles/10186473

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 來看看 mnist 的型態
print type(mnist)
print mnist.train.num_examples
print mnist.validation.num_examples
print mnist.test.num_examples

print("讓我們看一下 MNIST 訓練還有測試的資料集長得如何")
train_img = mnist.train.images
train_label = mnist.train.labels
test_img = mnist.test.images
test_label = mnist.test.labels
print
print(" train_img 的 type : %s" % (type(train_img)))
print(" train_img 的 dimension : %s" % (train_img.shape,))
print(" train_label 的 type : %s" % (type(train_label)))
print(" train_label 的 dimension : %s" % (train_label.shape,))
print(" test_img 的 type : %s" % (type(test_img)))
print(" test_img 的 dimension : %s" % (test_img.shape,))
print(" test_label 的 type : %s" % (type(test_label)))
print(" test_label 的 dimension : %s" % (test_label.shape,))


每個 image 有 784 個數字，因為每張圖片其實是 28 pixels X 28 pixels，我們可以把它看成一個很大的 array

tensor 形狀為 [55000, 784]
第一個維度指的是圖片的 index
第二個則是每個圖片的 pixel 點，這個 pixel 點是一個介於 0 到 1 的值，來表示 pixel 點的強度

每個 MNIST 中的圖片都有一個對應的 label 也就是從 0 到 9 的數值．
在這裡每個 label 都是一個 one-hot vectors． 
one-hot vector 是指說只有一個維度是 1 其他都是 0．
數字 n 表示一個只有在第 n 維度（從 0 開始）數字為 1 的 10 維向量．
label 0 的表示法就是（[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]．
mnist.train.labels 是一個 [60000, 10] 的矩陣．


trainimg = mnist.train.images
trainlabel = mnist.train.labels
nsample = 1
randidx = np.random.randint(trainimg.shape[0], size=nsample)

for i in [0, 1, 2]:
    curr_img   = np.reshape(trainimg[i, :], (28, 28)) # 28 by 28 matrix 
    curr_label = np.argmax(trainlabel[i, :] ) # Label
    plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
    plt.title("" + str(i + 1) + "th Training Data " 
              + "Label is " + str(curr_label))
```
```
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 形状を元の画像サイズに変形
print(img.shape)  # (28, 28)

img_show(img)

```

```
neuralnet_mnist.py
https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch03/neuralnet_mnist.py
```

```
neuralnet_mnist_batch.py
https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch03/neuralnet_mnist_batch.py
```

## 第四章 神經網路的學習  

## 第五章 誤差反向傳播法 

## 第六章 與學習有關的技巧  

## 第七章 卷積神經網路  



## 第八章 深度學習 

```
Hinton 教授和他的兩個研究生, Alex Krizhevsky 和 Ilya Sutskever, 2012 年底成立了一個名叫 DNNresearch 
(深度神經網路研究)的公司, 三個月後就被google以五百萬美元收購. 
Hinton 從此一半時間留在多倫多大學,另外一半時間在矽谷. 另外兩位研究生則成為谷歌的全職雇員.

原來在紐約大學教書的楊立昆 (Yann LeCun), 2013 年底被臉書聘請為其人工智慧研究院的總管.

曾在斯坦福大學和穀歌工作的吳恩達 (Andrew Ng), 2012年創立了網上教育公司 Coursera, 
2014年五月被百度聘任為首席科學家, 負責百度大腦的計畫.

```

附錄A Softmax-with-Loss層的計算圖 

