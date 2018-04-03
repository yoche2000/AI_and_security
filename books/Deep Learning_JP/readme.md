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


## 第二章 感知器 
```
>* 1957年，Rosenblatt發明瞭感知機（或稱感知器，Perceptron），
>* 是神經網路的雛形，同時也是支援向量機的基礎，在當時引起了不小的轟動。
>* 感知機是二類分類的線性分類模型，其輸入為例項的特徵向量，輸出為例項的類別，取+1和-1二值。
>* 感知機對應於輸入空間（特徵空間）中將例項劃分為正負兩類的分離超平面，屬於判別模型。
>* 感知機學習旨在求出將訓練資料進行線性劃分的分離超平面。

```

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



## 第三章 神經網路   


```

```


## 第四章 神經網路的學習  

## 第五章 誤差反向傳播法 

## 第六章 與學習有關的技巧  

## 第七章 卷積神經網路  

## 第八章 深度學習 

附錄A Softmax-with-Loss層的計算圖 


```

```

```

```

```

```

```

```

```

```
