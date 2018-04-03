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


```

```


## 第四章 神經網路的學習  

## 第五章 誤差反向傳播法 

## 第六章 與學習有關的技巧  

## 第七章 卷積神經網路  

### ImageNet(2009)
```
2009年，一群在普林斯頓大學電腦系的華人學者發表了論文 
"ImageNet: A large scale hierarchical image database)，
宣佈建立了第一個超大型圖像資料庫，供電腦視覺研究者使用。

這個資料庫建立之初，包含了三百二十萬個圖像。它的目的是要把英文裡的八萬個名詞，每個詞收集五百到一千個高清圖片，
存放到資料庫裡。最終達到五千萬以上的圖像。

2010年，以 ImageNet 為基礎的大型圖像識別競賽，
ImageNet Large Scale Visual Recognition Challenge 2010 (ILSVRC2010) 第一次舉辦。
ImageNet＠2010[第一屆]==>http://image-net.org/challenges/LSVRC/2010/pascal_ilsvrc.pdf
ImageNet＠2012==>CNN:::AlexNet　計算機視覺領域取得了重大成果
    多倫多大學的Geoffrey Hinton、Ilya Sutskever和Alex Krizhevsky提出了一種深度卷積神經網絡結構(CNN)：AlexNet，
    成績比當時的第二名高出41%。

2013 年的 ImageNet 競賽, 獲勝的團隊是來自紐約大學的研究生 Matt Zeiler, 其圖像識別模型 top 5 的錯誤率, 降到了 11.5%.
Zeiler 的模型共有六千五百萬個自由參數, 在 Nvidia 的GPU 上運行了整整十天才完成訓練.

2014年, 競賽第一名是來自牛津大學的 VGG 團隊, top 5 錯誤率降到了 7.4%.
VGG的模型使用了十九層卷積神經網路, 一點四億個自由參數, 在四個 Nvidia 的 GPU 上運行了將近三周才完成培訓.

如何繼續提高模型的識別能力? 是不斷增加網路的深度和參數數目就可以簡單解決的嗎?

2015 年
來自微軟亞洲研究院的何愷明和孫健 (Jian Sun, 音譯), 西安交通大學的張翔宇 (Xiangyu Zhang, 音譯), 
中國科技大學的任少慶 (Shaoqing Ren, 音譯)四人的團隊 MSRA (MicroSoft Research Asia),
在2015 年十二月的 Imagenet 圖像識別的競賽中, 橫空出世.

他們研究的第一個問題是,一個普通的神經網路,是不是簡單地堆砌更多層神經元,就可以提高學習能力?
在研究一個圖像識別的經典問題 CIFAR-10 的時候,他們發現一個 56層的簡單神經網路,
識別錯誤率反而高於一個20層的模型.


```

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

### 語音辨識
```
一直到2009年之前, 主流的語音辨識技術, 依靠的是統計學上的兩個演算法模型:
高斯混合模型 (Gaussian Mixture Model)
隱藏瑪律科夫模型 (Hidden Markov Model)

2009年, Hinton 和他的研究生, Ahmed-Rahman Mohamed 和 George Dahl, 合作發表論文, 
"Deep Belief Network for Phone Recognition" (深信度網路用於電話語音辨識), 
在一個叫做 TIMIT 的標準測試上, 識別錯誤率降到了 23%, 超過了當時其它所有演算法的技術水準.

Hinton 和鄧力早在九十年代初就有聯絡與合作. Hinton 和他的研究生, 
2009年被邀請來微軟合作研究, 把深度學習的最新成就應用到語音辨識上.

http://wangchuan.blog.caixin.com/archives/146374

```

```
Heroes of Deep Learning: Andrew Ng interviews Geoffrey Hinton
https://www.youtube.com/watch?v=-eyhCTvrEtE

Geoffrey Hinton talk "What is wrong with convolutional neural nets ?" 
https://www.youtube.com/watch?v=rTawFwUvnLE

Heroes of Deep Learning: Andrew Ng interviews Ian Goodfellow
https://www.youtube.com/watch?v=pWAc9B2zJS4



 
 
  
  Deep Learning Chapter 1
  Introduction presented by Ian Goodfellow

  
  
  https://www.youtube.com/watch?v=vi7lACKOUao

  
 

```

```

```

```

```

```

```
