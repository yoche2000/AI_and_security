# ImageNet(2009)


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
