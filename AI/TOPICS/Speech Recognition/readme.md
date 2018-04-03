# 語音辨識

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
2012年十月, Geoffrey Hinton, 鄧力和其他幾位代表四個不同機構 (多倫多大學, 微軟, 穀歌, IBM) 的研究者, 聯合發表論文,
"深度神經網路在語音辨識的聲學模型中的應用: 四個研究小組的共同觀點" 
(Deep Neural Networks for Acoustic Modelling in Speech Recognition: The Shared Views of Four Research Groups ).

研究者們借用了Hinton 使用的"限制玻爾茲曼機" (RBM) 的演算法, 對神經網路進行了"預培訓". 
深度神經網路模型 (DNN), 在這裡, 替代了高斯混合模型 (GMM), 來估算識別文字的幾率.
DNN-HMM 模型的表現, 全面超越了傳統的 GMM-HMM模型, 有的時候錯誤率降低超過20%以上.
在google的一個語音輸入基準測試中,單詞錯誤率 (Word Error Rate) 最低達到了 12.3%
谷歌的研究者 Jeff Dean 評價, "這是20年來,在語音辨識領域, 最大的一次性進步. ".

2013年三月, 多倫多大學的 Alex Graves 領銜發表論文, "深度迴圈神經網路用於語音辨識" 
(Speech Recognition with Recurrent Neural Network). 
論文中使用 RNN/LSTM 的技術, 一個包含三個隱層, 四百三十萬個自由參數的網路, 
在一個叫做 TIMIT 的基準測試中, 所謂的"音位錯誤率"達到 17.7%, 優於同期的其它所有技術的表現水準.

2015年五月, Google宣佈, 依靠 RNN/LSTM 相關的技術, Google Voice的單詞錯誤率降到了8% (正常人大約 4%).

2015年十二月, 百度 AI 實驗室的 Dario Amodei 領銜發表論文, "英語和漢語的端對端的語音辨識". 
(之所以叫端對端, 是指一個模組就可以解決整個問題, 不需要多個模組和太多人工干預.)

論文的模型, 使用的是 LSTM 的一個簡化的變種, 叫做"封閉迴圈單元" (Gated Recurrent Unit).

百度的英文語音辨識系統, 接受了將近一萬兩千小時的語音訓練, 在 16個GPU上完成訓練需要 3-5 天.
在一個叫 WSJ Eval'92 的基準測試中, 其單詞錯誤率達到3.1%, 已經超過正常人的識別能力 (5%).  
在另外一個小型漢語基準測試中, 機器的識別錯誤率只有3.7%, 而一個五人小組的集體識別錯誤率則為4%.

按照這個趨勢, 機器在語音辨識的各種基準測試上的準確度, 很快將全面趕上並且超過普通人了. 
這是在圖像識別之後, 人工智慧即將攻克的另一個難關.

百度的吳恩達這樣說, "許多人不理解準確度在 95% 和 99% 之間的差異. 
99%將改變遊戲規則.當你有 99%的正確率時, 用戶的體驗不會有(明顯)損害".

```
```
Heroes of Deep Learning: Andrew Ng interviews Geoffrey Hinton
https://www.youtube.com/watch?v=-eyhCTvrEtE

Geoffrey Hinton talk "What is wrong with convolutional neural nets ?" 
https://www.youtube.com/watch?v=rTawFwUvnLE

Heroes of Deep Learning: Andrew Ng interviews Ian Goodfellow
https://www.youtube.com/watch?v=pWAc9B2zJS4

Deep Learning Chapter 1 Introduction presented by Ian Goodfellow
https://www.youtube.com/watch?v=vi7lACKOUao


```


```


```

```

```
