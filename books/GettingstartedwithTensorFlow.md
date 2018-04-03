# Getting started with TensorFlow<<深度學習快速入門：使用TensorFlow>>
```
作者： Giancarlo Zaccone  譯者： 傅運文
出版社：博碩  出版日期：2017/01/11
```
TensorFlow環境安裝


```
Chapter 1 TensorFlow：基本概念
機器學習與深度學習的基礎
TensorFlow：總體概述
Python的基礎
安裝TensorFlow
第一次實地操作 資料流圖形
TensorFlow程式設計模型 如何使用TensorBoard
```
```
#first_session.py

#a simple Python code
x = 1
y = x + 9
print(y)

#....and the tensorflow translation of the previous code
import tensorflow as tf

x = tf.constant(1, name='x')
y = tf.Variable(x+9,name='y')
print(y)

```

```

```
Chapter 2 用TensorFlow求解數學問題
張量資料結構  複數及碎形（fractals）  計算梯度（gradient） 隨機數值

Chapter 3 機器學習簡介與應用
線性迴歸演算法   分類（Classifiers）  資料群集（Data clustering）

Chapter 4 類神經網路簡介
什麼是類神經網路？
單層感知器及其應用案例:邏輯斯迴歸（logistic regression） 
多層感知器及其應用案例:函數近似（function approximation）

Chapter 5 深度學習
深度學習技術 | 卷積神經網路CNN  CNN架構  CNN的TensorFlow實作
遞迴神經網路RNN  RNN架構  LSTM網路  使用TensorFlow進行自然語言處理

Chapter 6 GPU程式設計和TensorFlow服務
GPU程式設計 TensorFlow服務（TensorFlow Serving）
如何安裝TensorFlow Serving  如何使用TensorFlow Serving
訓練和輸出模型  執行session  載入與輸出一個TensorFlow模型  測試伺服器
```
