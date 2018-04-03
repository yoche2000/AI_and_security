# Getting started with TensorFlow
```
<<深度學習快速入門：使用TensorFlow>>
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
#first_session_only_tensorflow.py

import tensorflow as tf

x = tf.constant(1, name='x')
y = tf.Variable(x+9,name='y')


model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    print(session.run(y))

```


```
import tensorflow as tf

a = tf.placeholder("int32")
b = tf.placeholder("int32")

y = tf.multiply(a,b)

sess = tf.Session()

print sess.run(y, feed_dict={a : 2,b:5})
```

```
# -*- coding: utf-8 -*-
# Using Tensorboard
#----------------------------------
#
# We illustrate the various ways to use
#  Tensorboard

import os
import io
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Initialize a graph session
sess = tf.Session()

# Create a visualizer object
summary_writer = tf.summary.FileWriter('tensorboard', tf.get_default_graph())

# Create tensorboard folder if not exists
if not os.path.exists('tensorboard'):
    os.makedirs('tensorboard')
print('Running a slowed down linear regression. '
      'Run the command: $tensorboard --logdir="tensorboard"  '
      ' Then navigate to http://127.0.0.1:6006')

# You can also specify a port option with --port 6006

# Wait a few seconds for user to run tensorboard commands
time.sleep(3)

# Some parameters
batch_size = 50
generations = 100

# Create sample input data
x_data = np.arange(1000)/10.
true_slope = 2.
y_data = x_data * true_slope + np.random.normal(loc=0.0, scale=25, size=1000)

# Split into train/test
train_ix = np.random.choice(len(x_data), size=int(len(x_data)*0.9), replace=False)
test_ix = np.setdiff1d(np.arange(1000), train_ix)
x_data_train, y_data_train = x_data[train_ix], y_data[train_ix]
x_data_test, y_data_test = x_data[test_ix], y_data[test_ix]

# Declare placeholders
x_graph_input = tf.placeholder(tf.float32, [None])
y_graph_input = tf.placeholder(tf.float32, [None])

# Declare model variables
m = tf.Variable(tf.random_normal([1], dtype=tf.float32), name='Slope')

# Declare model
output = tf.multiply(m, x_graph_input, name='Batch_Multiplication')

# Declare loss function (L1)
residuals = output - y_graph_input
l2_loss = tf.reduce_mean(tf.abs(residuals), name="L2_Loss")

# Declare optimization function
my_optim = tf.train.GradientDescentOptimizer(0.01)
train_step = my_optim.minimize(l2_loss)

# Visualize a scalar
with tf.name_scope('Slope_Estimate'):
    tf.summary.scalar('Slope_Estimate', tf.squeeze(m))

# Visualize a histogram (errors)
with tf.name_scope('Loss_and_Residuals'):
    tf.summary.histogram('Histogram_Errors', l2_loss)
    tf.summary.histogram('Histogram_Residuals', residuals)



# Declare summary merging operation
summary_op = tf.summary.merge_all()

# Initialize Variables
init = tf.initialize_all_variables()
sess.run(init)

for i in range(generations):
    batch_indices = np.random.choice(len(x_data_train), size=batch_size)
    x_batch = x_data_train[batch_indices]
    y_batch = y_data_train[batch_indices]
    _, train_loss, summary = sess.run([train_step, l2_loss, summary_op],
                             feed_dict={x_graph_input: x_batch,
                                        y_graph_input: y_batch})

    test_loss, test_resids = sess.run([l2_loss, residuals], feed_dict={x_graph_input: x_data_test,
                                                                       y_graph_input: y_data_test})

    if (i+1)%10==0:
        print('Generation {} of {}. Train Loss: {:.3}, Test Loss: {:.3}.'.format(i+1, generations, train_loss, test_loss))

    log_writer = tf.summary.FileWriter('tensorboard')
    log_writer.add_summary(summary, i)
    time.sleep(0.5)

#Create a function to save a protobuf bytes version of the graph
def gen_linear_plot(slope):
    linear_prediction = x_data * slope
    plt.plot(x_data, y_data, 'b.', label='data')
    plt.plot(x_data, linear_prediction, 'r-', linewidth=3, label='predicted line')
    plt.legend(loc='upper left')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return(buf)

# Add image to tensorboard (plot the linear fit!)
slope = sess.run(m)
plot_buf = gen_linear_plot(slope[0])
# Convert PNG buffer to TF image
image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
# Add the batch dimension
image = tf.expand_dims(image, 0)
# Add image summary
image_summary_op = tf.summary.image("Linear Plot", image)
image_summary = sess.run(image_summary_op)
log_writer.add_summary(image_summary, i)
log_writer.close()
```


# Chapter 2 用TensorFlow求解數學問題
```
張量資料結構  複數及碎形（fractals）  計算梯度（gradient） 隨機數值
```
## 張量資料結構
```
import numpy as np

tensor_1d = np.array([1.3,1,4.0,23.99])

print tensor_1d

print tensor_1d[0]

print tensor_1d[2]

import tensorflow as tf

tf_tensor = tf.convert_to_tensor(tensor_1d, dtype=tf.float64)

with tf.Session() as sess:
    print sess.run(tf_tensor)
    print sess.run(tf_tensor[0])
    print sess.run(tf_tensor[2])


tensor_2d=np.array([(1,2,3,4),(4,5,6,7),(8,9,10,11),(12,13,14,15)])

print tensor_2d
print tensor_2d[3][3]
print tensor_2d[0:2,0:2]

tf_tensor=tf.placeholder("float64",tensor_2d,name='x')
with tf.Session() as sess:
    print sess.run(x)
```
## 複數及碎形（fractals）  
```
#Import libraries for simulation
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
     



#MANDELBROT SET
Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]

#JULIA SET
#Y, X = np.mgrid[-2:2:0.005, -2:2:0.005]

#Definiamo il punto corrente 
Z = X+1j*Y
c = tf.constant(Z.astype("complex64"))

zs = tf.Variable(c)
ns = tf.Variable(tf.zeros_like(c, "float32"))

#c = complex(0.0,0.75)
#c = complex(-1.5,-1.5)
sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

# Compute the new values of z: z^2 + x
zs_ = zs*zs + c
#zs_ = zs*zs - c

# Have we diverged with this new value?
not_diverged = tf.complex_abs(zs_) < 4

step = tf.group(
  zs.assign(zs_),
  ns.assign_add(tf.cast(not_diverged, "float32"))
  )

for i in range(200): step.run()

plt.imshow(ns.eval())
plt.show()

```
## 計算梯度（gradient） 隨機數值
```
import tensorflow as tf

def my_loss_function(var, data):
    return tf.abs(tf.subtract(var, data))

def my_other_loss_function(var, data):
    return tf.square(tf.subtract(var, data))

data = tf.placeholder(tf.float32)
var = tf.Variable(1.)
loss = my_loss_function(var, data)
var_grad = tf.gradients(loss, [var])[0]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    var_grad_val = sess.run(var_grad, feed_dict={data: 4})
    print(var_grad_val)
```

```
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
