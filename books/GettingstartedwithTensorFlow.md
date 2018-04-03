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
### monte carlo method
```
import tensorflow as tf

trials = 100
hits = 0
x = tf.random_uniform([1],minval=0,maxval=1,dtype=tf.float32)
y = tf.random_uniform([1],minval=0,maxval=1,dtype=tf.float32)

sess = tf.Session()
with sess.as_default():
    for i in range(1,trials):
        for j in range(1,trials):
            if x.eval()**2 + y.eval()**2 < 1 :
                hits = hits + 1
        print (4 * float(hits) / i)/trials
```

### random number亂數產生

```
import tensorflow as tf
import matplotlib.pyplot as plt
#import matplotlib

# Create a tensor of shape [100] consisting of random normal values, with mean
# 0 and standard deviation 2.
norm = tf.random_normal([100], mean=0, stddev=2)
with tf.Session() as session:
    plt.hist(norm.eval(),normed=True)
    plt.show()


uniform = tf.random_uniform([100],minval=0,maxval=1,dtype=tf.float32)
with tf.Session() as session:
    print uniform.eval()
    plt.hist(uniform.eval(),normed=True)
    plt.show()


uniform_with_seed = tf.random_uniform([1], seed=1)
uniform_without_seed = tf.random_uniform([1])

# Repeatedly running this block with the same graph will generate the same
# sequence of values for 'a', but different sequences of values for 'b'.
print("First Run")
with tf.Session() as first_session:
  print("uniform with (seed = 1) = {}"\
        .format(first_session.run(uniform_with_seed)))
  print("uniform with (seed = 1) = {}"\
        .format(first_session.run(uniform_with_seed)))
  print("uniform without seed = {}"\
        .format(first_session.run(uniform_without_seed)))
  print("uniform without seed = {}"\
        .format(first_session.run(uniform_without_seed)))

print("Second Run")
with tf.Session() as second_session:
  print("uniform with (seed = 1) = {}"\
        .format(second_session.run(uniform_with_seed)))
  print("uniform with (seed = 1) = {}"\
        .format(second_session.run(uniform_with_seed)))
  print("uniform without seed = {}"\
        .format(second_session.run(uniform_without_seed)))
  print("uniform without seed = {}"\
        .format(second_session.run(uniform_without_seed)))



import tensorflow as tf

trials = 100
hits = 0
x = tf.random_uniform([1],minval=-1,maxval=1,dtype=tf.float32)
y = tf.random_uniform([1],minval=-1,maxval=1,dtype=tf.float32)
pi = []
sess = tf.Session()
with sess.as_default():
    for i in range(1,trials):
        for j in range(1,trials):
            if x.eval()**2 + y.eval()**2 < 1 :
                hits = hits + 1
                pi.append((4 * float(hits) / i)/trials)

plt.plot(pi)
plt.show()
```



### partial differential equation(略)



## Chapter 3 機器學習簡介與應用

線性迴歸演算法   分類（Classifiers）  資料群集（Data clustering）

```
import input_data
import numpy as np
import matplotlib.pyplot as plt

#Using input_data we load the data sets :

mnist_images = input_data.read_data_sets\
               ("MNIST_data/",\
                one_hot=False)

train.next_batch(10) returns the first 10 images :
pixels,real_values = mnist_images.train.next_batch(10)

#it also returns two lists, the matrix of the pixels loaded, and the list that contains the real values loaded:

print "list of values loaded ",real_values
example_to_visualize = 5
print "element N?" + str(example_to_visualize + 1)\
                    + " of the list plotted"

```
## KMeans 
```
K-Means 是 J. B. MacQueen 於1967年所提出的分群演算法
必須事前設定群集的數量 k，然後找尋下列公式的極大值，以達到分群的最佳化之目的

```
### 演算法
```

1. 隨機指派群集中心：
    在訓練組資料中「隨機」找出K筆紀錄來作為初始種子(初始群集的中心)
2. 產生初始群集：
    計算每一筆紀錄到各個隨機種子之間的距離，然後比較該筆紀錄究竟離哪一個隨機種子最近，
    然後這筆紀錄就會被指派到最接近的那個群集中心，此時就會形成一個群集邊界，
    產生了初始群集的成員集合
3. 產生新的質量中心：
    根據邊界內的每一個案例重新計算出該群集的質量中心，
    利用新的質量中心取代之前的隨機種子，來做為該群的中心
4. 變動群集邊界：
    指定完新的質量中心之後，再一次比較每一筆紀錄與新的群集中心之間的距離，
    然後根據距離，再度重新分配每一個案例所屬的群集

5. 持續反覆 3, 4 的步驟，一直執行到群集成員不再變動為止
```
### 演算法實作
```
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

# Ignore all GPUs, tf random forest does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
full_data_x = mnist.train.images

# Parameters
num_steps = 50 # Total steps to train
batch_size = 1024 # The number of samples per batch
k = 25 # The number of clusters
num_classes = 10 # The 10 digits
num_features = 784 # Each image is 28x28 pixels

# Input images
X = tf.placeholder(tf.float32, shape=[None, num_features])
# Labels (for assigning a label to a centroid and testing)
Y = tf.placeholder(tf.float32, shape=[None, num_classes])

# K-Means Parameters
kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine',
                use_mini_batch=True)

# Build KMeans graph
(all_scores, cluster_idx, scores, cluster_centers_initialized, init_op,
train_op) = kmeans.training_graph()
cluster_idx = cluster_idx[0] # fix for cluster_idx being a tuple
avg_distance = tf.reduce_mean(scores)

# Initialize the variables (i.e. assign their default value)
init_vars = tf.global_variables_initializer()

# Start TensorFlow session
sess = tf.Session()

# Run the initializer
sess.run(init_vars, feed_dict={X: full_data_x})
sess.run(init_op, feed_dict={X: full_data_x})

# Training
for i in range(1, num_steps + 1):
    _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                         feed_dict={X: full_data_x})
    if i % 10 == 0 or i == 1:
        print("Step %i, Avg Distance: %f" % (i, d))

# Assign a label to each centroid
# Count total number of labels per centroid, using the label of each training
# sample to their closest centroid (given by 'idx')
counts = np.zeros(shape=(k, num_classes))
for i in range(len(idx)):
    counts[idx[i]] += mnist.train.labels[i]
# Assign the most frequent label to the centroid
labels_map = [np.argmax(c) for c in counts]
labels_map = tf.convert_to_tensor(labels_map)

# Evaluation ops
# Lookup: centroid_id -> label
cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
# Compute accuracy
correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Test Model
test_x, test_y = mnist.test.images, mnist.test.labels
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))

```



```


```


```


```

```
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
