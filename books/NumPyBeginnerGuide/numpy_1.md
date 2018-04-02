# NumPy Beginner's Guide - Second Edition Ivan Idris April 2013

有新版: NumPy: Beginner's Guide - Third Edition  Ivan Idris  June 2015

### 兩個向量a和b做加法運算
```
向量a的取值為0~n的整數的平方，例如n取2時，向量a為0、1或4。
向量b的取值為0~n的整數的立方，例如n取3時，向量b為0、1或8。

向量===數學意義上的一維陣列，
學習如何用NumPy陣列表示矩陣。
```
```
def pythonsum(n):
  a = range(n)
  b = range(n)
  c = []

for i in range(len(a)):
  a[i] = i ** 2
  b[i] = i ** 3
  c.append(a[i] + b[i])
  return c
```

```
def numpysum(n):
  a = numpy.arange(n) ** 2
  b = numpy.arange(n) ** 3
  c = a + b
return c
```
### NumPy在陣列操作上的效率優於純Python代碼。

```
那麼究竟快多少呢？接下來的程式將告訴我們答案，它以微秒的精度分別
記錄下numpysum()和pythonsum()函數的耗時
```

```
#!/usr/bin/env/python

import sys
from datetime import datetime
import numpy as np

"""
 Chapter 1 of NumPy Beginners Guide.
 This program demonstrates vector addition the Python way.
 Run from the command line as follows
     
  python vectorsum.py n
 
 where n is an integer that specifies the size of the vectors.

 The first vector to be added contains the squares of 0 up to n. 
 The second vector contains the cubes of 0 up to n.
 The program prints the last 2 elements of the sum and the elapsed time.
"""

def numpysum(n):
   a = np.arange(n) ** 2
   b = np.arange(n) ** 3
   c = a + b

   return c

def pythonsum(n):
   a = range(n)
   b = range(n)
   c = []

   for i in range(len(a)):
       a[i] = i ** 2
       b[i] = i ** 3
       c.append(a[i] + b[i])

   return c
   

size = int(sys.argv[1])

start = datetime.now()
c = pythonsum(size)
delta = datetime.now() - start
print "The last 2 elements of the sum", c[-2:]
print "PythonSum elapsed time in microseconds", delta.microseconds

start = datetime.now()
c = numpysum(size)
delta = datetime.now() - start
print "The last 2 elements of the sum", c[-2:]
print "NumPySum elapsed time in microseconds", delta.microseconds
```
```
python vectorsum.py 44444
```
```

```
```

```
```

```
  
