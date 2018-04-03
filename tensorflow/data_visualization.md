# 資料視覺化 data_visualization

```
使用套件
>* matplotlib
>* 
>* 

畫底下圖形:
直方圖（Histogram）
散佈圖（Scatter plot）
線圖（Line plot）
長條圖（Bar plot）
盒鬚圖（Box plot）
```

### 直方圖（Histogram）
```
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

normal_samples = np.random.normal(size = 100000) 
# 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數

uniform_samples = np.random.uniform(size = 100000) 
# 生成 100000 組介於 0 與 1 之間均勻分配隨機變數

plt.hist(normal_samples)
plt.show()
plt.hist(uniform_samples)
plt.show()

```


### 散佈圖（Scatter plot）
```
%matplotlib inline

import matplotlib.pyplot as plt

speed = [4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25]
dist = [2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85]

plt.scatter(speed, dist)
plt.show()

```


### 線圖（Line plot）
```
%matplotlib inline

import matplotlib.pyplot as plt

speed = [4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25]
dist = [2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85]

plt.plot(speed, dist)
plt.show()

```


### 長條圖（Bar plot）
```
# 使用 matplotlib.pyplot 的 bar() 方法。

%matplotlib inline

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

cyl = [6 ,6 ,4 ,6 ,8 ,6 ,8 ,4 ,4 ,6 ,6 ,8 ,8 ,8 ,8 ,8 ,8 ,4 ,4 ,4 ,4 ,8 ,8 ,8 ,8 ,4 ,4 ,4 ,8 ,6 ,8 ,4]

labels, values = zip(*Counter(cyl).items())
width = 1

plt.bar(indexes, values)
plt.xticks(indexes + width * 0.5, labels)
plt.show()
```


### 盒鬚圖（Box plot）
```
# 使用 matplotlib.pyplot 的 boxplot() 方法。

%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

normal_samples = np.random.normal(size = 100000) # 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數

plt.boxplot(normal_samples)
plt.show()
```


### 輸出圖形::使用圖形物件的 savefig() 方法
```
import numpy as np
import matplotlib.pyplot as plt

normal_samples = np.random.normal(size = 100000) # 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數

plt.hist(normal_samples)
plt.savefig(filename = "my_hist.png", format = "png")
```


### 
```

```


### 
```

```


### 
```

```


### 
```

```


### 
```

```


### 
```

```
