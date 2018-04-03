# 資料視覺化 data_visualization

使用套件
>* matplotlib
>* Seaborn
>* Bokeh
```
畫底下圖形:
直方圖（Histogram）
散佈圖（Scatter plot）
線圖（Line plot）
長條圖（Bar plot）
盒鬚圖（Box plot）
```
## matplotlib

https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day18.md

### 直方圖（Histogram）:使用 matplotlib.pyplot 的 hist() 方法
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


### 散佈圖（Scatter plot）:使用 matplotlib.pyplot 的 scatter() 方法
```
%matplotlib inline

import matplotlib.pyplot as plt

speed = [4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25]
dist = [2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85]

plt.scatter(speed, dist)
plt.show()

```


### 線圖（Line plot）:使用 matplotlib.pyplot 的 plot() 方法
```
%matplotlib inline

import matplotlib.pyplot as plt

speed = [4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25]
dist = [2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85]

plt.plot(speed, dist)
plt.show()

```


### 長條圖（Bar plot）:使用 matplotlib.pyplot 的 bar() 方法
```
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


### 盒鬚圖（Box plot）:使用 matplotlib.pyplot 的 boxplot() 方法
```
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
## Seaborn

https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day19.md

Seaborn 套件在我們的開發環境沒有安裝，但我們可以透過 conda 指令在終端機安裝。

$ conda install -c anaconda seaborn=0.7.1

我們的開發環境是 Jupyter Notebook，這個指令可以讓圖形不會在新視窗呈現。

%matplotlib inline

### 直方圖（Histogram）:使用 seaborn 套件的 distplot() 方法
```
%matplotlib inline

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

normal_samples = np.random.normal(size = 100000) 
# 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數

sns.distplot(normal_samples)

# 預設會附上 **kernel density estimate（KDE）**曲線
```


### 散佈圖（Scatter plot）:使用 seaborn 套件的 joinplot() 方法
```
%matplotlib inline

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

speed = [4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25]
dist = [2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85]

cars_df = pd.DataFrame(
    {"speed": speed,
     "dist": dist
    }
)

sns.jointplot(x = "speed", y = "dist", data = cars_df)
```


### 線圖（Line plot）:使用 seaborn 套件的 factorplot() 方法
```
%matplotlib inline

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

speed = [4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25]
dist = [2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85]

cars_df = pd.DataFrame(
    {"speed": speed,
     "dist": dist
    }
)

sns.factorplot(data = cars_df, x="speed", y="dist", ci = None)
```


### 長條圖（Bar plot）:使用 seaborn 套件的 countplot() 方法
```
%matplotlib inline

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

cyl = [6 ,6 ,4 ,6 ,8 ,6 ,8 ,4 ,4 ,6 ,6 ,8 ,8 ,8 ,8 ,8 ,8 ,4 ,4 ,4 ,4 ,8 ,8 ,8 ,8 ,4 ,4 ,4 ,8 ,6 ,8 ,4]
cyl_df = pd.DataFrame({"cyl": cyl})

sns.countplot(x = "cyl", data=cyl_df)
```


### 盒鬚圖（Box plot）:使用 seaborn 套件的 boxplot() 方法
```
%matplotlib inline

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

normal_samples = np.random.normal(size = 100000) # 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數
sns.boxplot(normal_samples)
```

## Bokeh

https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day20.md

Anaconda 版本已經安裝了 Bokeh 

如果你的版本沒有，只要在終端機執行這段程式即可

$ conda install -c anaconda bokeh=0.12.3

### 直方圖（Histogram）:使用 bokeh.charts 的 Histogram() 方法
```
from bokeh.charts import Histogram, show
import numpy as np

normal_samples = np.random.normal(size = 100000) 
# 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數

hist = Histogram(normal_samples)
show(hist)
```


### 散佈圖（Scatter plot）:使用 bokeh.charts 的 Scatter() 方法
```
from bokeh.charts import Scatter, show
import pandas as pd

speed = [4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25]
dist = [2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85]

cars_df = pd.DataFrame(
    {"speed": speed,
     "dist": dist
    }
)

scatter = Scatter(cars_df, x = "speed", y = "dist")
show(scatter)
```


### 線圖（Line plot）:使用 bokeh.charts 的 Line() 方法
```
from bokeh.charts import Line, show
import pandas as pd

speed = [4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25]
dist = [2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85]

cars_df = pd.DataFrame(
    {"speed": speed,
     "dist": dist
    }
)

line = Line(cars_df, x = "speed", y = "dist")
show(line)
```


### 長條圖（Bar plot）:使用 bokeh.charts 的 Bar() 方法
```
from bokeh.charts import Bar, show
import pandas as pd

cyls = [11, 7, 14]
labels = ["4", "6", "8"]
cyl_df = pd.DataFrame({
    "cyl": cyls,
    "label": labels
})

bar = Bar(cyl_df, values = "cyl", label = "label")
show(bar)
```


### 盒鬚圖（Box plot）:使用 bokeh.charts 的 BoxPlot() 方法
```
from bokeh.charts import BoxPlot, show, output_notebook
import pandas as pd

output_notebook()

mpg = [21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26, 30.4, 15.8, 19.7, 15, 21.4]
cyl = [6, 6, 4, 6, 8, 6, 8, 4, 4, 6, 6, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8, 4, 4, 4, 8, 6, 8, 4]

mtcars_df = pd.DataFrame({
    "mpg": mpg,
    "cyl": cyl
})

box = BoxPlot(mtcars_df, values = "mpg", label = "cyl")
show(box)
```
