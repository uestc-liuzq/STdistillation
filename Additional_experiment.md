## Precisions on Time Series Classification
We also record the overall precision results on the task of time series classification. The results are as follows. TimeDC achieves the best performance among all the baselines. 

![](img/precision.png)
## Time of Coreset methods and TimeDC
We present the time of coreset constrcution and training time of coreset methods and TimeDC as follows, which shows the training time of TimeDC is comparable with those of coreset methods.

|            Dataset            | Weather |         |         |        |  ETTh1 |         |         |        |
|:-----------------------------:|:-------:|:-------:|:-------:|:------:|:------:|:-------:|:-------:|:------:|
|        Method (PL = 96)       |  Random | K-means | Herding | TimeDC | Random | K-means | Herding | TimeDC |
| Coreset Construction Time (s) |   1.85  |  10.14  |  63.66  |  None  |  2.00  |   6.07  |  69.01  |  None  |
|       Training Time (s)       |  20.37  |  20.40  |  20.38  |  20.41 |  9.54  |   9.56  |   9.55  |  9.56  |