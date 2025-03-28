# Less is More: Efficient Time Series Dataset Condensation via Two-fold Modal Matching (PVLDB 2025)

<img src="framework.png" width="90%" height="80%">

### Citation
Please cite the following paper if this paper/repository is useful for your research.
```
@article{miao2024less,
  title={Less is more: Efficient time series dataset condensation via two-fold modal matching},
  author={Miao, Hao and Liu, Ziqiao and Zhao, Yan and Guo, Chenjuan and Yang, Bin and Zheng, Kai and Jensen, Christian S},
  journal={PVLDB},
  volume={18},
  number={2},
  pages={226--238},
  year={2024}
}
```

## Running
- **Data Preparation:** _Weather_, _Traffic_, _Electricity_ and _ETT_ can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy).

- **Generating Expert Trajectories:** Run each script in ```./scripts_buffer/``` to generate expert trajectories, for example
    ```
    sh ./scripts_buffer/weather.sh
    ```

- **Time Series Dataset Condensation with TimeDC:** After obtaining expert trajectories, run each script in ```./scripts_distill/``` to perform time series dataset condensation, for example
    ```
    sh ./scipts_distill/weather.sh
    ```

<!--
## Additional Experiments

### Effect of Expert Trajectory Numbers
We study the effect of the number of expert trajectories on _Weather_ as shown in the following Table. From the Table, we can easily observe that as the increase of the Trajectories number, the performance of TimeDC gets better. This is because more expert trajectories may bring more knowledge, which gives more guidance on time series dataset condensation.
| Trajectory Number (PL=96)     | MAE     | RMSE    | Trajectory Number (PL=192)     | MAE     | RMSE   |
|:-----------------------------:|:-------:|:-------:|:-----------------------------:|:-------:|:-------:|
|        1                      |  0.341  | 0.303   |        1                      |  0.349  | 0.311   |
|        3                      |  0.324  | 0.286   |        3                      |  0.332  | 0.296   |
|        5                      |  0.306  | 0.275   |        5                      |  0.325  | 0.279   |
|        10                     |  0.257  | 0.188   |        10                     |  0.285  | 0.247   |

### Time Comparison Among Coreset Methods and TimeDC
We present the time of coreset construction and training time of coreset methods and TimeDC as follows, which shows the training time of TimeDC is comparable with those of coreset methods.

|            Dataset            | Weather |         |         |        |  ETTh1 |         |         |        |
|:-----------------------------:|:-------:|:-------:|:-------:|:------:|:------:|:-------:|:-------:|:------:|
|        Method (PL = 96)       |  Random | K-means | Herding | TimeDC | Random | K-means | Herding | TimeDC |
| Coreset Construction Time (s) |   1.85  |  10.14  |  63.66  |  None  |  2.00  |   6.07  |  69.01  |  None  |
|       Training Time (s)       |  20.37  |  20.40  |  20.38  |  20.41 |  9.54  |   9.56  |   9.55  |  9.56  |

### Effect of the Size of the Condensed Time Series (TS) Datasets across Different Methods
Please see the figures in the folder **./figures/** regarding effect of the size of the condensed time series (TS) datasets across different methods

### Precisions on Time Series Classification
We also record the overall precision results on the task of time series classification. The results are as follows. TimeDC achieves the best performance among all the baselines. 

<img src="precision.png" width="50%" height="50%">
-->

## Requirements
```
python >= 3.8
Pytorch >= 1.11
numpy >=1.21.2
torchvision >=0.12
```
