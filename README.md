# Adaptive Prediction Models

The reproduction of the paper [Adaptive Prediction Models for Data Center Resources Utilization Estimation](https://ieeexplore.ieee.org/document/8786216)

## Requirements

- python >= 3.9

Please install other packages according `requirements.txt`.

## Dataset

### Alibaba

1. Download `cluster-trace-v2017` from [alibaba/clusterdata](https://github.com/alibaba/clusterdata).

2. Put `server_usage.csv` in `data/trace_201708/`.
    ```
    $ mkdir -p data/trace_201708
    $ mv <your_downloaded_folder>/server_usage.csv data/trace_201708
    ```

### Kaggle Demand

1. Download `train_0irEZ2H.csv` from [kaggle/demand-forecasting](https://www.kaggle.com/datasets/aswathrao/demand-forecasting?select=train_0irEZ2H.csv).

2. Rename `train_0irEZ2H.csv` to `train.csv`.

3. Put `train.csv` in `data/kaggle_demand`.
    ```
    $ mkdir -p data/kaggle_demand
    $ mv <your_downloaded_folder>/train.csv data/kaggle_demand
    ```

## Data Preprocess
> Extract and filter features from time series using TSFRESH and [features-selector](https://github.com/WillKoehrsen/feature-selector)

Execute this command:
```
$ python data_helper.py
```

The structure of `data/` folder will be like:

```
data/
|-- kaggle_demand
|   |-- df_rolled_12.csv
|   |-- extracted_features_12_106.csv
|   |-- labels_12_106.csv
|   |-- train.csv
|   `-- uniform_data.csv
`-- trace_201708
    |-- df_rolled_12.csv
    |-- extracted_features_12_106.csv
    |-- labels_12_106.csv
    `-- server_usage.csv
```

### Alibaba

For Alibaba Dataset, if the window size is `60`, the `df_rolled_12.csv`, `extracted_features_12_106.csv` and `labels_12_106.csv` will be generated in `data/trace_201708/`.

This process is too slow, you can download these files from [here](https://drive.google.com/file/d/1Bb-HHCLcsjgNGbd9QJo0e3QeEsWwa5eM/view?usp=sharing). And put it in `data/` folder.

### Kaggle Demand

Also, you can download files from [here](https://drive.google.com/file/d/1_v8IA_xI430-xGZEE19klSPSVk-rhQNe/view?usp=sharing) for `kaggle_demand`, and put it in `data/` folder.

Note, we transform the value of `units_sold` using `np.log1p()`. If you want get the real value of `units_sold`, you can use `np.expm()` to restore it.


## Run

If you wanna train the model from scratch, just `ams.run()` in `main.py`.

Otherwise, `ams.run(test=True)` will load the dumped model from files.

After preprocessing the data, you can change the code in `main.py` for the specific task, and run this command to get the result
```
$ python main.py
```

The dumped models will be downloaded from [alibaba-dumped](https://drive.google.com/file/d/1yDhjVhdyzH09mFpDuPgW_ndQ-Xs5k1A4/view?usp=sharing) and [kaggle-demand-dumped](https://drive.google.com/file/d/1MxCbhhp8hUJJuLDbb0Oswh0yLhow94Tv/view?usp=sharing). Just put it into `out/` folder.

The structure of `out/` folder will be like:
```
out/
|-- alibaba
|   |-- ams.pickle
|   |-- gb_model.pickle
|   |-- gpr_model.pickle
|   |-- lr_model.pickle
|   |-- svm_model.pickle
|   |-- test_method.pickle
|   `-- train_method.pickle
`-- kaggle_demand
    |-- ams.pickle
    |-- gb_model.pickle
    |-- gpr_model.pickle
    |-- lr_model.pickle
    |-- svm_model.pickle
    |-- test_method.pickle
    `-- train_method.pickle
```

## Report

### Alibaba

1. AMS Evaluation Results using RDF

| Classifier | TPR    | FPR    | TNR    | FNR    | Precision | Recall | F1-score | Accuracy |
| ---------- | ------ | ------ | ------ | ------ | --------- | ------ | -------- | -------- |
| RDF        | 0.6585 | 0.1138 | 0.8862 | 0.3415 | 0.6585    | 0.6585 | 0.6585   | 0.6585   |

2. RMSE and MAE for Different Methods

| Method                           | RMSE   | MAE    |
| -------------------------------- | ------ | ------ |
| Linear Regression                | 4.2173 | 3.2020 |
| SVM                              | 6.0078 | 4.9288 |
| GB (Gradient Boosting)           | 3.8648 | 2.9206 |
| GPR (Gaussian Process Regressor) | 4.2752 | 3.2570 |
| Proposed                         | 2.9524 | 1.8990 |

3. Time of some processes

| Process                                      | Time          |
| -------------------------------------------- | ------------- |
| Roll time series                             | 9.00s         |
| Extract features using TSFRESH               | 446s (7m 26s) |
| Filter features using FeatureSelector        | About 3h ~ 4h |
| Train Linear Regression Model                | 0.40s         |
| Train SVM Model                              | 287.44s       |
| Train GB (Gradient Boosting) Model           | 35.76s        |
| Train GPR (Gaussian Process Regressor) Model | 2.35s         |
| Train AMS using RDF                          | 89.75s        |

### Kaggle Demand

1. AMS Evaluation Results using RDF

| Classifier | TPR    | FPR    | TNR    | FNR    | Precision | Recall | F1-score | Accuracy |
| ---------- | ------ | ------ | ------ | ------ | --------- | ------ | -------- | -------- |
| RDF        | 0.3891 | 0.2036 | 0.7964 | 0.6109 | 0.3891    | 0.3891 | 0.3891   | 0.3891   |


2. RMSE and MAE for Different Methods

| Method                           | RMSE   | MAE    |
| -------------------------------- | ------ | ------ |
| Linear Regression                | 0.5503 | 0.3900 |
| SVM                              | 0.5598 | 0.4117 |
| GB (Gradient Boosting)           | 0.5440 | 0.3833 |
| GPR (Gaussian Process Regressor) | 0.5560 | 0.3948 |
| Proposed                         | 0.4431 | 0.2819 |

3. Time of some processes

| Process                                      | Time     |
| -------------------------------------------- | -------- |
| Roll time series                             | 15.55s   |
| Extract features using TSFRESH               | 1007.06s |
| Filter features using FeatureSelector        | 730.74s  |
| Train Linear Regression Model                | 0.14s    |
| Train SVM Model                              | 236.93s  |
| Train GB (Gradient Boosting) Model           | 18.45s   |
| Train GPR (Gaussian Process Regressor) Model | 1.77s    |
| Train AMS using RDF                          | 74.47s   |

## Acknowledgments

- [WillKoehrsen/features-selector](https://github.com/WillKoehrsen/feature-selector)
