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

## Data Preprocess
> Extract and filter features from time series using TSFRESH and [features-selector](https://github.com/WillKoehrsen/feature-selector)

Execute this command:
```
$ python data_helper.py
```
If the window size is `60`, the `df_rolled_12.csv`, `extracted_features_12_106.csv` and `labels_12_106.csv` will be generated in `data/trace_201708/`.

This process is too slow, you can download these files from [here](https://drive.google.com/file/d/1Bb-HHCLcsjgNGbd9QJo0e3QeEsWwa5eM/view?usp=sharing). And put it in `data/` folder.

## Run

After preprocessing the data, you can run this command to get the result
```
$ python main.py
```

## Acknowledgments

- [WillKoehrsen/features-selector](https://github.com/WillKoehrsen/feature-selector)
