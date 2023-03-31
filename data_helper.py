import os
import time

import numpy as np
import pandas as pd
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import impute, roll_time_series

from third_party.feature_selector import FeatureSelector


def data_preprocessing_alibaba(filename, window_size, n_features, save_folder):
    """Deprecated"""
    df = pd.read_csv(filename, header=None)
    df = df[[0, 1, 2]]
    df.rename(columns={0: 'time', 1: 'machine_id', 2: 'cpu'}, inplace=True)

    df['cpu'] = df['cpu'].fillna(0)
    df.sort_values(['machine_id', 'time'], inplace=True)

    # rolling time series
    df_rolled = roll_time_series(
        df,
        column_id='machine_id',
        column_sort='time',
        max_timeshift=window_size - 1,
        min_timeshift=window_size - 1,
    )
    print(f">>> Successfully rolling time series")

    # extract features
    extracted_features = extract_features(
        df_rolled.drop('machine_id', axis=1),
        column_id='id',
        column_sort='time',
        column_value='cpu',
        impute_function=impute,
        default_fc_parameters=EfficientFCParameters(),
    )
    print(
        f">>> Successfully extract features using tsfresh, the number of features is {extracted_features.shape[1]}"
    )

    # get labels
    y = df.groupby('machine_id').apply(
        lambda x: x.set_index('time')['cpu'].shift(-1)[:-1]
    )

    y = y[y.index.isin(extracted_features.index)]
    extracted_features = extracted_features[extracted_features.index.isin(y.index)]

    # save rolled data
    time_series_list = []
    for i, d in df_rolled.groupby('id'):
        time_series_list.append(d['cpu'].values.tolist() + list(eval(str(i))))
    df_rolled = pd.DataFrame(time_series_list)
    cols = list(df_rolled.columns)
    cols[-2] = 'machine_id'
    cols[-1] = 'time'
    df_rolled.columns = cols
    df_rolled = df_rolled.set_index(['machine_id', 'time'])
    df_rolled = df_rolled[df_rolled.index.isin(y.index)]
    df_rolled_filename = os.path.join(save_folder, f"df_rolled_{window_size}.csv")
    df_rolled.to_csv(df_rolled_filename)
    print(f">>> Save df_rolled in {df_rolled_filename}, length is {df_rolled.shape[0]}")

    # reduce features
    fs = FeatureSelector(data=extracted_features, labels=y)
    fs.identify_single_unique()
    fs.identify_collinear(correlation_threshold=0.98)
    fs.identify_zero_importance(
        task='regression', eval_metric='l2', n_iterations=10, early_stopping=True
    )
    extracted_features = fs.remove(
        methods=['single_unique', 'collinear', 'zero_importance']
    )
    extracted_features = impute(extracted_features)

    print(
        f">>> After FeatureSelector, the number of features is {extracted_features.shape[1]}"
    )

    # select features with tsfresh
    extracted_features = select_features(extracted_features, y)
    print(
        f">>> After select_features, the number of features is {extracted_features.shape[1]}"
    )

    #
    relevance_table = calculate_relevance_table(extracted_features, y)
    relevance_table = relevance_table[relevance_table.relevant]
    relevance_table.sort_values("p_value", inplace=True)
    relevance_table.reset_index(drop=True, inplace=True)
    extracted_features = extracted_features[relevance_table.iloc[:n_features, 0].values]

    extracted_features.reset_index(inplace=True)
    extracted_features.rename(
        columns={
            'level_0': 'machine_id',
            'level_1': 'time',
        },
        inplace=True,
    )

    features_filename = os.path.join(
        save_folder, f"extracted_features_{window_size}_{n_features}.csv"
    )
    extracted_features.to_csv(features_filename, index=None)
    print(
        f">>> Save features in {features_filename}, length is {extracted_features.shape[0]}"
    )
    labels_filename = os.path.join(
        save_folder, f"labels_{window_size}_{n_features}.csv"
    )
    y.to_csv(labels_filename)
    print(f">>> Save features in {labels_filename}, length is {y.shape[0]}")


def uniform_kaggle_demand(filename, save_folder):
    """Convert raw data to uniform format"""
    df = pd.read_csv(filename)
    df = df.sort_values(['store_id', 'sku_id', 'week'])
    df['week'] = pd.to_datetime(df['week'])
    df['store_sku'] = df['store_id'].astype('str') + "_" + df['sku_id'].astype('str')
    # use np.expm() to resotre it.
    df['units_sold'] = np.log1p(df['units_sold'])

    df = df.groupby(['store_sku', 'week'])['units_sold'].sum().to_frame()
    df.reset_index(inplace=True)
    df.rename(
        columns={'store_sku': 'entry_id', 'week': 'time', 'units_sold': 'value'},
        inplace=True,
    )

    df.to_csv(os.path.join(save_folder, 'uniform_data.csv'), index=None)


class DataHelper:
    def __init__(
        self, filename, window_size, n_features=106, n_jobs=0, save_folder=None
    ):
        self.df = pd.read_csv(filename)
        self.window_size = window_size

        self.save_folder = save_folder
        if self.save_folder is None:
            self.save_folder = os.path.dirname(filename)

        self.n_features = n_features

        self.df_rolled = None
        self.labels = None
        self.extracted_features = None

        self.n_jobs = self.get_n_jobs(n_jobs)
        print(f">>> The number of jobs is {self.n_jobs}")

    def get_n_jobs(self, n_jobs):
        from multiprocessing import cpu_count

        default_n_jobs = int(cpu_count()) // 2

        n_jobs = n_jobs or default_n_jobs
        n_jobs = min(n_jobs, default_n_jobs)
        return max(1, n_jobs)

    def run(self):
        self._make_time_series_data()
        self._extract_features()

        # intersection of labels and extracted_features
        self.labels = self.labels[self.labels.index.isin(self.extracted_features.index)]
        self.extracted_features = self.extracted_features[
            self.extracted_features.index.isin(self.labels.index)
        ]

        self._filter_features_stage1()
        self._filter_features_stage2()
        self._filter_features_stage3()

        self._make_rolled_data()

        self._save_data()

    def _make_time_series_data(self):
        """Get time series data according to window size.

        The time_series data will be a DataFrame with 4 columns, like:
            entry_id    time        value       id
        0     1      2011-01-08       2     (1, 2011-01-20)
        1     1      2011-01-09       3     (1, 2011-01-20)
                        ...
        11    1      2011-01-20       5     (1, 2011-01-20)

        The labels will be a DataFrame with 1 columns, like:
            entry_id    time      value
        0       1     2011-01-08    3
                    ...
        """
        st = time.time()
        self.df_rolled = roll_time_series(
            self.df,
            column_id='entry_id',
            column_sort='time',
            max_timeshift=self.window_size - 1,
            min_timeshift=self.window_size - 1,
        )

        self.labels = self.df.groupby('entry_id').apply(
            lambda x: x.set_index('time')['value'].shift(-1)[:-1].to_frame()
        )
        print(
            f">>> The time of _make_time_series_data function is {(time.time() - st):.2f}s."
        )

    def _extract_features(self):
        st = time.time()
        self.extracted_features = extract_features(
            self.df_rolled.drop('entry_id', axis=1),
            column_id='id',
            column_sort='time',
            column_value='value',
            impute_function=impute,
            default_fc_parameters=EfficientFCParameters(),
            n_jobs=self.n_jobs,
        )
        print(
            f">>> The time of _extract_features function is {(time.time() - st):.2f}s."
        )

    def _filter_features_stage1(self):
        st = time.time()
        print("====== Filter Features Stage #1 ======")

        fs = FeatureSelector(
            data=self.extracted_features, labels=self.labels, n_jobs=self.n_jobs
        )
        fs.identify_single_unique()
        fs.identify_collinear(correlation_threshold=0.98)
        fs.identify_zero_importance(
            task='regression', eval_metric='l2', n_iterations=10, early_stopping=True
        )

        self.extracted_features = fs.remove(
            methods=['single_unique', 'collinear', 'zero_importance']
        )
        self.extracted_features = impute(self.extracted_features)

        print(
            f">>> After FeatureSelector, the number of features is {self.extracted_features.shape[1]}"
        )
        print(
            f">>> The time of _filter_features_stage1 function is {(time.time() - st):.2f}s."
        )

    def _filter_features_stage2(self):
        st = time.time()
        print("====== Filter Features Stage #2 ======")

        self.extracted_features = select_features(
            self.extracted_features, self.labels['value']
        )

        print(
            f">>> After select_features function, the number of features is {self.extracted_features.shape[1]}"
        )
        print(
            f">>> The time of _filter_features_stage2 function is {(time.time() - st):.2f}s."
        )

    def _filter_features_stage3(self):
        st = time.time()
        print("====== Filter Features Stage #3 ======")

        relevance_table = calculate_relevance_table(
            self.extracted_features, self.labels['value']
        )
        relevance_table = relevance_table[relevance_table.relevant]
        relevance_table.sort_values("p_value", inplace=True)
        relevance_table.reset_index(drop=True, inplace=True)
        self.extracted_features = self.extracted_features[
            relevance_table.iloc[: self.n_features, 0].values
        ]

        print(
            f">>> The time of _filter_features_stage3 function is {(time.time() - st):.2f}s."
        )

    def _make_rolled_data(self):
        """Convert df_roll format

        The result will be a DataFrame, like:
            entry_id    time       0 1 2 3 ... 11
        0       1     2011-01-20    ...
        """
        st = time.time()
        time_series_list = []

        # i is a tuple: (<entry_id>, <time>)
        for i, d in self.df_rolled.groupby('id'):
            time_series_list.append(d['value'].values.tolist() + list(eval(str(i))))
        self.df_rolled = pd.DataFrame(time_series_list)
        cols = list(self.df_rolled.columns)
        cols[-2] = 'entry_id'
        cols[-1] = 'time'
        self.df_rolled.columns = cols
        self.df_rolled = self.df_rolled.set_index(['entry_id', 'time'])

        # intersection of df_rolled and labels
        self.df_rolled = self.df_rolled[self.df_rolled.index.isin(self.labels.index)]
        print(
            f">>> The time of _make_rolled_data function is {(time.time() - st):.2f}s."
        )

    def _save_data(self):
        df_rolled_filename = os.path.join(
            self.save_folder, f"df_rolled_{self.window_size}.csv"
        )
        self.df_rolled.to_csv(df_rolled_filename)
        print(
            f">>> Save df_rolled in {df_rolled_filename}, length is {self.df_rolled.shape[0]}"
        )

        features_filename = os.path.join(
            self.save_folder,
            f"extracted_features_{self.window_size}_{self.n_features}.csv",
        )
        self.extracted_features.to_csv(features_filename, index=None)
        print(
            f">>> Save features in {features_filename}, length is {self.extracted_features.shape[0]}"
        )

        labels_filename = os.path.join(
            save_folder, f"labels_{self.window_size}_{self.n_features}.csv"
        )
        self.labels.to_csv(labels_filename)
        print(f">>> Save labels in {labels_filename}, length is {self.labels.shape[0]}")


if __name__ == '__main__':
    # save_folder = 'data/trace_201708'
    # filename = os.path.join(save_folder, 'server_usage.csv')
    # data_preprocessing_alibaba(filename, 60//5, 106, save_folder)

    save_folder = 'data/kaggle_demand'
    filename = os.path.join(save_folder, 'train.csv')
    # uniform_kaggle_demand(filename, save_folder)

    data_helper = DataHelper(
        filename=os.path.join(save_folder, 'uniform_data.csv'),
        window_size=12,
        n_jobs=12,
    )
    data_helper.run()
