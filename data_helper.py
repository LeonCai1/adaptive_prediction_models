import os
import time

import numpy as np
import pandas as pd
from tsfel import time_series_features_extractor
from tsfel import get_features_by_domain


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


def data_preprocessing_alibaba(filename, save_folder):
    """Deprecated"""
    df = pd.read_csv(filename, header=None)
    df = df[[1, 0, 2]]
    df.rename(columns={0: 'time', 1: 'entry_id', 2: 'value'}, inplace=True)

    df['value'] = df['value'].fillna(0)
    df.sort_values(['entry_id', 'time'], inplace=True)
    df.to_csv(os.path.join(save_folder, 'uniform_data_alibaba.csv'), index=None)


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
        # self._make_time_series_data()
        self._extract_features()
        self._get_labels(1)

        # self._make_rolled_data()
        self._get_rolled()

        self._save_data()

    def _extract_features(self):
        st = time.time()
        cfg = get_features_by_domain()
        self.df_rolled = []
        for group_name, group_df in self.df.groupby('entry_id'):
            if self.extracted_features is None:
                temp = time_series_features_extractor(cfg, group_df['value'], window_size=self.window_size, verbose=1,
                                                      n_jobs=-1).iloc[:-1, :]
                self.extracted_features = pd.DataFrame(temp)
                for name, data in group_df.groupby(group_df.index // self.window_size):
                    if len(data) != self.window_size:
                        continue
                    self.df_rolled.append(data['value'].values)
                self.df_rolled = self.df_rolled[:-1]
                # pd.DataFrame(group_df.groupby(group_df.index // self.window_size).filter(lambda x: len(x) == self.window_size)).
                # self.df_rolled = pd.DataFrame([rolled.value[0::self.window_size]])
            else:
                temp = time_series_features_extractor(cfg, group_df['value'], window_size=self.window_size, verbose=1,
                                                      n_jobs=-1).iloc[:-1, :]
                self.extracted_features = pd.concat([self.extracted_features, pd.DataFrame(temp)], ignore_index=True)
                #
                for i in range(group_df.shape[0] // self.window_size):
                    start_index = i * self.window_size
                    data = group_df[start_index:start_index + self.window_size]
                    if start_index + self.window_size > group_df.shape[0]:
                        break
                    self.df_rolled.append(data['value'].values)
                self.df_rolled = self.df_rolled[:-1]
                # self.df_rolled = pd.concat([self.df_rolled,pd.DataFrame(group_df.groupby(group_df.index // self.window_size).filter(
                #     lambda x: len(x) == self.window_size)).transpose()])
                # self.df_rolled = pd.concat([self.df_rolled, pd.DataFrame([rolled.value[0::self.window_size]])],
                # ignore_index=True)
        # self.extracted_features = test.index.apply(
        #     lambda x: time_series_features_extractor(cfg, x['value'], window_size=self.window_size, verbose=1,
        #                                              n_jobs=-1))
        print(
            f">>> The time of _extract_features function is {(time.time() - st):.2f}s."
        )

    def _get_labels(self, n):
        self.labels = self.df.groupby('entry_id').apply(
            lambda x: x.iloc[self.window_size::self.window_size] if x.shape[0] % self.window_size == 0 else
            x.iloc[self.window_size: self.window_size * (x.shape[0] // self.window_size):self.window_size]
        )

    def _get_rolled(self):
        self.df_rolled = pd.concat(
            [self.labels.loc[:, ['time']].reset_index(), pd.DataFrame(self.df_rolled).reset_index()], axis=1).drop(
            ['index', 'level_1'], axis=1)
        # remove last partition if it has fewer than 60 elements
        # self.labels = self.labels.groupby('entry_id').apply(lambda x: x[:-1] if len(x) % 12 != 0 else x)

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
    save_folder = 'data/trace_201708'
    filename = os.path.join(save_folder, 'server_usage.csv')
    data_preprocessing_alibaba(filename, save_folder)

    # save_folder = 'data/kaggle_demand'
    # filename = os.path.join(save_folder, 'train.csv')
    # uniform_kaggle_demand(filename, save_folder)

    data_helper = DataHelper(
        filename=os.path.join(save_folder, 'uniform_data_alibaba.csv'),
        window_size=12,
        n_jobs=12,
    )
    data_helper.run()
