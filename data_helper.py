import os
import pandas as pd
from tsfresh.utilities.dataframe_functions import roll_time_series, impute
from tsfresh import extract_features, select_features
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.feature_extraction import EfficientFCParameters 
    
from third_party.feature_selector import FeatureSelector


def data_preprocessing_alibaba(filename, window_size, n_features, save_folder):
    df = pd.read_csv(filename, header=None)
    df = df[[0, 1, 2]]
    df.rename(columns={
        0: 'time',
        1: 'machine_id',
        2: 'cpu'
    }, inplace=True)

    df['cpu'] = df['cpu'].fillna(0)
    df.sort_values(['machine_id', 'time'], inplace=True)

    # rolling time series
    df_rolled = roll_time_series(
        df,
        column_id='machine_id',
        column_sort='time',
        max_timeshift=window_size-1,
        min_timeshift=window_size-1,
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
    print(f">>> Successfully extract features using tsfresh, the number of features is {extracted_features.shape[1]}")

    # get labels
    y = df.groupby('machine_id').apply(lambda x: x.set_index('time')['cpu'].shift(-1)[:-1])

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
    fs.identify_zero_importance(task='regression', eval_metric='l2', n_iterations=10, early_stopping=True)
    extracted_features = fs.remove(methods=['single_unique', 'collinear', 'zero_importance'])
    extracted_features = impute(extracted_features)

    print(f">>> After FeatureSelector, the number of features is {extracted_features.shape[1]}")
    
    # select features with tsfresh
    extracted_features = select_features(extracted_features, y)
    print(f">>> After select_features, the number of features is {extracted_features.shape[1]}")

    #
    relevance_table = calculate_relevance_table(extracted_features, y)
    relevance_table = relevance_table[relevance_table.relevant]
    relevance_table.sort_values("p_value", inplace=True)
    relevance_table.reset_index(drop=True, inplace=True)
    extracted_features = extracted_features[relevance_table.iloc[:n_features, 0].values]
    
    extracted_features.reset_index(inplace=True)
    extracted_features.rename(columns={
        'level_0': 'machine_id',
        'level_1': 'time',
    }, inplace=True)

    features_filename = os.path.join(save_folder, f"extracted_features_{window_size}_{n_features}.csv")
    extracted_features.to_csv(features_filename, index=None)
    print(f">>> Save features in {features_filename}, length is {extracted_features.shape[0]}")
    labels_filename = os.path.join(save_folder, f"labels_{window_size}_{n_features}.csv")
    y.to_csv(labels_filename)
    print(f">>> Save features in {labels_filename}, length is {y.shape[0]}")

 
if __name__ == '__main__':
    save_folder = 'data/trace_201708'
    filename = os.path.join(save_folder, 'server_usage.csv')
    data_preprocessing_alibaba(filename, 60//5, 106, save_folder)
