import pandas as pd
from tsfresh.utilities.dataframe_functions import roll_time_series, impute
from tsfresh import extract_features, select_features
from tsfresh.feature_selection.relevance import calculate_relevance_table
    
from third_party.feature_selector import FeatureSelector

window_size = 60
n_features = 106

def data_preprocessing_alibaba(filename):
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
    )
    print(f">>> Successfully extract features using tsfresh, the number of features is {extracted_features.shape[1]}")

    # get labels
    y = df.groupby('machine_id').apply(lambda x: x.set_index('time')['cpu'].shift(-1)[:-1])

    y = y[y.index.isin(extracted_features.index)]
    extracted_features = extracted_features[extracted_features.index.isin(y.index)]

    # reduce features
    fs = FeatureSelector(data=extracted_features, labels=y)
    fs.identify_single_unique()
    fs.identify_collinear(correlation_threshold=0.98)
    fs.identify_zero_importance(task='regression', eval_metric='l2', n_iterations=10, early_stopping=True)
    extracted_features = fs.remove(methods=['single_unique', 'collinear', 'zero_importance'])

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
    y.reset_index(inplace=True)

    extracted_features.to_csv('extracted_features.csv')
    y.to_csv('label.csv')
    

if __name__ == '__main__':
    data_preprocessing_alibaba(filename='data/trace_201708/server_usage.csv')