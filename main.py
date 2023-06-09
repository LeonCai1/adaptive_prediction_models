import os
import pickle
import time

import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import lightgbm as lgb


def cal_mae_and_rmse(preds, targets):
    rmse = mean_squared_error(preds, targets, squared=False)
    mae = mean_absolute_error(preds, targets)
    return rmse, mae


def cal_tpr_tnr_fpr_fnr(preds, targets):
    cm = confusion_matrix(preds, targets)

    tp = np.diagonal(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    tp, fp = tp.sum(), fp.sum()
    fn, tn = fn.sum(), tn.sum()

    tpr = tp / (tp + fn)
    tnr = tn / (fp + tn)
    fpr = fp / (fp + tn)
    fnr = fn / (tp + fn)
    return tpr, tnr, fpr, fnr


def report_ams(preds, targets):
    accuracy = accuracy_score(preds, targets)
    precision = precision_score(preds, targets, average='micro')
    recall = recall_score(preds, targets, average='micro')
    f1 = f1_score(preds, targets, average='micro')

    tpr, tnr, fpr, fnr = cal_tpr_tnr_fpr_fnr(preds, targets)

    print(f"=======================")
    print(f"=== AMS Performance ===")
    print(f">>> TPR is {tpr:.4f}")
    print(f">>> FPR is {fpr:.4f}")
    print(f">>> TNR is {tnr:.4f}")
    print(f">>> FNR is {fnr:.4f}")

    print(f">>> Accuracy score is {accuracy:.4f}")
    print(f">>> Precision score is {precision:.4f}")
    print(f">>> Recall score is {recall:.4f}")
    print(f">>> F1 score is {f1:.4f}")


class AMS:
    METHOD_MAPPING = {
        0: 'lr_model',
        1: 'svm_model',
        2: 'gb_model',
        3: 'gpr_model',
        4: 'lgb_model',
    }

    def __init__(self, base_folder, window_size, out_folder='out'):
        # read data
        self.df_rolled = pd.read_csv(
            os.path.join(base_folder, f"df_rolled_{window_size}.csv")
        )

        self.df_rolled = self.df_rolled.drop(self.df_rolled.columns[0], axis= 1)

        self.df_features = pd.read_csv(
            os.path.join(base_folder, f"extracted_features_{window_size}_106.csv")
        )
        self.df_labels = pd.read_csv(
            os.path.join(base_folder, f"labels_{window_size}_106.csv")
        )

        self.df_labels = self.df_labels.drop(self.df_labels.columns[1:3], axis=1)

        self.sample_num = self.df_rolled.shape[0]

        self.out_folder = out_folder
        os.makedirs(self.out_folder, exist_ok=True)

        self.train_rolled, self.test_rolled = None, None
        self.train_features, self.test_features = None, None
        self.train_labels, self.test_labels = None, None

        # 5 models
        self.lr_model, self.lr_preds = None, None
        self.svm_model, self.svm_preds = None, None
        self.gb_model, self.gb_preds = None, None
        self.gpr_model, self.gpr_preds = None, None
        self.lgb_model, self.lgb_preds = None, None

        # identify method
        self.train_method, self.test_method = None, None

        # AMS model
        self.ams, self.pred_method = None, None

    def run(self, test=False):
        """
        train from scratch and save models if test is False.
        just load models if test is True.
        """
        warnings.filterwarnings(action='ignore', category=UserWarning)
        # split data
        self._split_data()

        load = test

        # get 4 models
        self.lr_model, self.lr_preds = self._get_lr_model(load)
        self.svm_model, self.svm_preds = self._get_svm_model(load)
        self.gb_model, self.gb_preds = self._get_gb_model(load)
        self.gpr_model, self.gpr_preds = self._get_gpr_model(load)
        self.lgb_model, self.lgb_preds = self._get_lgb_model(load)

        # identify method
        self.train_method, self.test_method = self._get_train_test_method(load)

        # get AMS model
        self.ams, self.pred_method = self._get_ams(load)

        # test AMS model
        self.test()

    def _split_data(self):
        indices = list(range(self.sample_num))
        train_indices, test_indices = train_test_split(
            indices, test_size=0.2, random_state=42
        )

        self.train_rolled = self.df_rolled.iloc[train_indices, 2:].values
        self.test_rolled = self.df_rolled.iloc[test_indices, 2:].values
        self.train_features = self.df_features.iloc[train_indices, 2:].values
        self.test_features = self.df_features.iloc[test_indices, 2:].values

        self.train_labels = self.df_labels.iloc[train_indices, 2:].values
        self.test_labels = self.df_labels.iloc[test_indices, 2:].values

        self.train_indices, self.test_indices = train_indices, test_indices
        print(f">>>> Successfully split data !")

    def _base_model_eval(self, model):
        """Evaluate basic model: lr, svm, gb, gpr"""
        preds = model.predict(self.test_rolled).reshape(-1)
        rmse, mae = cal_mae_and_rmse(preds, self.test_labels.reshape(-1))
        return preds, rmse, mae

    def _load_obj(self, obj_name):
        obj_path = os.path.join(self.out_folder, f'{obj_name}.pickle')
        obj = None
        with open(obj_path, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def _dump_obj(self, obj, obj_name):
        obj_path = os.path.join(self.out_folder, f'{obj_name}.pickle')
        with open(obj_path, 'wb') as f:
            pickle.dump(obj, f)
        print(f">>> Save {obj_name} to {obj_path}")

    def _get_lr_model(self, load=False):
        print(f"====== Linear Regression Model ======")
        st = time.time()
        # Fit Model
        if load:
            lr_model = self._load_obj('lr_model')
        else:
            lr_model = LinearRegression().fit(
                self.train_rolled, self.train_labels.reshape(-1)
            )

        # Save Model
        if not load:
            self._dump_obj(lr_model, 'lr_model')
        # Eval Model
        lr_preds, rmse, mae = self._base_model_eval(lr_model)
        print(f">>> RMSE and MAE for Linear Regression Model is {rmse:.4f}, {mae:.4f}")
        print(
            f"#### The time for training Linear Regression Model is {(time.time() - st):.2f}s"
        )
        return lr_model, lr_preds

    def _get_svm_model(self, load=False):
        print(f"====== SVM Model ======")
        st = time.time()
        # Fit Model
        if load:
            svm_model = self._load_obj('svm_model')
        else:
            svm_model = SVR(max_iter=10000).fit(
                self.train_rolled, self.train_labels.reshape(-1)
            )
        # Save Model
        if not load:
            self._dump_obj(svm_model, 'svm_model')
        # Eval Model
        svm_preds, rmse, mae = self._base_model_eval(svm_model)
        print(f">>> RMSE and MAE for SVM Model is {rmse:.4f}, {mae:.4f}")
        print(f"#### The time for training SVM Model is {(time.time() - st):.2f}s")
        return svm_model, svm_preds

    def _get_gb_model(self, load=False):
        print(f"====== Gradient Boosting Model ======")
        st = time.time()
        # Fit Model
        if load:
            gb_model = self._load_obj('gb_model')
        else:
            gb_model = GradientBoostingRegressor(random_state=42).fit(
                self.train_rolled, self.train_labels.reshape(-1)
            )
        # Save Model
        if not load:
            self._dump_obj(gb_model, 'gb_model')
        # Eval Model
        gb_preds, rmse, mae = self._base_model_eval(gb_model)
        print(
            f">>> RMSE and MAE for Gradient Boosting Regressor Model is {rmse:.4f}, {mae:.4f}"
        )
        print(
            f"#### The time for training Gradient Boosting Model is {(time.time() - st):.2f}s"
        )
        return gb_model, gb_preds

    def _get_gpr_model(self, load=False):
        print(f"====== Gaussian Process Regressor Model ======")
        st = time.time()
        # Fit Model
        if load:
            gpr_model = self._load_obj('gpr_model')
        else:
            kernel = DotProduct() + WhiteKernel()
            fit_num = 1000
            gpr_model = GaussianProcessRegressor(kernel=kernel, random_state=42).fit(
                self.train_rolled[:fit_num], self.train_labels[:fit_num].reshape(-1)
            )
        # Save Model
        if not load:
            self._dump_obj(gpr_model, 'gpr_model')
        # Eval Model
        gpr_preds, rmse, mae = self._base_model_eval(gpr_model)
        print(
            f">>> RMSE and MAE for Gaussian Process Regressor Model is {rmse:.4f}, {mae:.4f}"
        )
        print(
            f"#### The time for training Gaussian Process Regressor Model is {(time.time() - st):.2f}s"
        )
        return gpr_model, gpr_preds

    def _get_lgb_model(self, load=False):
        print(f"===== Light Gbm model")
        st = time.time()
        # Fit Model
        if load:
            lgb_model = self._load_obj('lgb_model')
        else:
            #tuned
            # lgb_model = lgb.LGBMRegressor(bagging_fraction=0.8, bagging_frequency=4, boosting_type='gbdt',
            #                         class_weight=None, colsample_bytree=1.0, feature_fraction=0.5,
            #                         importance_type='split', learning_rate=0.1, max_depth=30,
            #                         min_child_samples=20, min_child_weight=30, min_data_in_leaf=70,
            #                         min_split_gain=0.0001, n_estimators=200, n_jobs=-1,
            #                         num_leaves=1200, objective=None, random_state=None, reg_alpha=0.0,
            #                         reg_lambda=0.0, silent=True, subsample=1.0,
            #                         subsample_for_bin=200000, subsample_freq=0)
            lgb_model = lgb.LGBMRegressor(objective='regression')

            lgb_model.fit(self.train_rolled, self.train_labels.reshape(-1), verbose= -1)
        # Save Model:
        if not load:
            self._dump_obj(lgb_model, 'lgb_model')

        lgb_preds, rmse, mae = self._base_model_eval(lgb_model)
        print(
            f">>> RMSE and MAE for Light GBM is {rmse:.4f}, {mae:.4f}"
        )
        print(
            f"#### The time for training Light GBM is {(time.time() - st):.2f}s"
        )
        return lgb_model, lgb_preds

    def _get_train_test_method(self, load=False):
        print(f"====== Get Train and Test Methods ======")
        train_method, test_method = None, None
        if load:
            train_method = self._load_obj('train_method')
            test_method = self._load_obj('test_method')
        else:
            lr_p = self.lr_model.predict(self.train_rolled).reshape(-1)
            svm_p = self.svm_model.predict(self.train_rolled).reshape(-1)
            gb_p = self.gb_model.predict(self.train_rolled).reshape(-1)
            gpr_p = self.gpr_model.predict(self.train_rolled).reshape(-1)
            lgb_p = self.lgb_model.predict(self.train_rolled).reshape(-1)

            # identify method for train dataset
            stacked_p = np.stack([lr_p, svm_p, gb_p, gpr_p, lgb_p], #, lgb_p],
                                 axis=1)
            train_method = (
                np.abs(stacked_p - self.train_labels).argmin(axis=1).reshape(-1)
            )

            # identify method for test dataset
            stacked_preds = np.stack(
                [self.lr_preds, self.svm_preds, self.gb_preds, self.gpr_preds,self.lgb_preds], #self.lgb_preds],
                axis=1,
            )
            test_method = (
                np.abs(stacked_preds - self.test_labels).argmin(axis=1).reshape(-1)
            )

        # Save
        if not load:
            self._dump_obj(train_method, 'train_method')
            self._dump_obj(test_method, 'test_method')
        print(f">>>> Successfully Identify Method !")
        return train_method, test_method

    def _get_ams(self, load=False):
        print(f"====== Adaptive Model Selection Model ======")
        st = time.time()
        # Fit Model
        if load:
            ams = self._load_obj('ams')
        else:
            ams = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=20, min_samples_leaf=5,
                                         n_estimators=200)
            ams = ams.fit(self.train_features, self.train_method)
        # Save Model
        self._dump_obj(ams, 'ams')
        print(
            f"#### The time for training random Forest Classifier Model is {(time.time() - st):.2f}s"
        )
        # Eval AMS
        pred_method = ams.predict(self.test_features)
        report_ams(pred_method, self.test_method)
        return ams, pred_method

    def test(self):
        preds = []
        for idx, mid in enumerate(self.test_method):
            m = f"self.{self.METHOD_MAPPING[mid]}"
            preds.append(eval(m).predict(self.test_rolled[idx: idx + 1])[0])
        preds = np.array(preds, dtype='float64')

        rmse, mae = cal_mae_and_rmse(preds, self.test_labels.reshape(-1))
        print(f">>> RMSE and MAE for The Proposed Method is {rmse:.4f}, {mae:.4f}")


if __name__ == "__main__":
    # ====== Alibaba ======
    base_folder = 'data/trace_201708/'
    window_size = 60//5
    ams = AMS(base_folder, window_size, out_folder='out/alibaba/')

    # Train
    # ams.run()

    # Test
    # ams.run(test=True)

    # ====== Kaggle Demand ======
    # base_folder = 'data/kaggle_demand/'
    # window_size = 13
    # ams = AMS(base_folder, window_size, out_folder='out/kaggle_demand/')

    # Train
    ams.run()

    # Test
    # ams.run(test=True)

