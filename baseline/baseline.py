
from tqdm import tqdm


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder,StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd

from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    LinearRegression,
    SGDRegressor
)

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import accuracy_score,precision_score,recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from dataclasses import dataclass
from typing import Callable, List
from sklearn.base import ClassifierMixin, RegressorMixin


CLASSIFICATION_MODELS = [
    LogisticRegression(),
    SGDClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    ExtraTreesClassifier(),
    GradientBoostingClassifier(),
    HistGradientBoostingClassifier()
]

REGRESSION_MODELS = [
    LinearRegression(),
    SGDRegressor(),
    RandomForestRegressor(),
    AdaBoostRegressor(),
    ExtraTreesRegressor(),
    GradientBoostingRegressor(),
    HistGradientBoostingRegressor()
]

def get_clf_models():
    return CLASSIFICATION_MODELS

def get_reg_models():
    return REGRESSION_MODELS


@dataclass
class BaselineConfig:
    models:List
    test_size:float = 0.2
    n_max_one_hot:int = 20
    label_encoder:Callable = None
    random_state:int = None
    vebose:int = 0


class Baseline():

    def __init__(self, config:BaselineConfig):
        self.models_ = self._init_models(config)
        self.random_state_ = config.random_state
        self.test_size_ = config.test_size
        self.n_max_one_hot_ = config.n_max_one_hot

        self.cat_oh_pipe_ = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OneHotEncoder(handle_unknown='ignore', sparse=False),
        )

        self.cat_ord_pipe_ = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
        )

        self.num_pipe_ = make_pipeline(
            SimpleImputer(strategy='mean'),
            StandardScaler(),
        )

        if not isinstance(config.label_encoder, Callable):
            raise ValueError("lable encoder must be a callable")
        self.target_pipe_ = make_pipeline(
            FunctionTransformer(func=config.label_encoder)
        )

        if all([isinstance(m, ClassifierMixin) for m in config.models]):
            self.task_ = 'classification'
        elif all([isinstance(m, RegressorMixin) for m in config.models]):
            self.task_ = 'regression'
        else:
            raise ValueError("all models must be either classifiers of regressors of sci-kit learn")
            

            
    def _init_models(self, config):
        models = [m for m in config.models]
        for m in models:
            if "random_state" in m.get_params():
                m.set_params(random_state=config.random_state)
            if "verbose" in m.get_params():
                m.set_params(verbose=config.verbose)
        return models


    def _get_cat_and_num_features(self, X:pd.DataFrame):
        cat_features = X.select_dtypes(include=['object','category']).columns.to_list()
        num_features = X.select_dtypes(exclude=['object','category']).columns.to_list()
        return cat_features, num_features


    def _get_oh_ord_features(self, X:pd.DataFrame, n_max_one_hot=20):
        uniques = X.nunique()
        ord_features = uniques[uniques > n_max_one_hot].index.to_list()
        oh_features = uniques[uniques <= n_max_one_hot].index.to_list()
        return oh_features, ord_features


    def _fit_preprocess(self, X:pd.DataFrame):
        self.cat_features_, self.num_features_ = self._get_cat_and_num_features(X)
        self.oh_features_, self.ord_features_ = self._get_oh_ord_features(X[self.cat_features_], self.n_max_one_hot_)
        self.cat_oh_pipe_.fit(X[self.oh_features_])
        self.cat_ord_pipe_.fit(X[self.ord_features_])
        self.num_pipe_.fit(X[self.num_features_])


    def _transform_preprocess(self, X, y):
        y_enc = self.target_pipe_.transform(y)
        x_oh = self.cat_oh_pipe_.transform(X[self.oh_features_])
        x_ord = self.cat_ord_pipe_.transform(X[self.ord_features_])
        x_num = self.num_pipe_.transform(X[self.num_features_])
        x_full = np.concatenate([x_oh,x_ord,x_num],axis=1)
        return x_full,y_enc


    def _fit_models(self, X, y):
        for m in tqdm(self.models_, "Fitting Models..."):
            m.fit(X, y)

    def _predict_models(self, X):
        y_preds = []
        for m in self.models_:
            y_preds.append(m.predict(X))
        return y_preds
        
    def _eval_models(self, y_true, y_preds):
        if self.task_ == 'classification':
            metrics = [
                (
                    accuracy_score(y_true,yp),
                    recall_score(y_true,yp),
                    accuracy_score(y_true,yp),
                )
                for yp in y_preds
            ]
            report = pd.DataFrame(
                [[m[0],m[1],m[2]] for m in metrics],
                columns=['accuracy','precision','recall'],
                index=[m.__str__() for m in self.models_]
            )
        else:
            metrics = [
                (
                    mean_absolute_error(y_true,yp),
                    mean_squared_error(y_true,yp),
                    mean_absolute_percentage_error(y_true,yp),
                    r2_score(y_true,yp),
                )
                for yp in y_preds
            ]
            report = pd.DataFrame(
                [[m[0],m[1],m[2]] for m in metrics],
                columns=['mae','mse','mape','r2'],
                index=[m.__str__() for m in self.models_]
            )

        return report

    def __call__(self, X:pd.DataFrame, y:pd.Series):
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size_, random_state=self.random_state_
        )

        self._fit_preprocess(X_train)
        
        xtrain, ytrain = self._transform_preprocess(X_train, y_train)

        self._fit_models(xtrain, ytrain)
        
        xtest, ytest = self._transform_preprocess(X_test, y_test)

        y_preds = self._predict_models(xtest)

        return self._eval_models(ytest, y_preds)
