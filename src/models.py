"""

miqp-clf2lin
------------

Collection of functions that define the scikit-learn models to be tried for
regression, feature selection and classification.
Custom scoring functions are also specified here.

"""

from collections import OrderedDict
import logging

from sklearn.pipeline import Pipeline
from sklearn import metrics

from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import MultiTaskLassoCV

from sklearn.dummy import DummyClassifier
from sklearn.dummy import DummyRegressor

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger('learn_miqp_logger')


""" Models """


def regression_model_def(random_state):
    """
    :param random_state: int
    :return: Definition of regression models and hyper-parameters to be searched.
    """
    reg_models_dict = OrderedDict()
    reg_hyparams_dict = OrderedDict()

    reg_models_dict['DummyR'] = Pipeline([
        ('scl', StandardScaler()),
        ('reg', DummyRegressor(strategy='mean'))
    ])

    reg_models_dict['SVR'] = Pipeline([
        ('scl', StandardScaler()),
        ('reg', SVR())
    ])
    reg_hyparams_dict['SVR'] = [{
        'reg__kernel': ['rbf'],
        'reg__gamma': ['scale', 'auto', 1e-1, 1e-2, 1e-3],
        'reg__C': [0.1, 0.5, 1., 5., 10.],
        'reg__epsilon': [0.01, 0.05, 0.1, 0.5, 1]
    }]

    reg_models_dict['RTree'] = Pipeline([
        ('scl', StandardScaler()),
        ('reg', DecisionTreeRegressor(random_state=random_state))
    ])
    reg_hyparams_dict['RTree'] = [{
        'reg__criterion': ['mse'],
        'reg__splitter': ['best'],
        'reg__max_depth': [2, 3, 5, 10],
        'reg__min_samples_leaf': [1, 2, 5]
    }]

    reg_models_dict['RFR'] = Pipeline([
        ('scl', StandardScaler()),
        ('reg', RandomForestRegressor(random_state=random_state))
    ])
    reg_hyparams_dict['RFR'] = [{
        'reg__criterion': ['mse'],
        'reg__n_estimators': [10, 25, 50, 100],
        'reg__max_depth': [2, 3, 5, 10],
        'reg__min_samples_leaf': [1, 2, 5]
    }]

    return reg_models_dict, reg_hyparams_dict


def classification_model_def(random_state):
    """
    :param random_state: int
    :return: Definition of classification models and hyper-parameters to be searched.
    """
    models_dict = OrderedDict()
    hyparams_dict = OrderedDict()

    models_dict['DummyC'] = Pipeline([
        ('scl', StandardScaler()),
        ('clf', DummyClassifier(strategy='stratified', random_state=random_state))
    ])

    models_dict['LogReg'] = Pipeline([
        ('scl', StandardScaler()),
        ('clf', LogisticRegression(fit_intercept=True, random_state=random_state, multi_class='auto'))
    ])
    hyparams_dict['LogReg'] = [{
        'clf__penalty': ['l1', 'l2'],
        'clf__C': [0.1, 0.5, 1, 5, 10],
        'clf__solver': ['liblinear']
    }]

    models_dict['SVM'] = Pipeline([
        ('scl', StandardScaler()),
        ('clf', SVC(probability=False))
    ])
    hyparams_dict['SVM'] = [{
        'clf__kernel': ['rbf'],
        'clf__gamma': ['scale', 'auto', 1e-1, 1e-2, 1e-3],
        'clf__C': [0.1, 0.5, 1., 5., 10.]
    }]

    models_dict['Tree'] = Pipeline([
        ('scl', StandardScaler()),
        ('clf', DecisionTreeClassifier(random_state=random_state))
    ])
    hyparams_dict['Tree'] = [{
        'clf__criterion': ['gini', 'entropy'],
        'clf__splitter': ['best'],
        'clf__max_depth': [2, 3, 5, 10],
        'clf__min_samples_leaf': [1, 2, 5]
    }]

    models_dict['RF'] = Pipeline([
        ('scl', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=random_state))
    ])
    hyparams_dict['RF'] = [{
        'clf__criterion': ['gini', 'entropy'],
        'clf__n_estimators': [10, 25, 50, 75, 100],
        'clf__max_depth': [2, 3, 5, 10],
        'clf__min_samples_leaf': [1, 2, 5]
    }]
    return models_dict, hyparams_dict


""" Feature selectors """


def set_svc_selector(random_state, poly=False, degree=2, max_features=15):
    """
    :param random_state: int
    :param poly: bool, whether to expand the feature set polynomially
    :param degree: int, degree of polynomial expansion for the features (if poly=True)
    :param max_features: int
    :return: define and return objects for ft selection pipeline using SVC
    """
    scaler_def = StandardScaler()
    selector_def = SelectFromModel(
        estimator=SVC(random_state=random_state, kernel='linear'),
        max_features=max_features
    )
    ft_scores_attr_def = 'coef_'
    if poly:
        poly_fts_def = PolynomialFeatures(degree=degree)
        return poly_fts_def, scaler_def, selector_def, ft_scores_attr_def
    return None, scaler_def, selector_def, ft_scores_attr_def


def set_mtl_selector(random_state, poly=False, degree=2):
    """
    :param random_state: int
    :param poly: bool, whether to expand the feature set polynomially
    :param degree: int, degree of polynomial expansion for the features (if poly=True)
    :return: define and return objects for ft selection pipeline using MultiTaskLassoCV
    Note: the number of selected features is not specified.
    """
    scaler_def = StandardScaler()
    selector_def = SelectFromModel(estimator=MultiTaskLassoCV(random_state=random_state, cv=5, tol=0.0001))
    ft_scores_attr_def = 'coef_'
    if poly:
        poly_fts_def = PolynomialFeatures(degree=degree)
        return poly_fts_def, scaler_def, selector_def, ft_scores_attr_def
    return None, scaler_def, selector_def, ft_scores_attr_def


def set_rfecv_selector(random_state, n_folds=5, poly=False, degree=2):
    """
    :param random_state: int
    :param n_folds: int
    :param poly: bool, whether to expand the feature set polynomially
    :param degree: int, degree of polynomial expansion for the features (if poly=True)
    :return: define and return objects for ft selection pipeline using RFECV and RF
    """
    scaler_def = StandardScaler()
    selector_def = RFECV(
        estimator=RandomForestClassifier(random_state=random_state, n_estimators=100),
        cv=n_folds
    )
    ft_scores_attr_def = 'feature_importances_'
    if poly:
        poly_fts_def = PolynomialFeatures(degree=degree)
        return poly_fts_def, scaler_def, selector_def, ft_scores_attr_def
    return None, scaler_def, selector_def, ft_scores_attr_def


""" Custom scoring functions for model selection """


def w_accuracy(y_true, y_pred, exp_obj):
    """
    :param y_true: pd.Series
    :param y_pred: pd.Series
    :param exp_obj: train-test initialized LearningExperiment object
    :return:
    """
    ws = exp_obj.W_train
    return metrics.accuracy_score(y_true, y_pred, sample_weight=ws.loc[y_true.index.values].values.reshape(-1))


def w_loss(y_true, y_pred, exp_obj):
    """
    :param y_true: pd.Series
    :param y_pred: pd.Series
    :param exp_obj: train-test initialized LearningExperiment object
    :return:
    """
    ws_sel = exp_obj.W_train.loc[y_true.index.values]
    ws_misc = ws_sel * (y_true != y_pred)
    loss = sum(ws_misc) / sum(ws_sel)
    return loss


def wtarget_loss(y_true, y_pred, exp_obj):
    """
    :param y_true: pd.Series
    :param y_pred: pd.Series
    :param exp_obj: train-test initialized LearningExperiment object
    :return:
    """
    ws_sel = exp_obj.W_train.loc[y_true.index.values]
    ts_target = exp_obj.tt_train
    ws_misc = ws_sel * (y_true != y_pred)
    loss = sum(ws_misc) / sum(ts_target)
    return loss
