"""

miqp-clf2lin
------------

Definition of LearningExperiment, a customized class to manage learning experiments with scikit-learn.

"""

import numpy as np
import pandas as pd
from collections import OrderedDict
from itertools import compress
from scipy.stats import stats
import logging

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn import metrics

import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

logger = logging.getLogger('learn_miqp_logger')


""" Static methods """


def shifted_gmean(col, eps=1):
    """
    :param col: column of Pandas DataFrame
    :param eps: float, shift in the mean
    :return:
    """
    return stats.gmean(col.astype(dtype=np.float64) + eps) - eps


""" LearningExperiment Class """


class LearningExperiment:
    """
    A customized class to manage learning experiments with scikit-learn.

    """
    def __init__(
            self, setting, df,
            clf_fts, clf_target, clf_target_names,
            random_state, multi_class,
            reg=False, reg_fts=None, reg_target=None,
            mtl_targets=None,
            description=None, verbose=True
    ):
        """
        :param setting: str, describing data setting {'AllMulti', 'BinNoTie', 'BinLabel', 'BinLabelReg'}
        :param df: DataFrame, containing selected data for the experiment
        :param clf_fts: list of str, columns of df to be used as features
        :param clf_target: str, column of df to be used as label
        :param clf_target_names: list of str, containing names of classes
        :param random_state: int, random state for all scikit routines
        :param multi_class: bool, whether the classification will be multi-class
        :param reg: bool, whether regression will be performed to predict additional ft (require special split)
        :param reg_fts: list of str, columns of df to be used as features for regression
        :param reg_target: str, the name of the column of df to be used as regression target
        :param mtl_targets: list of str, names of the columns to be used as targets in MultiTaskLasso
        :param description: str, description of the experiment, optional
        :param verbose: bool, control verbosity of output
        """
        self.df = df
        self.clf_target = clf_target
        self.clf_fts = clf_fts  # Note: reassigned (but *not* mutated) in train test split creation!
        self.reg_target = reg_target
        self.reg_fts = reg_fts
        if reg:
            assert self.reg_target
            assert self.reg_fts
        self.mtl_targets = mtl_targets
        self.__description = description

        self.params = {
            'setting': setting,
            'clf_target_names': clf_target_names,
            'random_state': random_state,
            'multi_class': multi_class,
            'reg': reg,
            'verbose': verbose,
            'num_folds': None,
            'val_scoring': None,
            'average_type': '',
        }

        # initialize reg/classification and train/test splits
        self.X_reg = None
        self.X_clf = None
        self.y_reg = None
        self.y_clf = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # data structures to compute and save results
        self.test_labeling_df = None
        self.results = OrderedDict()
        self.opt_scores = {}

        # feature selection objects
        self.red_X_train = None  # 'reduced' training and test sets, w/ selected fts only
        self.red_X_test = None
        self.y_mtl_train = None  # train and test targets for MultiTaskLasso selector
        self.y_mtl_test = None

        # weights and clf_target time setting
        self.W_train = pd.Series([])
        self.W_test = pd.Series([])
        self.tt_train = pd.Series([])  # 'target times' for train and test
        self.tt_test = pd.Series([])

    def __repr__(self):
        return 'LearningExperiment description:\n{}'.format(self.__description)

    @property
    def description(self):
        return self.__description

    @description.setter
    def description(self, description):
        self.__description = description

    def set_regression_split(self, reg_test_size=0.70):
        """
        :param reg_test_size: float in (0, 1), size proportion of split to perform *regression test* (i.e., total clf)
        :return:
        Regression/classification split is stored in X_reg, X_clf, y_reg, y_clf attributes.
        Non-features columns (Weight included) should be dropped from X_reg and X_clf.
        Note that X_clf will be used to get regression predictions;
        its indexing is also used to select the proper subset of self.df for performing clf.
        """
        if not self.params['reg']:
            raise ValueError("Regression was not initially specified!")
        self.X_reg, self.X_clf, self.y_reg, self.y_clf = train_test_split(
            self.df[self.reg_fts],
            self.df[self.reg_target],
            test_size=reg_test_size,
            random_state=self.params['random_state'],
            shuffle=True,
            stratify=self.df[self.clf_target]  # use the classification target to stratify the regression split
        )
        self.y_reg = self.y_reg.astype('float64')
        self.y_clf = self.y_clf.astype('float64')

        assert isinstance(self.X_reg, pd.DataFrame)
        # remove all non-features columns from X_reg and X_clf and self.reg_fts
        cols = ['Name', 'Origin', 'Curvature', 'Weight', 'MultiLabel', 'BinLabel', 'Time_L', 'Time_NL']
        self.X_reg.drop(cols, axis=1, inplace=True)
        self.X_clf.drop(cols, axis=1, inplace=True)
        self.reg_fts = [ft for ft in self.reg_fts if ft not in cols]

        if self.params['verbose']:
            logging.info("\n------------------------------------------------------------")
            logging.info("Regression-classification split created. Shapes: {} - {}".format(
                self.X_reg.shape, self.X_clf.shape
            ))
        return

    def set_train_test_split(self, test_size=0.25, reg_pred_col_name=None):
        """
        :param test_size: float in (0, 1), size proportion of test set for classification
        :param reg_pred_col_name: str, name of the column containing regression *predictions* to be used in clf training
        :return:
        Train test split for *classification* is created, and with the determined split Weight stored in corresponding
        class attributes.
        If regression was performed, idx are selected using self.X_clf and the new predicted column is added.
        Non-features columns are used to define test_labeling_df DataFrame (on the test set instances only),
        which will contain clf predictions and be used to compute optimization measures.
        Optimization measures for default and target are computed using get_t_clf and compute_opt_measures.
        Non-features columns (Weight included) are then dropped from X_train and X_test, and from self.clf_fts as well.
        """
        if self.params['reg'] and not reg_pred_col_name:
            raise ValueError("Specify name of columns predicted with regression, to be added for classification!")

        if self.params['reg']:
            # identify data for clf and add new col to features for clf
            clf_df = self.df.loc[self.X_clf.index]
            clf_df[reg_pred_col_name] = self.results['regression']['pred_df'][reg_pred_col_name]
            logging.info("\nPrediction from regressor {} added to fts for classification.".format(reg_pred_col_name))
            self.clf_fts = self.clf_fts + [reg_pred_col_name]  # append method would mutate the object
            t_target = self.get_t_target(y_target_name=self.clf_target, df_in=clf_df)
            # perform split
            self.X_train, self.X_test, self.y_train, self.y_test, self.W_train, self.W_test, self.tt_train, self.tt_test = train_test_split(
                clf_df[self.clf_fts],
                clf_df[self.clf_target],
                clf_df['Weight'],
                t_target,
                test_size=test_size,
                random_state=self.params['random_state'],
                shuffle=True,
                stratify=clf_df[self.clf_target]
            )
        else:
            t_target = self.get_t_target(y_target_name=self.clf_target, df_in=self.df)
            # perform split on original data
            self.X_train, self.X_test, self.y_train, self.y_test, self.W_train, self.W_test, self.tt_train, self.tt_test = train_test_split(
                self.df[self.clf_fts],
                self.df[self.clf_target],
                self.df['Weight'],
                t_target,
                test_size=test_size,
                random_state=self.params['random_state'],
                shuffle=True,
                stratify=self.df[self.clf_target]
            )

        self.y_train = self.y_train.astype('float64')
        self.y_test = self.y_test.astype('float64')

        assert isinstance(self.X_test, pd.DataFrame)
        cols_labeling = ['Name', 'Curvature', 'Origin', 'Weight', 'MultiLabel', 'BinLabel', 'Time_L', 'Time_NL']

        if not all(c in self.X_test.columns for c in cols_labeling):
            raise ValueError('Some columns were not found in DataFrame. Specify {}'.format(cols_labeling))
        self.test_labeling_df = self.X_test[cols_labeling].copy()

        # get optimization measures for L (default) and target
        self.opt_scores['def_t_sum'], self.opt_scores['def_t_sgmean'] = self.compute_opt_measures('Time_L')
        self.get_t_clf(y_col_name=self.clf_target, time_col_name='t_{}'.format(self.clf_target))
        self.opt_scores['target_t_sum'], self.opt_scores['target_t_sgmean'] = self.compute_opt_measures('t_{}'.format(self.clf_target))

        # remove all non-features columns
        self.X_train.drop(cols_labeling, axis=1, inplace=True)
        self.X_test.drop(cols_labeling, axis=1, inplace=True)
        self.clf_fts = [ft for ft in self.clf_fts if ft not in cols_labeling]
        assert len(self.clf_fts) == self.X_train.shape[1]
        assert self.X_train.shape[1] == self.X_test.shape[1]

        if self.params['verbose']:
            logging.info("\n------------------------------------------------------------")
            logging.info("Classification train-test split created. Shapes: {} - {}".format(self.X_train.shape, self.X_test.shape))
            logging.info("Weights created. Shapes: {} - {}".format(self.W_train.shape, self.W_test.shape))
            logging.info("Test labeling DF created. Shape: {}. Columns: {}".format(self.test_labeling_df.shape,
                                                                                   self.test_labeling_df.columns))
        return

    def regression_model_selection(self, model_select_pipe, hyparams, num_folds=5):
        """
        :param model_select_pipe:  OrderedDict of Pipeline objects, keys should be regression models' names
        :param hyparams: OrderedDict, containing hyper-parameters for grid-search, keys should be regr. models' names
        :param num_folds: int, number of cross-validation folds
        :return:
        Performs model selection for regression on pipelines in model_select_pipe, using in
        GridSearchCV hyper-parameters specified in hyparams. All results are stored in self.results['regression'].
        Note: no special scoring function is specified (we use the standard ones for regression).
        """
        self.results['regression'] = {
            'hyparams': hyparams,
            'pipe': model_select_pipe,
            'cv_results': OrderedDict(),
            'scores': OrderedDict(),
            'best_estimators': OrderedDict(),
            'pred_df': pd.DataFrame(index=self.X_clf.index)
        }

        if self.params['verbose']:
            logging.info("\n------------------------------------------------------------")
            logging.info("Regression model selection...")

        for model_name in self.results['regression']['pipe'].keys():
            if self.params['verbose']:
                logging.info('\nModel: {}'.format(model_name))

            if model_name == 'DummyR':
                dummy = self.results['regression']['pipe'][model_name]
                # fit
                dummy.fit(self.X_reg, self.y_reg)
                self.results['regression']['cv_results'][model_name] = None
                # predict
                y_pred = dummy.predict(self.X_clf)

            else:  # other models
                # instantiate GridSearchCV
                gscv = GridSearchCV(
                    estimator=self.results['regression']['pipe'][model_name],
                    param_grid=self.results['regression']['hyparams'][model_name],
                    scoring=None,
                    iid=True,
                    cv=num_folds,
                    return_train_score=True
                )
                # fit
                gscv.fit(self.X_reg, self.y_reg)
                self.results['regression']['cv_results'][model_name] = gscv.cv_results_
                if self.params['verbose']:
                    logging.info('Best params: {}'.format(gscv.best_params_))
                    logging.info('Best training score: {:.3f}'.format(gscv.best_score_))
                    means = gscv.cv_results_['mean_test_score']
                    stds = gscv.cv_results_['std_test_score']
                    for mean, std, params in zip(means, stds, gscv.cv_results_['params']):
                        logging.debug('{:.3f} (+/- {:.3f}) for {}'.format(mean, std, params))  # or std*2?

                # predict on test (i.e., clf set) with identified best parameters and get scores
                y_pred = gscv.predict(self.X_clf)

            if self.params['verbose']:
                logging.info('Test MSE: {}'.format(metrics.mean_squared_error(y_true=self.y_clf, y_pred=y_pred)))
                logging.info('Test MAE: {}'.format(metrics.mean_absolute_error(y_true=self.y_clf, y_pred=y_pred)))

            # save scores, predictions and best models
            self.results['regression']['scores'][model_name] = [
                metrics.mean_squared_error(y_true=self.y_clf, y_pred=y_pred),
                metrics.mean_absolute_error(y_true=self.y_clf, y_pred=y_pred)
            ]
            self.results['regression']['pred_df'][model_name] = pd.Series(y_pred, index=self.X_clf.index)
            if model_name == 'DummyR':
                self.results['regression']['best_estimators'][model_name] = dummy
            else:
                self.results['regression']['best_estimators'][model_name] = gscv.best_estimator_

        return

    def classification_model_selection(self, model_select_pipe, hyparams, val_scoring, val_scoring_name,
                                       num_folds=5, average_type='weighted', use_sample_weight=False, reduced=False):
        """
        :param model_select_pipe: OrderedDict of Pipeline objects, keys should be clf models' names
        :param hyparams: OrderedDict, containing hyper-parameters for grid-search, keys should be clf models' names
        :param val_scoring: str or custom callable, scoring function used in cross-validation
        :param val_scoring_name: str, name of the scoring function used in cross-validation (e.g., 'accuracy')
        :param num_folds: int, number of cross-validation folds
        :param average_type: str, type of average applied to results
        :param use_sample_weight: bool, whether to use sample_weight in fit
        :param reduced: bool, whether to perform clf model selection on reduced dataset (i.e., after ft selection)
        :return:
        Perform classification model selection on pipelines in model_select_pipe, GridSearchCV hyper-parameters
        specified in hyparams. All results are stored in self.results['model_sel'].
        Note: validation scoring cannot (always) be used on the test set.
        """
        self.params['num_folds'] = num_folds
        self.params['val_scoring'] = val_scoring
        self.params['average_type'] = average_type

        self.results['model_sel'] = {
            'use_sample_weight': use_sample_weight,
            'hyparams': hyparams,
            'pipe': model_select_pipe,
            'cv_results': OrderedDict(),
            'clf_reports': OrderedDict(),
            'conf_matrices': OrderedDict(),
            'scores': OrderedDict(),
            'best_estimators': OrderedDict()
        }

        if reduced and 'ft_sel' not in self.results:
            raise ValueError("'Reduced' mode specified but 'ft_sel' step was not performed yet.")

        if self.params['verbose']:
            logging.info("\n------------------------------------------------------------")
            logging.info("Model selection...")
            logging.info("Validation uses scoring {}".format(val_scoring_name))

        for model_name in self.results['model_sel']['pipe'].keys():
            if self.params['verbose']:
                logging.info('\nModel: {}'.format(model_name))

            if model_name == 'DummyC':
                dummy = self.results['model_sel']['pipe'][model_name]
                # fit
                if reduced:
                    dummy.fit(self.red_X_train, self.y_train)
                    if self.params['verbose']:
                        logging.info("Fitting on reduced data: {}".format(self.red_X_train.shape))
                else:
                    dummy.fit(self.X_train, self.y_train)
                    if self.params['verbose']:
                        logging.info("Fitting on original data: {}".format(self.X_train.shape))
                self.results['model_sel']['cv_results'][model_name] = None

                # predict on test and get scores
                if reduced:
                    y_pred = dummy.predict(self.red_X_test)
                    pd.testing.assert_index_equal(self.test_labeling_df.index, self.red_X_test.index)
                else:
                    y_pred = dummy.predict(self.X_test)
                    pd.testing.assert_index_equal(self.test_labeling_df.index, self.X_test.index)

            else:  # other models
                # instantiate GridSearchCV
                gscv = GridSearchCV(
                    estimator=self.results['model_sel']['pipe'][model_name],
                    param_grid=self.results['model_sel']['hyparams'][model_name],
                    scoring=self.params['val_scoring'],
                    iid=True,
                    cv=self.params['num_folds'],
                    return_train_score=True
                )
                # fit
                if reduced:
                    if use_sample_weight:
                        gscv.fit(self.red_X_train, self.y_train, clf__sample_weight=np.array(self.W_train))
                        if self.params['verbose']:
                            logging.info("Fitting on reduced data using sampleW: {}".format(self.red_X_train.shape))
                    else:
                        gscv.fit(self.red_X_train, self.y_train)
                        if self.params['verbose']:
                            logging.info("Fitting on reduced data: {}".format(self.red_X_train.shape))
                else:
                    if use_sample_weight:
                        gscv.fit(self.X_train, self.y_train, clf__sample_weight=np.array(self.W_train))
                        if self.params['verbose']:
                            logging.info("Fitting on original data using sampleW: {}".format(self.X_train.shape))
                    else:
                        gscv.fit(self.X_train, self.y_train)
                        if self.params['verbose']:
                            logging.info("Fitting on original data: {}".format(self.X_train.shape))
                self.results['model_sel']['cv_results'][model_name] = gscv.cv_results_
                if self.params['verbose']:
                    logging.info('Best params: {}'.format(gscv.best_params_))
                    logging.info('Best training score: {:.3f}'.format(gscv.best_score_))
                    means = gscv.cv_results_['mean_test_score']
                    stds = gscv.cv_results_['std_test_score']
                    for mean, std, params in zip(means, stds, gscv.cv_results_['params']):
                        logging.debug('{:.3f} (+/- {:.3f}) for {}'.format(mean, std, params))  # or std*2?

                # predict on test with identified best parameters and get scores
                if reduced:
                    y_pred = gscv.predict(self.red_X_test)
                    pd.testing.assert_index_equal(self.test_labeling_df.index, self.red_X_test.index)
                else:
                    y_pred = gscv.predict(self.X_test)
                    pd.testing.assert_index_equal(self.test_labeling_df.index, self.X_test.index)

            y_name = 'y_{}'.format(model_name)
            t_name = 't_{}'.format(model_name)
            self.test_labeling_df[y_name] = y_pred
            self.get_t_clf(y_col_name=y_name, time_col_name=t_name)
            t_sum, t_sgmean = self.compute_opt_measures(t_name)

            if self.params['verbose']:
                logging.info('Test accuracy score: {}'.format(metrics.accuracy_score(self.y_test, y_pred)))
                logging.info('Test b_accuracy score: {}'.format(metrics.balanced_accuracy_score(self.y_test, y_pred)))
                logging.info('Test W_accuracy score: {}'.format(
                    metrics.accuracy_score(self.y_test, y_pred, sample_weight=self.W_test)))
                logging.info('Test bW_accuracy score: {}'.format(
                    metrics.balanced_accuracy_score(self.y_test, y_pred, sample_weight=self.W_test)))
                logging.info('Classification report:\n{}'.format(
                    metrics.classification_report(self.y_test, y_pred, target_names=self.params['clf_target_names'])))
                logging.info('Confusion matrix:\n{}'.format(metrics.confusion_matrix(self.y_test, y_pred)))

            # store scores
            self.results['model_sel']['clf_reports'][model_name] = metrics.classification_report(
                self.y_test, y_pred,
                target_names=self.params['clf_target_names'],
                output_dict=True
            )
            self.results['model_sel']['conf_matrices'][model_name] = metrics.confusion_matrix(self.y_test, y_pred)
            self.results['model_sel']['scores'][model_name] = [
                metrics.accuracy_score(self.y_test, y_pred),
                metrics.balanced_accuracy_score(self.y_test, y_pred),
                metrics.accuracy_score(self.y_test, y_pred, sample_weight=self.W_test),
                metrics.balanced_accuracy_score(self.y_test, y_pred, sample_weight=self.W_test),
                metrics.precision_score(self.y_test, y_pred, average=self.params['average_type']),
                metrics.recall_score(self.y_test, y_pred, average=self.params['average_type']),
                metrics.f1_score(self.y_test, y_pred, average=self.params['average_type']),
                t_sum, t_sgmean,
                t_sum / self.opt_scores['target_t_sum'],
                self.opt_scores['def_t_sum'] / t_sum
            ]
            if model_name == 'DummyC':
                self.results['model_sel']['best_estimators'][model_name] = dummy
            else:
                self.results['model_sel']['best_estimators'][model_name] = gscv.best_estimator_

        return

    def feature_selection(self, poly, scaler, selector,
                          ft_scores_attr='coef_',
                          use_sample_weight=False, average_type='weighted'):
        """
        :param poly: PolynomialFeatures object, to expand the original features set
        :param scaler: scaler object, to build the ft selection pipeline (e.g., StandardScaler)
        :param selector: selector object with estimator specified, to build the ft selection pipeline (e.g., SelectFromModel, RFECV)
        :param ft_scores_attr: str, either 'feature_importances_' or 'coef_', depending on the estimator of selector
        :param use_sample_weight: bool, whether to use sample_weight in fit (not always possible, depends on selector)
        :param average_type: str, average type for scores computation
        :return:
        Note: RFECV and RFE *require* use_sample_weight=False.
        Note: we can specify a custom scoring function in the selector of type RFECV, *BUT not* specify weights in it.
        This is because the fit method of RFECV transforms Pandas objects into arrays, so
        when selecting the appropriate weights for the CV fold the use of .index is forbidden.
        """
        if poly:
            scaler_clone = clone(scaler)
            # build the ft selection pipeline
            ft_sel_pipe = Pipeline([('poly', poly),
                                    ('scl', scaler),
                                    ('selector', selector)])
        else:
            scaler_clone = clone(scaler)
            # build the ft selection pipeline
            ft_sel_pipe = Pipeline([('scl', scaler),
                                    ('selector', selector)])

        self.results['ft_sel'] = {
            'use_sample_weight': use_sample_weight,
            'ft_pipe': ft_sel_pipe,
            'clf_reports': OrderedDict(),
            'conf_matrices': OrderedDict(),
            'scores': OrderedDict(),
            'ft_estimators': OrderedDict()
        }

        if self.params['verbose']:
            logging.info("\n------------------------------------------------------------")
            logging.info("Feature selection on X_train {}...".format(self.X_train.shape))

        # scale and fit selection model
        if use_sample_weight:
            ft_sel_pipe.fit(self.X_train, self.y_train, selector__sample_weight=self.W_train.values)
            logging.info("Sample weights are set for fitting the selector model")
        else:
            ft_sel_pipe.fit(self.X_train, self.y_train)

        # get support
        # logging.info("Fts names: {}".format(poly.get_feature_names(self.features)))
        fit_selector = ft_sel_pipe.named_steps['selector']
        ft_support = fit_selector.get_support()
        selector_model = fit_selector.estimator_

        # save reduced datasets
        if poly:
            self.results['ft_sel']['selected_fts'] = list(compress(poly.get_feature_names(self.clf_fts), ft_support))
            poly_X_train = pd.DataFrame(data=poly.fit_transform(self.X_train),
                                        index=self.X_train.index,
                                        columns=poly.get_feature_names(self.clf_fts))
            poly_X_train = poly_X_train[self.results['ft_sel']['selected_fts']].copy()

            poly_X_test = pd.DataFrame(data=poly.fit_transform(self.X_test),
                                       index=self.X_test.index,
                                       columns=poly.get_feature_names(self.clf_fts))
            poly_X_test = poly_X_test[self.results['ft_sel']['selected_fts']].copy()
            self.red_X_train = poly_X_train
            self.red_X_test = poly_X_test

        else:
            self.results['ft_sel']['selected_fts'] = list(compress(self.clf_fts, ft_support.tolist()))
            self.red_X_train = self.X_train[self.results['ft_sel']['selected_fts']].copy()
            self.red_X_test = self.X_test[self.results['ft_sel']['selected_fts']].copy()

        # get fts scores
        if not hasattr(selector_model, ft_scores_attr):
            raise ValueError(
                "Ft scores '{}' not an attribute of {} object.".format(
                    ft_scores_attr, type(selector_model)))
        self.results['ft_sel']['ft_scores'] = getattr(selector_model, ft_scores_attr)

        if self.params['verbose']:
            logging.info("\nSelector {} "
                         "\nselected {} features:\n{}".format(fit_selector,
                                                              len(self.results['ft_sel']['selected_fts']),
                                                              self.results['ft_sel']['selected_fts']
                                                              ))

        # fit and score base estimator on reduced dataset (with scaler) and save it
        model_clone = clone(selector_model)  # clones are not fit on data
        model_pipe = Pipeline([('scl', scaler_clone),
                               ('clf', model_clone)])
        if use_sample_weight:
            model_pipe.fit(self.red_X_train, self.y_train, clf__sample_weight=self.W_train.values)
        else:
            model_pipe.fit(self.red_X_train, self.y_train)

        y_pred = model_pipe.predict(self.red_X_test)
        pd.testing.assert_index_equal(self.test_labeling_df.index, self.red_X_test.index)
        y_name = 'y_FtSelModel'
        t_name = 't_FtSelModel'
        self.test_labeling_df[y_name] = y_pred
        self.get_t_clf(y_col_name=y_name, time_col_name=t_name)
        t_sum, t_sgmean = self.compute_opt_measures(t_name)

        self.results['ft_sel']['clf_reports']['selector_model'] = metrics.classification_report(
            self.y_test, y_pred,
            target_names=self.params['clf_target_names'],
            output_dict=True
        )
        self.results['ft_sel']['conf_matrices']['selector_model'] = metrics.confusion_matrix(self.y_test, y_pred)
        self.results['ft_sel']['scores']['selector_model'] = [
            metrics.accuracy_score(self.y_test, y_pred),
            metrics.balanced_accuracy_score(self.y_test, y_pred),
            metrics.accuracy_score(self.y_test, y_pred, sample_weight=self.W_test),
            metrics.balanced_accuracy_score(self.y_test, y_pred, sample_weight=self.W_test),
            metrics.precision_score(self.y_test, y_pred, average=average_type),
            metrics.recall_score(self.y_test, y_pred, average=average_type),
            metrics.f1_score(self.y_test, y_pred, average=average_type),
            t_sum, t_sgmean,
            t_sum / self.opt_scores['target_t_sum'],
            self.opt_scores['def_t_sum'] / t_sum
        ]
        self.results['ft_sel']['ft_estimators']['selector_model'] = model_pipe

        return

    def mtl_feature_selection(self, poly, scaler, selector,
                              ft_scores_attr='coef_'):
        """
        :param poly: PolynomialFeatures object, to expand the original features set
        :param scaler: scaler object, to build the ft selection pipeline (e.g., StandardScaler)
        :param selector: selector object with estimator specified, to build the ft selection pipeline
        :param ft_scores_attr: str, either 'feature_importances_' or 'coef_', depending on the estimator of selector
        :return:
        Perform feature selection using MultiTaskLasso regression.
        Note: no need for double split of the dataset, ft selection is performed on X_train, with lasso targets y_mtl_*.
        Also, note that regression step could be applied before this type of ft selection.
        """
        if not self.mtl_targets:
            raise ValueError("No targets were specified for MTL selector.")
        if poly:
            scaler_clone = clone(scaler)
            # build the ft selection pipeline
            ft_sel_pipe = Pipeline([('poly', poly),
                                    ('scl', scaler),
                                    ('selector', selector)])
        else:
            scaler_clone = clone(scaler)
            # build the ft selection pipeline
            ft_sel_pipe = Pipeline([('scl', scaler),
                                    ('selector', selector)])

        self.results['ft_sel'] = {
            'ft_pipe': ft_sel_pipe,
        }

        # create self.y_mtl from self.mtl_targets
        # select from self.df the portion of self.X_train and the columns of self.mtl_targets
        self.y_mtl_train = self.df.loc[self.X_train.index][self.mtl_targets]
        self.y_mtl_test = self.df.loc[self.X_test.index][self.mtl_targets]

        if self.params['verbose']:
            logging.info("Created targets for MultiTaskLasso {}: {} {}".format(
                self.y_mtl_train.columns, self.y_mtl_train.shape, self.y_mtl_test.shape
            ))
            logging.info("\n------------------------------------------------------------")
            logging.info("Feature selection with MTL on X_train {} and y_mtl_train {}...".format(
                self.X_train.shape, self.y_mtl_train.shape
            ))

        # scale and fit selection model
        ft_sel_pipe.fit(self.X_train, self.y_mtl_train)

        # get support
        fit_selector = ft_sel_pipe.named_steps['selector']
        selector_model = fit_selector.estimator_
        ft_support = fit_selector.get_support()
        # save estimator (not refitting and scoring since there is no classification!)
        self.results['ft_sel']['ft_estimator'] = selector_model

        # save reduced datasets
        if poly:
            self.results['ft_sel']['selected_fts'] = list(compress(poly.get_feature_names(self.clf_fts), ft_support))
            poly_X_train = pd.DataFrame(data=poly.fit_transform(self.X_train),
                                        index=self.X_train.index,
                                        columns=poly.get_feature_names(self.clf_fts))
            poly_X_train = poly_X_train[self.results['ft_sel']['selected_fts']].copy()

            poly_X_test = pd.DataFrame(data=poly.fit_transform(self.X_test),
                                       index=self.X_test.index,
                                       columns=poly.get_feature_names(self.clf_fts))
            poly_X_test = poly_X_test[self.results['ft_sel']['selected_fts']].copy()
            self.red_X_train = poly_X_train
            self.red_X_test = poly_X_test
        else:
            self.results['ft_sel']['selected_fts'] = list(compress(self.clf_fts, ft_support.tolist()))
            self.red_X_train = self.X_train[self.results['ft_sel']['selected_fts']].copy()
            self.red_X_test = self.X_test[self.results['ft_sel']['selected_fts']].copy()

        # get fts scores
        if not hasattr(selector_model, ft_scores_attr):
            raise ValueError(
                "Ft scores '{}' not an attribute of {} object.".format(
                    ft_scores_attr, type(selector_model)))
        self.results['ft_sel']['ft_scores'] = getattr(selector_model, ft_scores_attr)

        if self.params['verbose']:
            logging.info("\nSelector {} "
                         "\nselected {} features:\n{}".format(fit_selector,
                                                              len(self.results['ft_sel']['selected_fts']),
                                                              self.results['ft_sel']['selected_fts']
                                                              ))
        return

    def get_t_target(self, y_target_name, df_in):
        """
        :param y_target_name: str, name of the column containing target
        :param df_in: DataFrame, to iterate through
        :return: pd.Series, containing the computed target times
        Note: this is basically what done in get_t_clf, except we do *not* use test_labeling_df etc.
        We use t_target for computing the custom loss function wtarget_loss.
        """
        assert isinstance(df_in, pd.DataFrame)
        t_target = pd.Series(index=df_in.index)
        for index, row in df_in.iterrows():
            if row[y_target_name] == -1:
                t_target.at[index] = row['Time_NL']
            elif row[y_target_name] == 1:
                t_target.at[index] = row['Time_L']
            elif row[y_target_name] == 0:
                assert self.params['multi_class']
                t_target.at[index] = (row['Time_L'] + row['Time_NL']) / 2.
            else:
                raise ValueError("Found invalid label")
        return t_target

    def get_t_clf(self, y_col_name, time_col_name):
        """
        :param y_col_name: str, name of column containing (predicted) labels
        :param time_col_name: str, name of the column that will be created with (predicted) times
        :return:
        Add a column to self.test_labeling_df with the times corresponding to the predicted labels.
        """
        assert 'Time_L' in list(self.test_labeling_df.columns)
        assert 'Time_NL' in list(self.test_labeling_df.columns)

        loc = self.test_labeling_df.columns.get_loc(y_col_name)
        self.test_labeling_df.insert(loc+1, time_col_name, None)
        for index, row in self.test_labeling_df.iterrows():
            if row[y_col_name] == -1:
                self.test_labeling_df.at[index, time_col_name] = row['Time_NL']
            elif row[y_col_name] == 1:
                self.test_labeling_df.at[index, time_col_name] = row['Time_L']
            elif row[y_col_name] == 0:
                assert self.params['multi_class']
                self.test_labeling_df.at[index, time_col_name] = (row['Time_L'] + row['Time_NL']) / 2.
            else:
                raise ValueError("Found invalid label in column {}".format(y_col_name))
        return

    def compute_opt_measures(self, time_col_name):
        """
        :param time_col_name: str, name of the column with (predicted) times
        :return:
        Compute 'optimization' measures to score the classifier (and the default solver).
        """
        assert time_col_name in list(self.test_labeling_df.columns)
        t_sum = self.test_labeling_df[time_col_name].sum()
        t_sgmean = shifted_gmean(col=self.test_labeling_df[time_col_name])
        return t_sum, t_sgmean

    def get_scores_table(self, step='model_sel', scores_entry='scores'):
        """
        :param step: str, key of self.results
        :param scores_entry: str, key of self.results[step] containing scores
        :return:
        Scores table of *classification* estimators in step, in DataFrame format.
        """
        if step == 'regression':
            raise ValueError('Cannot print scores table for regression!')
        if step == 'ft_sel' and self.mtl_targets:
            raise ValueError('Cannot print scores table for ft selector of type MultiTaskLasso!')
        opt_dict = OrderedDict()
        opt_dict['target'] = [
            None, None, None, None, None, None, None,
            self.opt_scores['target_t_sum'],
            self.opt_scores['target_t_sgmean'],
            self.opt_scores['target_t_sum'] / self.opt_scores['target_t_sum'],
            self.opt_scores['def_t_sum'] / self.opt_scores['target_t_sum']
        ]
        opt_dict['def'] = [
            None, None, None, None, None, None, None,
            self.opt_scores['def_t_sum'],
            self.opt_scores['def_t_sgmean'],
            self.opt_scores['def_t_sum'] / self.opt_scores['target_t_sum'],
            self.opt_scores['def_t_sum'] / self.opt_scores['def_t_sum']
        ]
        opt_df = pd.DataFrame.from_dict(opt_dict, orient='index')
        opt_df.columns = ['accuracy', 'b_accuracy', 'w_accuracy', 'bw_accuracy', 'precision', 'recall', 'f1-score',
                          't_sum', 't_sgmean', 't_sum_on_target', 'def_on_t_sum']

        scores_df = pd.DataFrame.from_dict(self.results[step][scores_entry], orient='index')
        scores_df.columns = ['accuracy', 'b_accuracy', 'w_accuracy', 'bw_accuracy', 'precision', 'recall', 'f1-score',
                             't_sum', 't_sgmean', 't_sum_on_target', 'def_on_t_sum']

        return pd.concat([scores_df, opt_df], axis=0, join='outer')
