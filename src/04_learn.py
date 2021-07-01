"""

miqp-clf2lin
------------

Perform learning procedures on the MIQP data, including regression, feature selection and classification.
Main parameters are passed via a .yaml file containing the desired configuration for the experiment (see ../configs/).
Experiments are performed with scikit-learn, and managed via the custom class in LearningExperiment.
Models and hyper-parameters are instead specified in models.py.

Once a LearningExperiment object is correctly initialized, the specified learning steps unfold.
Results as well as objects themselves are saved for every seed. Averaged results and plots are also produced.

Modules dependencies
    - experiment.py
        - LearningExperiment: custom class to manage scikit-learn experiments
    - models.py
        - functions defining regression, feature selection and classification models, as well as scoring methods

    Note: the dictionary of feature subsets (obtained with feature_subsets.py) is also used,
    and its path should be specified in the config.yaml file.
    'reg_fts' and 'clf_fts' entries should correspond to keys of the dictionary.

Legend for configuration .yaml files entries:

    data_setting:   one in {AllMulti, BinNoTie, BinLabel}
    random_seeds:   list of seeds for which the experiment will be repeated
    description:    literal description of the experiment
    short_name:     short description of the experiment
    plot_title:     title for results plots
    save:           whether to save results
    verbose:        whether to print messages during the experiment

    reg:            whether regression should be performed
    reg_fts:        a key in fts_dict, fts to be used for regression
    reg_target:     target for regression
    reg_model:      regression model whose prediction should be used in classification (one of the trained models)

    selector_type:  type of feature selector, one in {None, SVC, RFECV, MTL}
    max_fts:        max number of features for SVC selector
    mtl_targets:    list of targets for MTL selector
    poly_fts:       whether to expand feature set polynomially
    fts_samplew:    whether to use sample weights when fitting selectors

    clf_fts:        a key in fts_dict, fts to be used for classification
    scorer:         scoring function to be used in cross-validation
    clf_samplew:    whether to use sample weights when fitting classifiers

    data_path:      absolute path to data
    fts_dict_path:  absolute path to fts_dict, containing feature subsets
    save_path:      absolute path for saving results

Run as

    python 04_learn.py --cfg_file_path <PATH-TO-YAML-CONFIGURATION>

"""


import pandas as pd
import os
import pickle
import uuid
import argparse
import time
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from sklearn.metrics import make_scorer
from sklearn.externals import joblib

import warnings
from sklearn.exceptions import DataConversionWarning

from experiment import LearningExperiment
import models

# options
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
sns.set(style="ticks")
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


if __name__ == '__main__':

    # Parser definition
    parser = argparse.ArgumentParser(description='Parser for MIQP learning experiments.')
    parser.add_argument(
        '--cfg_file_path',
        type=str,
        required=True,
        help='Path to config.yaml file to be loaded for the learning experiment.'
    )
    args = parser.parse_args()

    # Load config
    config = yaml.safe_load(open(args.cfg_file_path, 'rb'))

    # Experiment identifier
    uid = str(uuid.uuid4())[:8]
    dirname = '{}_{}_Reg{}_Sel{}_Clf{}_{}'.format(
        config['short_name'], config['data_setting'], config['reg'], config['selector_type'], config['clf_fts'],
        uid
    )
    if not os.path.exists(os.path.join(config['save_path'], dirname)):
        os.makedirs(os.path.join(config['save_path'], dirname), exist_ok=True)

    # Logger definition
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(config['save_path'], dirname, dirname + '.log'),
        filemode='w',
        format='%(message)s'
    )
    logger = logging.getLogger('learn_miqp_logger')

    logging.info("Loaded config file\n{}\n".format(args.cfg_file_path))
    logging.info("Experiment description:\n{}".format(config['description']))

    # Data loading
    data = pickle.load(open(config['data_path'], 'rb'))
    assert isinstance(data, pd.DataFrame)
    logging.info("Loaded data shape: {}\n".format(data.shape))

    # Select from available settings
    data_label_dict = {
        'AllMulti': {
            'data': data.copy(),
            'target': 'MultiLabel',
            'multi_class': True,
        },
        'BinNoTie': {
            'data': data.loc[data['MultiLabel'] != 0].copy(),
            'target': 'MultiLabel',
            'multi_class': False,
        },
        'BinLabel': {
            'data': data.loc[~data['BinLabel'].isna()].copy(),
            'target': 'BinLabel',
            'multi_class': False,
        },
    }

    if config['data_setting'] not in data_label_dict:
        raise ValueError("Data setting not found in data_label_dict.")
    df = data_label_dict[config['data_setting']]['data']
    target = data_label_dict[config['data_setting']]['target']
    multi_class = data_label_dict[config['data_setting']]['multi_class']
    if multi_class:
        target_names = ['Class NL', 'Class T', 'Class L']
    else:
        target_names = ['Class NL', 'Class L']
    logging.info("Selected setting and data: {} (shape: {})".format(config['data_setting'], df.shape))
    logging.info("Selected clf target: {}".format(target))
    logging.info("Labels distribution in df: \n{}".format(df[target].value_counts()))

    # Features loading and setting selection
    fts_dict = pickle.load(open(config['fts_dict_path'], 'rb'))
    assert 'general_fts' in fts_dict
    clf_fts = fts_dict['general_fts'] + fts_dict[config['clf_fts']]
    logging.info("\nClf feature setting: {}".format(config['clf_fts']))
    logging.info("Loaded features are: {} (+ {} general ones)".format(
        len(fts_dict[config['clf_fts']]), len(fts_dict['general_fts'])))

    if config['reg']:
        reg_fts = fts_dict['general_fts'] + fts_dict[config['reg_fts']]
        reg_target = config['reg_target']
        logging.info("\nReg feature setting: {}".format(config['reg_fts']))
        logging.info("Loaded features are: {} (+ {} general ones)".format(
            len(fts_dict[config['reg_fts']]), len(fts_dict['general_fts'])))
        logging.info("Regression target is {}".format(reg_target))
    else:
        reg_fts = None
        reg_target = None

    if config['selector_type'] == 'MTL':
        if config['mtl_targets'] == 'None':
            raise ValueError("No targets were specified for MTL selector.")
        mtl_targets = config['mtl_targets']
    else:
        mtl_targets = None

    # Define data structure for box plots (one for each metric)
    metrics_list = ['accuracy', 'b_accuracy', 'w_accuracy', 'bw_accuracy', 'precision', 'recall', 'f1-score',
                    't_sum', 't_sgmean', 't_sum_on_target', 'def_on_t_sum']
    metrics_dict = {name: pd.DataFrame(data=None) for name in metrics_list}

    # Loop through seeds
    results_frames = []
    fts_selected = {}

    logging.info("\nExperiment {} will be repeated on {} different seeds ({})".format(uid, len(config['random_seeds']), config['random_seeds']))
    for seed in config['random_seeds']:
        logging.info("\n------------------------------------------------------------\nSeed {}".format(seed))
        logging.info("------------------------------------------------------------")

        t0 = time.time()

        # Instantiate the LearningExperiment class and set train/test split
        exp = LearningExperiment(
            setting=config['data_setting'],
            df=df,
            clf_fts=clf_fts,
            clf_target=target,
            clf_target_names=target_names,
            random_state=seed,
            multi_class=multi_class,
            reg=config['reg'],
            reg_fts=reg_fts,
            reg_target=reg_target,
            mtl_targets=mtl_targets,
            description=config['description'],
            verbose=config['verbose'],
        )

        """ Regression """
        if config['reg']:
            if config['reg_model'] == 'None':
                raise ValueError("Regression model to be used in classification not specified.")
            exp.set_regression_split(reg_test_size=0.70)

            reg_models, reg_hyparams = models.regression_model_def(seed)
            exp.regression_model_selection(
                model_select_pipe=reg_models,
                hyparams=reg_hyparams,
                num_folds=5
            )

        """ Split for classification """
        exp.set_train_test_split(test_size=0.25, reg_pred_col_name=config['reg_model'])

        """ Feature selection """
        reduced_fts = False
        # Define feature selection components using a set_selector method
        if config['selector_type'] == 'SVC':
            poly_fts, scaler, selector, ft_scores_attr = models.set_svc_selector(
                random_state=seed,
                poly=config['poly_fts'],
                degree=2,
                max_features=config['max_fts']
            )
            exp.feature_selection(
                poly=poly_fts,
                scaler=scaler,
                selector=selector,
                ft_scores_attr=ft_scores_attr,
                use_sample_weight=config['fts_samplew'],
                average_type='weighted'
            )
            reduced_fts = True
            fts_selected[seed] = exp.results['ft_sel']['selected_fts']
            logging.info("\n(seed {}) Scores after ft selection:\n{}".format(seed, exp.get_scores_table(step='ft_sel')))
        elif config['selector_type'] == 'RFECV':
            poly_fts, scaler, selector, ft_scores_attr = models.set_rfecv_selector(
                random_state=seed,
                n_folds=5,
                poly=config['poly_fts'],
                degree=2
            )
            if config['fts_samplew']:
                raise ValueError("fts_samplew cannot be used with selector {}".format(config['selector_type']))
            exp.feature_selection(
                poly=poly_fts,
                scaler=scaler,
                selector=selector,
                ft_scores_attr=ft_scores_attr,
                use_sample_weight=False,
                average_type='weighted'
            )
            reduced_fts = True
            fts_selected[seed] = exp.results['ft_sel']['selected_fts']
            logging.info("\n(seed {}) Scores after ft selection:\n{}".format(seed, exp.get_scores_table(step='ft_sel')))
        elif config['selector_type'] == 'MTL':
            if not exp.mtl_targets:
                raise ValueError("No targets were specified for MTL selector.")
            poly_fts, scaler, selector, ft_scores_attr = models.set_mtl_selector(
                random_state=seed,
                poly=config['poly_fts'],
                degree=2
            )
            if config['fts_samplew']:
                raise ValueError("fts_samplew cannot be used with selector {}".format(config['selector_type']))
            exp.mtl_feature_selection(
                poly=poly_fts,
                scaler=scaler,
                selector=selector,
                ft_scores_attr=ft_scores_attr
            )
            reduced_fts = True
            fts_selected[seed] = exp.results['ft_sel']['selected_fts']
        elif config['selector_type'] == 'None':
            logging.info("\nNo feature selection specified. Continuing with model selection.")
        else:
            raise ValueError("\nNo valid input for feature selection.")

        """ Classification model selection """
        # Define models and hyper-parameters for classification
        model_pipes, hyparams = models.classification_model_def(seed)

        # scoring options
        scorer_dict = {
            'accuracy': 'accuracy',
            'w_accuracy': make_scorer(models.w_accuracy, greater_is_better=True, exp_obj=exp),
            'w_loss': make_scorer(models.w_loss, greater_is_better=False, exp_obj=exp),
            'wtarget_loss': make_scorer(models.wtarget_loss, greater_is_better=False, exp_obj=exp)
        }

        # model selection
        exp.classification_model_selection(
            model_select_pipe=model_pipes,
            hyparams=hyparams,
            val_scoring=scorer_dict[config['scorer']],
            val_scoring_name=config['scorer'],
            num_folds=5,
            average_type='weighted',
            use_sample_weight=config['clf_samplew'],
            reduced=reduced_fts
        )
        results_frames.append(exp.get_scores_table(step='model_sel'))
        logging.info("\n(seed {}) Scores after model selection:\n{}".format(seed, exp.get_scores_table(step='model_sel')))

        # Perform other steps

        logging.info("\nTotal experiment time: {}".format(time.time() - t0))

        # Get metric-specific row for the seed
        for m in metrics_list:
            row = {name: exp.results['model_sel']['scores'][name][metrics_list.index(m)] for name in exp.results['model_sel']['scores'].keys()}
            metrics_dict[m] = metrics_dict[m].append(pd.Series(row), ignore_index=True)

        # Save the experiment
        if config['save']:
            name = dirname + '_seed_{}'.format(seed)
            joblib.dump(exp, os.path.join(config['save_path'], dirname, name))
            logging.info("\nExperiment output saved at {}".format(os.path.join(config['save_path'], dirname, name)))

    # Average results across seeds
    res_df = pd.concat(results_frames)
    mean_df = res_df.groupby(res_df.index).mean()
    logging.info("\n------------------------------------------------------------")
    logging.info("Averaged results:\n{}".format(mean_df))
    logging.info("\n{}".format(mean_df.to_latex()))
    if config['selector_type'] != 'None':
        logging.info("\nSelected features on seeds:")
        for s in fts_selected.keys():
            logging.info("Seed {}: {}\n{}".format(s, len(fts_selected[s]), fts_selected[s]))

    if config['save']:
        joblib.dump(mean_df, os.path.join(config['save_path'], dirname, '{}_mean_results'.format(uid)))
        joblib.dump(fts_selected, os.path.join(config['save_path'], dirname, '{}_fts_selected'.format(uid)))
        pickle.dump(config, open(os.path.join(config['save_path'], dirname, '{}_config.pkl'.format(uid)), 'wb'))

        # Create box plots
        title = config['plot_title']
        # title = '{} - Reg: {} - Sel: {} - Clf: {}'.format(
        #     config['data_setting'], config['reg'], config['selector_type'], config['clf_fts']
        # )
        for m in metrics_dict.keys():
            fig = plt.figure(figsize=(5, 3))
            ax = sns.boxplot(data=metrics_dict[m], orient='v', color="lightgrey", linewidth=1, width=.5, notch=False)
            ax.set_ylabel(ylabel='{}'.format(m))
            ax.set_title(label=title, fontsize=10)
            plt.savefig(os.path.join(config['save_path'], dirname, '{}_{}.pdf'.format(uid, m)),
                        bbox_inches='tight', pad_inches=0.05)
            plt.close()
