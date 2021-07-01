"""

miqp-clf2lin
------------

Merge data from benchmark and labeling with raw features, and apply preprocessing to add new columns.
Join different batches of data and compute static, preprocessing and root features (cf. doc/features.md).
All features are computed via DataFrame operations, and a new DF is built in features.compute_fts.

Modules dependencies
    - utilities.py
        - batch_preprocess: for a batch of instances, merge data and preprocess it by adding new columns
    - features.py
        - compute_fts: calls all functions to compute final features

Specify benchmark paths before running as

    python 03_get_features.py --batches setD -s --save_data_path ../data/setD/

"""

import os
import time
import pickle
import pandas as pd
import argparse

import utilities
import features

""" 
Pathways for different benchmark batches.
Specify:
    - agg_path: {}_aggregated.pkl location
    - raw_fts_path: {}_raw_ft_df.pkl location
    - save_dir: directory to save preprocessed batch DataFrames

"""

batches_dict = {
    'setD': {
        'agg_path': '../data/setD/setD_aggregated.pkl',
        'raw_fts_path': '../data/setD/setD_raw_ft_df.pkl',
        'save_dir': '../data/setD/',
    },
    'miqpall': {
        'agg_path': '',
        'raw_fts_path': '',
        'save_dir': '',
    },
    'neos': {
        'agg_path': '',
        'raw_fts_path': '',
        'save_dir': '',
    }
}


if __name__ == "__main__":

    # Parser definition
    parser = argparse.ArgumentParser(description='Arg parser for feature computation.')

    parser.add_argument(
        '--batches',
        type=str,
        nargs='+',
        default=['setD', 'miqpall', 'neos'],
        help='List of batches to be processed (all of them should appear as keys in batches_dict).'
    )
    parser.add_argument(
        '--save_data_path',
        type=str,
        help='Location where final data will be saved.'
    )
    parser.add_argument(
        '-s',
        '--save',
        action='store_true',
        default=False,
        help='Enable saving of results.'
    )

    args = parser.parse_args()

    for batch in args.batches:
        if batch not in batches_dict.keys():
            raise ValueError('Batch {} is not specified in batches_dict.'.format(batch))

    # to collect preprocessed frames
    prep_frames = []

    # preprocess each batch specified in args.batches and add it to prep_frames
    for batch in args.batches:
        print("\nProcessing batch {}...".format(batch))
        agg_df = pickle.load(open(batches_dict[batch]['agg_path'], 'rb'))
        raw_df = pickle.load(open(batches_dict[batch]['raw_fts_path'], 'rb'))
        prep_df = utilities.batch_preprocess(
            batch_name=batch,
            agg_df=agg_df,
            raw_fts_df=raw_df,
            save=args.save,
            save_dir=batches_dict[batch]['save_dir']
        )
        prep_frames.append(prep_df)

    # concatenate preprocessed frames
    data = pd.concat(
        objs=prep_frames,
        axis=0,
        join='outer',
    )
    print("\nConcatenated DF has shape {}".format(data.shape))

    # compute features
    print("\nComputing features...")
    t0 = time.time()
    data_fts = features.compute_fts(data)
    t1 = time.time()
    print("Features data shape: {}".format(data_fts.shape))
    print("Computing time: {}".format(t1 - t0))
    print("Columns:\n")
    print(data_fts.columns)

    print("\n--> nan values in {} rows".format(data_fts[data_fts.isnull().any(axis=1)].shape[0]))
    if data_fts[data_fts.isnull().any(axis=1)].shape[0] > 0:
        nan_cols = data_fts.isna().sum()
        print("--> nan columns are: ")
        print(nan_cols.loc[nan_cols > 0])

    if args.save:
        names = '_'.join(args.batches)
        # 'all_data_features_times.pkl'
        pickle.dump(data_fts, open(os.path.join(args.save_data_path, '{}_features_times.pkl'.format(names)), 'wb'))
