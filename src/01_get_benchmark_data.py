"""

miqp-clf2lin
------------

Read .res files produced during MIQP benchmarking to create a DataFrame (DF).
The data is subsequently cleaned-up, with consistency and solvability checks applied.
Labeling procedures and aggregations over seeds are applied to get a final DF.

Eventually, names of MIQPs in the final DF can be stored with their path
(this will be used to extract raw_fts).

Modules dependencies
    - res2pandas.py (proprietary script, not shared): code to organize benchmark data and run checks
    - label.py (proprietary script, not shared): code of labeling procedures

    NOTE: aggregating over seed wins corresponds to assigning a MultiLabel target {L, NL, T}.
    Binary labeling to get BinLabel is obtained *after* aggregation takes place.

    - utilities.py
        - get_path_dict: save dictionary of {model_name: path_to_model}

Specify benchmark paths before running as

    python 01_get_benchmark_data.py -s -b <BATCH_NAME>

"""

import pickle
import os
import argparse

import res2pandas as r2p
import label
import utilities


""" 
Pathways for different benchmark batches.
Specify:
    - res_dirs: list of directories containing .res files. NOTE: L first, then NL
    - exp_name: name of the experiment benchmark batch
    - pkl_dir: directory in which data will be saved
    - model_dir: (absolute) directory containing MIQP models

"""

batches_dict = {
    'setD': {
        'res_dirs': [],
        'exp_name': '',
        'pkl_dir': '',
        'model_dir': '',
    },
    'miqpall': {
        'res_dirs': [],
        'exp_name': '',
        'pkl_dir': '',
        'model_dir': '',
    },
    'neos': {
        'res_dirs': [],
        'exp_name': '',
        'pkl_dir': '',
        'model_dir': '',
    }
}

""" Aggregation functions for columns """

aggreg_func = {
    'Iters': label.shifted_gmean,
    'Nodes': label.shifted_gmean,
    'Time': label.shifted_gmean,
    'Status': label.num_timeout,
    'SeedWins': label.consistent_win,
    'Weight': label.shifted_gmean,
    'Root Dual Bound': label.arit_mean,
    'RtTime': label.shifted_gmean,
    'RLPTime': label.shifted_gmean,
    'Conss': label.arit_mean,
    'Vars': label.arit_mean,
    'Nonzs': label.arit_mean,
}

""" Renaming of columns """

renaming_dict = {
    'linearize Time': 'Time_L', 'nolinearize Time': 'Time_NL',
    'linearize Nodes': 'Nodes_L', 'nolinearize Nodes': 'Nodes_NL',
    'linearize Root Dual Bound': 'RootDualBound_L', 'nolinearize Root Dual Bound': 'RootDualBound_NL',
    'linearize timeouts': 'timeouts_L', 'nolinearize timeouts': 'timeouts_NL',
    'linearize RtTime': 'RtTime_L', 'nolinearize RtTime': 'RtTime_NL',
    'linearize RLPTime': 'RLPTime_L', 'nolinearize RLPTime': 'RLPTime_NL',
    'linearize Iters': 'Iters_L', 'nolinearize Iters': 'Iters_NL',
    'linearize Vars': 'Vars_L', 'nolinearize Vars': 'Vars_NL',
    'linearize Conss': 'Conss_L', 'nolinearize Conss': 'Conss_NL',
    'linearize Nonzs': 'Nonzs_L', 'nolinearize Nonzs': 'Nonzs_NL',
    'checks SeedWins': 'MultiLabel',
    'checks Weight': 'Weight',
}


if __name__ == '__main__':

    # Parser definition
    parser = argparse.ArgumentParser(description='Parser for MIQP benchmark data script.')
    parser.add_argument(
        '-b',
        '--batch_name',
        type=str,
        required=True,
        help='Name of the batch of instances to be processed. Should be a key in batches_dict.',
    )
    parser.add_argument(
        '--drop_na',
        action='store_false',
        default=True,
        help='Whether nan values should be dropped from DataFrame.'
    )
    parser.add_argument(
        '--drop_incons',
        action='store_false',
        default=True,
        help='Whether inconsistent runs should be dropped from DataFrame.'
    )
    parser.add_argument(
        '--onlysolvedany',
        action='store_false',
        default=True,
        help='Whether to keep only runs solved by at least one method.'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        default=False,
        help='Enable detailed printing to stdout.'
    )
    parser.add_argument(
        '-s',
        '--save',
        action='store_true',
        default=False,
        help='Enable saving of results.'
    )
    args = parser.parse_args()

    if args.batch_name not in batches_dict:
        raise ValueError('The specified batch_name is not in batches_dict.')

    # pathways
    res_dirs = batches_dict[args.batch_name]['res_dirs']
    exp_name = batches_dict[args.batch_name]['exp_name']
    pkl_dir = batches_dict[args.batch_name]['pkl_dir']
    model_dir = batches_dict[args.batch_name]['model_dir']

    # define additional dictionary to store ids of faulty runs
    batch_failures = {}

    # process .res files
    resdf, checks_failures = r2p.rescombine(
        directories=res_dirs,
        dropinconsistent=args.drop_incons,
        onlysolvedany=args.onlysolvedany,
        verbose=args.verbose
    )
    batch_failures['consistency'] = checks_failures['consistency']
    batch_failures['solvability'] = checks_failures['solvability']
    print("\nDF from .res for {} has shape {}".format(args.batch_name, resdf.shape))

    # check number and location of nans, and store them before eventually dropping them
    print("# of nans: {}".format(resdf[resdf.isnull().any(axis=1)].shape))
    print(resdf.isnull().sum())
    batch_failures['nans'] = resdf[resdf.isnull().any(axis=1)].index.tolist()  # tolist --> to_list in pandas 0.24.1
    if args.drop_na:
        resdf.dropna(axis=0, how='any', inplace=True)
        print("\nDF w/o NaN {} has shape {}\n".format(args.batch_name, resdf.shape))

    # tag the winner mode on each run and compute run weight
    resdf.columns = resdf.columns.swaplevel()
    resdf[('SeedWins', 'checks')] = resdf.apply(label.seed_wins, axis=1)
    resdf[('Weight', 'checks')] = resdf.apply(label.get_run_weight, axis=1)
    resdf.columns = resdf.columns.swaplevel()

    # for k in batch_failures.keys():
    #     print('{} runs failed {}'.format(len(batch_failures[k]), k))

    # store full data (not aggregated yet) and failures
    if args.save:
        pickle.dump(resdf, open(os.path.join(pkl_dir, '{}_full.pkl'.format(exp_name)), 'wb'))
        pickle.dump(batch_failures, open(os.path.join(pkl_dir, '{}_failures.pkl'.format(exp_name)), 'wb'))

    # aggregate over seeds
    agg_df = label.agg_frame(
        dfres=resdf,
        func_dict=aggreg_func,
        cols=['Time', 'Nodes', 'SeedWins', 'Root Dual Bound', 'Weight', 'Status',
              'RtTime', 'RLPTime', 'Iters', 'Conss', 'Vars', 'Nonzs']
    )
    print("\nAggregated DF for {} has shape {}".format(args.batch_name, agg_df.shape))

    # cols renaming
    agg_df.columns = [' '.join(col).strip() for col in agg_df.columns.values]
    agg_df.rename(columns=renaming_dict, inplace=True)

    # binary labeling
    agg_df = label.get_binlabel(agg_df)

    # save final DataFrame
    names = agg_df.index.tolist()
    if args.save:
        pickle.dump(names, open(os.path.join(pkl_dir, '{}_names.pkl'.format(exp_name)), 'wb'))
        pickle.dump(agg_df, open(os.path.join(pkl_dir, '{}_aggregated.pkl'.format(exp_name)), 'wb'))
        print("\nAll data outputs have been saved in \n{}.".format(pkl_dir))

    # print("Columns of aggregated DF are:\n{}\n".format(agg_df.columns))
    #
    # print(agg_df['MultiLabel'].value_counts())
    # print(agg_df['BinLabel'].value_counts())

    # create {name: path} dictionary
    path_dict = utilities.get_path_dict(
        batch_name=exp_name,
        names=names,
        model_dir=model_dir,
        save=args.save,
        save_dir=pkl_dir
    )
