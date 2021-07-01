"""

miqp-clf2lin
------------

Miscellanea utility functions.


"""

import numpy as np
import pandas as pd
import pickle
import os
import glob
from collections import OrderedDict


def get_path_dict(batch_name, names, model_dir, save, save_dir):
    """
    :param batch_name: str, name of the batch of benchmark instances
    :param names: list, of names of instances that passed benchmark
    :param model_dir: str, pathway to directory containing MIQP models
    :param save: bool, whether to save the output dictionary
    :param save_dir: str, pathway to directory where output should be saved
    :return: path_dict, dict {model_name: path_to_model}
    Compiles a dictionary of pathways of MIQP models that passed the benchmark checks.
    """
    print("\nCreating {name: path} dictionary...")
    # create and save dictionary of pathways to instances {name: path_to_name}
    path_dict = dict.fromkeys(names)
    count = 0
    count_add = 0
    for root, dirs, files in os.walk(os.path.relpath(model_dir)):  # abspath could be used if necessary
        for file in files:
            count += 1
            line = os.path.join(root, file)
            line_name = line.split('/')[-1]
            # remove suffix
            if line_name.endswith('.lp.gz'):
                name = line_name.replace('.lp.gz', '')
            elif line_name.endswith('.mps.gz'):
                name = line_name.replace('.mps.gz', '')
            elif line_name.endswith('.sav.gz'):
                name = line_name.replace('.sav.gz', '')
            else:
                name = line_name
                # print("Suffix not found for {}".format(name))
            if name in names:
                count_add += 1
                path_dict[name] = line
    if save:
        pickle.dump(path_dict, open(os.path.join(save_dir, '{}_path_dict.pkl'.format(batch_name)), 'wb'))
    print("Processed: {}".format(count))
    print("Added to dict: {}".format(count_add))
    return path_dict


def get_raw_fts_df(path_dir_npz, num_fts, batch_name, save, save_dir):
    """
    :param path_dir_npz: str, path to directory where npz files will be stored
    :param num_fts: int, number of raw fts (58)
    :param batch_name: str, name of the batch of benchmark instances
    :param save: bool, whether to save the output DataFrame
    :param save_dir: str, pathway to directory where output should be saved
    :return: ft_df, DataFrame
    indexed by instances in batch and with raw fts as columns.
    """
    current_dir = os.getcwd()
    # go through directory of npz files
    ft_dict = OrderedDict()
    os.chdir(path_dir_npz)
    for f in glob.glob('*.npz'):
        dat = np.load(f)  # contains: name, ft_arr, ft_time
        ft_dict[str(dat['name'])] = dat['ft_arr']

    # define DF
    ft_df_col = pd.DataFrame.from_dict(ft_dict)
    ft_df = ft_df_col.transpose()
    ft_df.columns = ['rf' + str(i) for i in range(1, num_fts + 1)]

    print("Raw fts DF has shape: {}".format(ft_df.shape))
    if save:
        os.chdir(current_dir)
        pickle.dump(ft_df, open(os.path.join(save_dir, '{}_raw_ft_df.pkl'.format(batch_name)), 'wb'))
    return ft_df


def batch_preprocess(batch_name, agg_df, raw_fts_df, save, save_dir):
    """
    :param batch_name: str, name of the dataset batch
    :param agg_df: DataFrame, aggregated data from benchmark
    :param raw_fts_df: DataFrame, containing raw features
    :param save: bool, whether to save the output DataFrame
    :param save_dir: str, pathway to directory where output should be saved
    :return: mdf, DataFrame
    where agg_df and raw_fts_df are merged and new columns added.
    """
    # add 'name' from index
    for f in [agg_df, raw_fts_df]:
        assert isinstance(f, pd.DataFrame)
        if 'name' not in f.columns:
            f['name'] = f.index
    # merge
    mdf = pd.merge(
        raw_fts_df, agg_df,
        how='inner',
        on='name',
        left_index=True,
        copy=True,
        validate='one_to_one'
    )
    print("Merged DF has shape {} (loaded: {} and {})".format(mdf.shape, raw_fts_df.shape, agg_df.shape))

    # add origin and other columns
    mdf.insert(0, 'origin', batch_name)
    mdf.insert(1, 'convex', None)
    mdf.insert(2, 'prob_eig_frac', None)
    mdf.insert(3, 'prob_eig_frac_original', None)

    for index, row in mdf.iterrows():

        # determine convexity using rf52 and rf53 (# of positive and negative eigenvalues after abs_tol correction)
        # min cases
        if row['rf1'] == 1 and row['rf53'] > 0:
            mdf.at[index, 'convex'] = False
        elif row['rf1'] == 1 and row['rf53'] == 0:
            mdf.at[index, 'convex'] = True
        # max cases
        elif row['rf1'] == -1 and row['rf52'] > 0:
            mdf.at[index, 'convex'] = False
        elif row['rf1'] == -1 and row['rf52'] == 0:
            mdf.at[index, 'convex'] = True

        # determine fraction of problematic eigenvalues (corrected and uncorrected)
        if row['rf1'] == 1:
            mdf.at[index, 'prob_eig_frac'] = row['rf53'] / float(row['rf4'])
            mdf.at[index, 'prob_eig_frac_original'] = row['rf46'] / float(row['rf4'])
        elif row['rf1'] == -1:
            mdf.at[index, 'prob_eig_frac'] = row['rf52'] / float(row['rf4'])
            mdf.at[index, 'prob_eig_frac_original'] = row['rf45'] / float(row['rf4'])

    print("Final DF has shape {}".format(mdf.shape))
    if save:
        pickle.dump(mdf, open(os.path.join(save_dir, '{}_preprocessed.pkl'.format(batch_name)), 'wb'))

    return mdf


if __name__ == '__main__':

    # uncomment different blocks to try utilities individually (out of the main scripts in which they are called)
    batch = 'setD'

    ##########################
    #       get_path_dict
    ##########################

    # batch = 'setD'
    # names_pkl_path = '../data/setD/setD_names.pkl'
    # instances_dir_path = '../instances/setD'
    # save_dir_path = '../data/setD'
    #
    # # read model names from processed benchmark
    # with open(names_pkl_path, 'rb') as f_names:
    #     names_list = pickle.load(f_names)
    # f_names.close()
    #
    # name_path_dict = get_path_dict(
    #         batch_name=batch,
    #         names=names_list,
    #         model_dir=instances_dir_path,
    #         save=True,
    #         save_dir=save_dir_path
    #     )

    ##########################

    ##########################
    #      get_raw_fts_df
    ##########################

    # batch = 'setD'
    # save_dir_path = '../data/setD'
    # dir_npz_path = '../data/setD/raw_npz'
    # n_fts = 58
    #
    # raw_df = get_raw_fts_df(
    #     path_dir_npz=dir_npz_path,
    #     num_fts=n_fts,
    #     batch_name=batch,
    #     save=True,
    #     save_dir=save_dir_path
    # )

    ##########################

    ##########################
    #    batch_preprocess
    ##########################

    # batch = 'setD'
    # agg_df_path = '../data/setD/setD_aggregated.pkl'
    # raw_ft_df_path = '../data/setD/setD_raw_ft_df.pkl'
    # save_dir_path = '../data/setD'
    #
    # df_prep = batch_preprocess(
    #     batch_name=batch,
    #     agg_df=pickle.load(open(agg_df_path, 'rb')),
    #     raw_fts_df=pickle.load(open(raw_ft_df_path, 'rb')),
    #     save=True,
    #     save_dir=save_dir_path
    # )
