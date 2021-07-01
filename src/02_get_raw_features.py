"""

miqp-clf2lin
------------

Extract raw attributes from instances in the datasets.
For each instance, *static* attributes are extracted (refer to doc/raw_features.md for the list of 58 raw attributes).
Each array is saved into an npz file.

NOTE: dynamic features are computed using information from the benchmark, the same used for labeling.

We compute raw features for instances that survived the labeling checks only (cf. 01_get_benchmark_data.py),
so we load the dictionary of {instance_name: path_to_instance} obtained via utilities.get_path_dict.
Once read by the solver, problems are checked on type and max_size (n <= 10K), and eventually discarded.

Raw features are grouped in different functions, which are then all called by raw_features.compute_static_raw_features.
Finally, utilities.get_raw_fts_df can be used to scan all produced npz files and create a unique DataFrame.

Modules dependencies
    - raw_features.py
        - compute_static_raw_features: calls all functions to extract raw static attributes
    - utilities.py
        - get_raw_fts_df: walks through path_dir_npz to create unique DataFrame for the batch of instances.

NOTE: this module requires cplex to read the MIQP instances, but no optimization takes place.

Specify benchmark paths before running as

    python 02_get_raw_features.py -b <BATCH_NAME> -s > <FILENAME>.log

"""

import argparse
import os
import pickle
import time

import cplex
import numpy as np
from scipy.sparse import dok_matrix
from scipy.sparse import linalg

import utilities
import raw_features


""" 
Pathways for different benchmark batches.
Specify:
    - dict_path: directory containing dict {name: path} for batch
    - npz_dir: directory to save npz files
    - df_dir: directory to save single DataFrame data
    
"""

batches_dict = {
    'setD': {
        'dict_path': '../data/setD/setD_path_dict.pkl',
        'npz_dir': '../data/setD/raw_npz',
        'df_dir': '../data/setD',
    },
    'miqpall': {
        'dict_path': '',
        'npz_dir': '',
        'df_dir': '',
    },
    'neos': {
        'dict_path': '',
        'npz_dir': '',
        'df_dir': '',
    }
}

if __name__ == "__main__":

    # Parser definition
    parser = argparse.ArgumentParser(description='Arg parser for raw fts extraction script.')

    parser.add_argument(
        '-b',
        '--batch_name',
        type=str,
        required=True,
        help='Name of the batch of instances to be processed. Should be a key in batches_dict.',
    )
    parser.add_argument(
        '--abs_tol_fix',
        type=float,
        default=9 * 1e-6,
        help='Tolerance on absolute value to determine zero eigenvalues (9*1e-6).'
    )
    parser.add_argument(
        '--eig_max_fix',
        type=int,
        default=500,
        help='Maximal number of eigenvalues computed (500).'
    )
    parser.add_argument(
        '--size_max',
        type=int,
        default=10000,
        help='Maximal problem size accepted.'
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

    path_to_store_npz = batches_dict[args.batch_name]['npz_dir']
    path_dict_pkl = batches_dict[args.batch_name]['dict_path']
    path_to_save_df = batches_dict[args.batch_name]['df_dir']

    if not os.path.exists(path_to_store_npz):
        os.makedirs(path_to_store_npz, exist_ok=True)

    # load dictionary {instance_name: path_to_instance}
    path_dict = pickle.load(open(path_dict_pkl, 'rb'))
    pb = cplex.Cplex()
    count = 0
    count_reject = 0
    for name in path_dict.keys():
        count += 1
        print("{} Processing {}".format(count, name))
        pb.read(path_dict[name])
        pb_size = pb.variables.get_num()

        # condition on problem type
        print("\tProblem type: ", pb.get_problem_type())
        if (pb.get_problem_type() != 7) and (pb.get_problem_type() != 8):
            print("\tProblem is not MIQP but {} instead.".format(pb.problem_type[pb.get_problem_type()]))
            count_reject += 1
            continue

        # condition on problem size
        if pb_size > args.size_max:
            print("\tSize {} is bigger than size_max {} - Instance rejected.".format(pb_size, args.size_max))
            count_reject += 1
            continue

        print("\tInformation and features extraction...")
        ft_arr, ft_time = raw_features.compute_static_raw_features(
            c=pb,
            filename=path_dict[name],
            abs_tol=args.abs_tol_fix,
            eig_max=args.eig_max_fix
        )
        np.savez(
            os.path.join(path_to_store_npz, '{}_fts'.format(name)),
            name=name,
            ft_arr=ft_arr,
            ft_time=ft_time,
        )
    print("\n# of processed instances is {}".format(count))
    print("# of rejected instances is {}".format(count_reject))
    print("# of created npz is {}".format(count - count_reject))

    pb.end()

    # create unique DF for the batch
    utilities.get_raw_fts_df(
        path_dir_npz=path_to_store_npz,
        num_fts=58,
        batch_name=args.batch_name,
        save=args.save,
        save_dir=path_to_save_df,
    )
