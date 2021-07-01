"""

miqp-clf2lin
------------

Definition of different feature subsets to be used in learning experiments.
A dictionary is created with different lists of features, and finally saved.
For a complete list of available features, refer to doc/processed_features.md

Run as

    python feature_subsets.py

"""

import argparse
import os
import pickle


if __name__ == "__main__":

    # Parser definition
    parser = argparse.ArgumentParser(description='Arg parser for feature subsets definition.')

    parser.add_argument(
        '--save_dict_path',
        type=str,
        default='../data',
        help='Location where dictionary will be saved.'
    )

    args = parser.parse_args()

    fts_dict = {
        'general_fts': [
            'Name', 'Origin', 'Curvature', 'BinLabel', 'MultiLabel', 'Weight', 'Time_L', 'Time_NL'
        ],

        'Init60': [
            # general
            'RSizes', 'RBin', 'RCont_RInt',
            # quadratic objective
            'RNnzDiagBin', 'RNnzDiagCont_RNnzDiagInt',
            'DiagDensity', 'OutDiagDensity', 'QDensity', 'RBinBin', 'RContCont_RIntInt',
            'RMixedBin', 'RMixedCont_RMixedInt',
            'RNonLinTerms', 'RNonLinTermsNnz', 'RelVarsLinInc', 'RelConssLinInc', 'RLinSizes',
            'NormMaxDegBin', 'NormMaxDegCont_NormMaxDegInt',
            'AvgDiagDom', 'RDiagCoeff', 'ROutDiagCoeff',
            # linear objective
            'RNnzBinLin', 'RNnzContLin_RNnzIntLin',
            'HasLinearTerm', 'LinDensity', 'RLinCoeff',
            # constraints
            'ConssDensity', 'RConssBin', 'RConssCont', 'RConssInt', 'RConssCoeff', 'RRhsCoeff',
            # spectrum
            'RQTrace', 'QSpecNorm',
            'RQRankEig', 'HardEigenPerc',
            'AvgSpecWidth',
            'RPosEigen', 'RNegEigen', 'RZeroEigen',
            'RAbsEigen', 'RNZeroEigenDiff', 'HardEigenPercDiff',
            # preprocessing
            'prep_RelVarsIncL', 'prep_RelVarsIncNL', 'prep_RelConssIncL', 'prep_RelConssIncNL',
            'prep_RSizesL', 'prep_RSizesNL',
            'prep_ConssDensityL', 'prep_ConssDensityNL', 'prep_ConssDensityDiff',
            'prep_RelConssDensityL', 'prep_RelConssDensityNL',
            # root node
            'root_RtTimeDiff', 'root_RLPTimeDiff',
            'root_SignRDBDiff', 'root_RelRDBDiff', 'root_RelSignRDBDiff',
        ],

        'Selected': [
            # general
            'RBin', 'RCont_RInt',
            # quadratic objective
            'RNnzDiagCont_RNnzDiagInt', 'OutDiagDensity', 'QDensity', 'RBinBin', 'RContCont_RIntInt',
            'RNonLinTerms', 'RelVarsLinInc', 'RLinSizes', 'NormMaxDegBin', 'NormMaxDegCont_NormMaxDegInt',
            # linear objective
            'RNnzContLin_RNnzIntLin',
            # constraints
            'ConssDensity', 'RConssInt',
            # spectrum
            'RQRankEig', 'HardEigenPerc',
            # preprocessing
            'prep_RelVarsIncL', 'prep_RelConssIncL', 'prep_RSizesL', 'prep_ConssDensityL'
        ],

    }

    pickle.dump(fts_dict, open(os.path.join(args.save_dict_path, 'fts_subsets.pkl'), 'wb'))
    print("Dictionary of feature subsets saved at \n{}".format(os.path.join(args.save_dict_path, 'fts_subsets.pkl')))
    print("Keys:\n{}".format(fts_dict.keys()))
