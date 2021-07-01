"""

miqp-clf2lin
------------

Functions to compute static, preprocessing and root features (cf. doc/features.md)
All features are computed via DataFrame operations, and a new DF is built in compute_fts.

NOTE: the feature computation assumes that
    - raw features
    - information from benchmark and labeling
    - additional columns from data preprocessing (e.g., on convexity)
are already together in the same DF (this is performed by utilities.batch_preprocess in get_features.py).

"""

import numpy as np
import pandas as pd


def general_fts(df_raw, df):
    """
    - sizes ratio: `m / n = rf5 / rf4`
    - ratio of binary vars: `# binaries / n = rf2 / n`
    - ratio of continuous+integer vars: `(# continuous + # integer) / n = (n - rf2) / n`
    """
    df['RSizes'] = df_raw.rf5 / df_raw.rf4
    df['RBin'] = df_raw.rf2 / df_raw.rf4
    df['RCont_RInt'] = (df_raw.rf4 - df_raw.rf2) / df_raw.rf4
    return df


def quadratic_obj_fts(df_raw, df):
    """
    - ratio nnz diagonal binaries: `rf6 / n`
    - ratio nnz diagonal continuous and integers: `(rf7 + rf8) / n`

    - diagonal density (ratio nnz *square* terms): `(rf6 + rf7 + rf8) / n`
    - out-diagonal density (ratio nnz out-diagonal terms): `2 * sum(rf9 + .. + rf14) / (n * (n-1))`
    - density of Q (ratio of all nnz terms): `[(rf6 + rf7 + rf8) + 2 * sum(rf9 + .. + rf14)] / (n * n)`

    - ratio products between binaries (diag and out-diag): `(rf6 + 2 * rf9) / (n * n)`
    - ratio products between continuous and between integers (diag and out-diag): `(rf7 + 2 * rf10 + rf8 + 2 * rf11) / (n * n)`

    - ratio *mixed* products involving binaries (out-diag): `2 * (rf12 + rf13) / (n * (n-1))`
    - ratio *mixed* products involving continuous and integers (out-diag): `2 * (rf12 + rf13 + rf14) / (n * (n-1))`

    - ratio of non-linearizable terms (all cont*cont, diag and out-diag): `(rf7 + 2 * rf10) / (n * n)`
    - ratio of non-linearizable terms wrt all nnz (all cont*cont, diag and out-diag): `(rf7 + 2 * rf10) / (all nnz)`

    - relative size increase of potential linearization (on variables): `(n + rf9 + rf11 + rf12 + rf13 + rf14) / n`
    - relative size increase of potential linearization (on constraints):
    - sizes `m / n` ratio after potential linearization:

    - normalized max degree of binary variables: `rf15 / (n - 1)`
    - normalized max degree of continuous or integer variables: `max(rf18, rf21) / (n - 1)`

    - **!!** averaged 'diagonal dominance' on rows: `rf29`
    - **!!** ratio biggest on smallest abs of diagonal nnz coefficients: `rf26 / rf25`
    - **!!** ratio biggest abs of nnz diagonal on smallest abs nnz Q coefficients: `rf26 / rf27`
    """
    nnz = df_raw.rf6 + df_raw.rf7 + df_raw.rf8 + \
          2 * (df_raw.rf9 + df_raw.rf10 + df_raw.rf11 + df_raw.rf12 + df_raw.rf13 + df_raw.rf14)

    df['RNnzDiagBin'] = df_raw.rf6 / df_raw.rf4
    df['RNnzDiagCont_RNnzDiagInt'] = (df_raw.rf7 + df_raw.rf8) / df_raw.rf4

    df['DiagDensity'] = (df_raw.rf6 + df_raw.rf7 + df_raw.rf8) / df_raw.rf4
    df['OutDiagDensity'] = np.where(df_raw.rf4 > 1,
                                    2 * (df_raw.rf9 + df_raw.rf10 + df_raw.rf11 + df_raw.rf12 + df_raw.rf13 + df_raw.rf14) / (df_raw.rf4 * (df_raw.rf4 - 1)),
                                    0)
    df['QDensity'] = nnz / (df_raw.rf4 * df_raw.rf4)

    df['RBinBin'] = (df_raw.rf6 + 2 * df_raw.rf9) / (df_raw.rf4 * df_raw.rf4)
    rcontcont = (df_raw.rf7 + 2 * df_raw.rf10) / (df_raw.rf4 * df_raw.rf4)
    rintint = (df_raw.rf8 + 2 * df_raw.rf11) / (df_raw.rf4 * df_raw.rf4)
    df['RContCont_RIntInt'] = rcontcont + rintint

    df['RMixedBin'] = np.where(df_raw.rf4 > 1,
                               2 * (df_raw.rf12 + df_raw.rf13) / (df_raw.rf4 * (df_raw.rf4 - 1)),
                               0)
    df['RMixedCont_RMixedInt'] = np.where(df_raw.rf4 > 1,
                                          2 * (df_raw.rf12 + df_raw.rf13 + df_raw.rf14) / (df_raw.rf4 * (df_raw.rf4 - 1)),
                                          0)

    df['RNonLinTerms'] = (df_raw.rf7 + 2 * df_raw.rf10) / (df_raw.rf4 * df_raw.rf4)
    df['RNonLinTermsNnz'] = np.where(nnz != 0, (df_raw.rf7 + 2 * df_raw.rf10) / nnz, 0)

    df['RelVarsLinInc'] = (df_raw.rf4 + df_raw.rf9 + df_raw.rf11 + df_raw.rf12 + df_raw.rf13 + df_raw.rf14) / df_raw.rf4
    df['RelConssLinInc'] = np.where(df_raw.rf5 != 0,
                                    (df_raw.rf5 + 1 * (df_raw.rf9 + df_raw.rf11 + df_raw.rf14) + 4 * (df_raw.rf12 + df_raw.rf13)) / df_raw.rf5,
                                    0)

    df['RLinSizes'] = df['RelConssLinInc'] / df['RelVarsLinInc']

    df['NormMaxDegBin'] = np.where(df_raw.rf4 > 1, df_raw.rf15 / (df_raw.rf4 - 1), 0)
    df['NormMaxDegCont_NormMaxDegInt'] = np.where(df_raw.rf4 > 1, pd.concat([df_raw.rf18, df_raw.rf21], axis=1).max(axis=1) / (df_raw.rf4 - 1), 0)

    df['AvgDiagDom'] = df_raw.rf29
    df['RDiagCoeff'] = np.where(df_raw.rf25 != 0, df_raw.rf26 / df_raw.rf25, 0)  # catch empty diagonal case
    df['ROutDiagCoeff'] = np.where(df_raw.rf27 != 0, df_raw.rf26 / df_raw.rf27, 0)
    return df


def linear_obj_fts(df_raw, df):
    """
    - ratio nnz binaries in linear term: `rf30 / n`
    - ratio nnz continuous and integers in linear term: `(rf31 + rf32) / n`
    - bool: has_linear_term
    - density of linear term: `(rf30 + rf31 + rf32) / n`

    - **!!** ratio biggest on smallest abs of linear coefficients: `abs(rf34 / rf33)`

    """
    df['RNnzBinLin'] = df_raw.rf30 / df_raw.rf4
    df['RNnzContLin_RNnzIntLin'] = (df_raw.rf31 + df_raw.rf32) / df_raw.rf4

    df['HasLinearTerm'] = np.where(df_raw.rf30 + df_raw.rf31 + df_raw.rf32 == 0, 0, 1)
    df['LinDensity'] = (df_raw.rf30 + df_raw.rf31 + df_raw.rf32) / df_raw.rf4

    df['RLinCoeff'] = np.where(df_raw.rf33 != 0, np.abs(df_raw.rf34 / df_raw.rf33), 0)  # contains cases with no linear term
    return df


def constraints_fts(df_raw, df):
    """
    - density of constraints: `(rf35 + rf36 + rf37) / (m * n)`

    - ratio constraints involving binaries: `rf38 / m`
    - ratio constraints involving continuous: `rf39 / m`
    - ratio constraints involving integers: `rf40 / m`

    - **!!** ratio biggest on smallest abs of constraints nnz coefficients: `rf42 / rf41`
    - **!!** ratio magnitudes smallest on biggest abs of rhs nnz coefficients: `abs(rf44 / rf43)`
    """
    df['ConssDensity'] = np.where(df_raw.rf5 != 0,
                                  (df_raw.rf35 + df_raw.rf36 + df_raw.rf37) / (df_raw.rf5 * df_raw.rf4),
                                  0)

    df['RConssBin'] = np.where(df_raw.rf5 != 0, df_raw.rf38 / df_raw.rf5, 0)
    df['RConssCont'] = np.where(df_raw.rf5 != 0, df_raw.rf39 / df_raw.rf5, 0)
    df['RConssInt'] = np.where(df_raw.rf5 != 0, df_raw.rf40 / df_raw.rf5, 0)

    df['RConssCoeff'] = np.where(df_raw.rf41 != 0, df_raw.rf42 / df_raw.rf41, 0)  # contains cases with no constraints
    df['RRhsCoeff'] = np.where(df_raw.rf43 != 0, df_raw.rf44 / df_raw.rf43, 0)  # contains cases with no constraints
    return df


def spectrum_fts(df_raw, df):
    """
    - **!!** trace of Q, normalized over n: `rf57 / n`
    - **!!** spectral norm of Q (corrected): `rf58`
    - rank of Q ratio (or, ratio of nonzero eigenvalues): `(rf52 + rf53) / n`

    - percentage of problematic (hard) eigenvalues in objective: `rf53 / n` if `rf1 == 1 (min)`, `rf52 / n` if `rf1 == -1 (max)`
    - **!!** width of spectrum, averaged with n: `(rf55 - rf56) / n`
    - ratio of positive eigenvalues (corrected): `rf52 / n`
    - ratio of negative eigenvalues (corrected): `rf53 / n`
    - ratio of zero eigenvalues (corrected): `rf54 / n`
    - **!!** ratio magnitudes of min over max abs eigenvalues (corrected): `abs(rf56) / abs(rf55)`

    - ratio of abs difference between original and corrected zero eigenvalues: `abs(rf54 - rf47) / n`
    - abs difference between original and corrected % of problematic eigenvalues in objective:

    """
    df['RQTrace'] = df_raw.rf57 / df_raw.rf4
    df['QSpecNorm'] = df_raw.rf58
    df['RQRankEig'] = (df_raw.rf52 + df_raw.rf53) / df_raw.rf4

    df['HardEigenPerc'] = df_raw.prob_eig_frac  # already computed in first data processing
    df['AvgSpecWidth'] = (df_raw.rf55 - df_raw.rf56) / df_raw.rf4
    df['RPosEigen'] = df_raw.rf52 / df_raw.rf4
    df['RNegEigen'] = df_raw.rf53 / df_raw.rf4
    df['RZeroEigen'] = df_raw.rf54 / df_raw.rf4
    df['RAbsEigen'] = np.where(df_raw.rf55 != 0, np.abs(df_raw.rf56 / df_raw.rf55), 0)

    df['RNZeroEigenDiff'] = np.abs(df_raw.rf54 - df_raw.rf47) / df_raw.rf4
    df['HardEigenPercDiff'] = np.abs(df_raw.prob_eig_frac - df_raw.prob_eig_frac_original)
    df[['HardEigenPerc', 'HardEigenPercDiff']] = df[['HardEigenPerc', 'HardEigenPercDiff']].apply(pd.to_numeric)
    return df


# dynamic features: preprocessing and root node
def root_node_fts(df_raw, df):
    """
    - relative variables increase for L: `Vars_L / n`
    - relative variables increase for NL: `Vars_NL / n`
    - relative constraints increase for L: `Conss_L / m`
    - relative constraints increase for NL: `Conss_NL / m`
    - sizes `m / n` ratio in L: `Conss_L / Vars_L`
    - sizes `m / n` ratio in NL: `Conss_NL / Vars_NL`

    - density of constraints in L: `Nonzs_L / (Conss_L * Vars_L)`
    - density of constraints in NL: `Nonzs_NL / (Conss_NL * Vars_NL)`
    - density of constraints difference between L and NL: `(difference of two above)`
    - relative density of constraints in L wrt original: `above for L / ConssDensity`
    - relative density of constraints in NL wrt original: `above for NL / ConssDensity`

    - difference of total root times (comprises preprocessing): `RtTime_L - RtTime_NL`
    - difference of LP root times: `RLPTime_L - RLPTime_NL`
    - sign of dual bounds at root: 1 if L better, -1 if NL better
    - relative difference of bounds at root:
        `abs(RootDualBound_L - RootDualBound_NL)/(1e-10 + max(abs(RootDualBound_L), abs(RootDualBound_NL)))`
    - signed relative difference of bounds at root
    """
    # preprocessing features
    df['prep_RelVarsIncL'] = df_raw.Vars_L / df_raw.rf4
    df['prep_RelVarsIncNL'] = df_raw.Vars_NL / df_raw.rf4
    df['prep_RelConssIncL'] = np.where(df_raw.rf5 != 0, df_raw.Conss_L / df_raw.rf5, 0)
    df['prep_RelConssIncNL'] = np.where(df_raw.rf5 != 0, df_raw.Conss_NL / df_raw.rf5, 0)
    df['prep_RSizesL'] = df_raw.Conss_L / df_raw.Vars_L
    df['prep_RSizesNL'] = df_raw.Conss_NL / df_raw.Vars_NL

    df['prep_ConssDensityL'] = np.where(df_raw.Conss_L != 0,
                                       df_raw.Nonzs_L / (df_raw.Conss_L * df_raw.Vars_L), 0)
    df['prep_ConssDensityNL'] = np.where(df_raw.Conss_NL != 0,
                                        df_raw.Nonzs_NL / (df_raw.Conss_NL * df_raw.Vars_NL), 0)
    df['prep_ConssDensityDiff'] = df['prep_ConssDensityL'] - df['prep_ConssDensityNL']
    df['prep_RelConssDensityL'] = np.where(df['ConssDensity'] != 0, df['prep_ConssDensityL'] / df['ConssDensity'], 0)
    df['prep_RelConssDensityNL'] = np.where(df['ConssDensity'] != 0, df['prep_ConssDensityNL'] / df['ConssDensity'], 0)

    # root features
    df['root_RtTimeDiff'] = df_raw.RtTime_L - df_raw.RtTime_NL
    df['root_RLPTimeDiff'] = df_raw.RLPTime_L - df_raw.RLPTime_NL

    # about RootDualBound (RDB)
    df.insert(df.shape[1], 'root_SignRDBDiff', None)
    df.insert(df.shape[1], 'root_RelRDBDiff', None)
    for index, row in df_raw.iterrows():
        df.at[index, 'root_RelRDBDiff'] = abs(row.RootDualBound_L - row.RootDualBound_NL) / (
        1e-10 + max(abs(row.RootDualBound_L), abs(row.RootDualBound_NL)))
        # determine best bound between L and NL
        if (row.rf1 == 1) & (row.RootDualBound_L >= row.RootDualBound_NL):
            df.at[index, 'root_SignRDBDiff'] = 1
        elif (row.rf1 == -1) & (row.RootDualBound_L <= row.RootDualBound_NL):
            df.at[index, 'root_SignRDBDiff'] = 1
        elif (row.rf1 == 1) & (row.RootDualBound_NL > row.RootDualBound_L):
            df.at[index, 'root_SignRDBDiff'] = -1
        elif (row.rf1 == -1) & (row.RootDualBound_NL < row.RootDualBound_L):
            df.at[index, 'root_SignRDBDiff'] = -1
        else:
            df.at[index, 'root_SignRDBDiff'] = None
    df['root_RelSignRDBDiff'] = df['root_SignRDBDiff'] * df['root_RelRDBDiff']
    df[['root_SignRDBDiff', 'root_RelRDBDiff', 'root_RelSignRDBDiff']] = df[
        ['root_SignRDBDiff', 'root_RelRDBDiff', 'root_RelSignRDBDiff']].apply(pd.to_numeric)

    return df


def compute_fts(df_raw):
    # create df with same indices as df_raw
    df = pd.DataFrame(index=df_raw.index)
    df['Name'] = df_raw.name
    df['Origin'] = df_raw.origin
    df['Curvature'] = np.where(df_raw.convex == True, 'Convex', 'Nonconvex')

    # compute features
    df = general_fts(df_raw, df)
    df = quadratic_obj_fts(df_raw, df)
    df = linear_obj_fts(df_raw, df)
    df = constraints_fts(df_raw, df)
    df = spectrum_fts(df_raw, df)

    # dynamic features
    df = root_node_fts(df_raw, df)

    # labels, weight and times
    df['BinLabel'] = df_raw.BinLabel
    df['MultiLabel'] = df_raw.MultiLabel
    df['Weight'] = df_raw.Weight
    df['Time_L'] = df_raw.Time_L
    df['Time_NL'] = df_raw.Time_NL

    return df
