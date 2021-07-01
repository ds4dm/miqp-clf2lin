## MIQPs - Features List 

Processed features, derived from 58 raw features `rf` (in `raw_features.md`) + (dyn).
 
Features marked with "**!!**" could be numerically unstable.


**General**

- `RSizes`: sizes ratio: `m / n = rf5 / rf4`
- `RBin`: ratio of binary vars: `# binaries / n = rf2 / n`
- `RCont_RInt`: ratio of continuous+integer vars: `(# continuous + # integer) / n = (n - rf2) / n`


**Quadratic objective**

- `RNnzDiagBin`: ratio nnz diagonal binaries: `rf6 / n`
- `RNnzDiagCont_RNnzDiagInt`: ratio nnz diagonal continuous and integers: `(rf7 + rf8) / n`
- `DiagDensity`: diagonal density (ratio nnz *square* terms): `(rf6 + rf7 + rf8) / n`
- `OutDiagDensity`: out-diagonal density (ratio nnz out-diagonal terms): `2 * sum(rf9 + .. + rf14) / (n * (n-1))`
- `QDensity`: density of Q (ratio of all nnz terms): `[(rf6 + rf7 + rf8) + 2 * sum(rf9 + .. + rf14)] / (n * n)`
- `RBinBin`: ratio products between binaries (diag and out-diag): `(rf6 + 2 * rf9) / (n * n)`
- `RContCont_RIntInt`: ratio products between continuous and between integers (diag and out-diag): `(rf7 + 2 * rf10 + rf8 + 2 * rf11) / (n * n)`
- `RMixedBin`: ratio *mixed* products involving binaries (out-diag): `2 * (rf12 + rf13) / (n * (n-1))`
- `RMixedCont_RMixedInt`: ratio *mixed* products involving continuous and integers (out-diag): `2 * (rf12 + rf13 + rf14) / (n * (n-1))`
- `RNonLinTerms`: ratio of non-linearizable terms (all cont*cont, diag and out-diag): `(rf7 + 2 * rf10) / (n * n)`
- `RNonLinTermsNnz`: ratio of non-linearizable terms wrt all nnz (all cont*cont, diag and out-diag): `(rf7 + 2 * rf10) / (all nnz)`
- `RelVarsLinInc`: relative size increase of potential linearization (on variables): `(n + rf9 + rf11 + rf12 + rf13 + rf14) / n`
- `RelConssLinInc`: relative size increase of potential linearization (on constraints): `(m + 1*(rf9 + rf11 + rf14) + 4*(rf12 + rf13)) / m`
- `RLinSizes`: sizes `m / n` ratio after potential linearization: `(ratio of previous two)`
- `NormMaxDegBin`: normalized max degree of binary variables: `rf15 / (n - 1)`
- `NormMaxDegCont_NormMaxDegInt`: normalized max degree of continuous or integer variables: `max(rf18, rf21) / (n - 1)`
- **!!** `AvgDiagDom`: averaged 'diagonal dominance' on rows: `rf29`
- **!!** `RDiagCoeff`: ratio biggest on smallest abs of diagonal nnz coefficients: `rf26 / rf25`
- **!!** `ROutDiagCoeff`: ratio biggest abs of nnz diagonal on smallest abs nnz Q coefficients: `rf26 / rf27`


**Linear objective**

- `RNnzBinLin`: ratio nnz binaries in linear term: `rf30 / n`
- `RNnzContLin_RNnzIntLin`: ratio nnz continuous and integers in linear term: `(rf31 + rf32) / n`
- `HasLinearTerm`: boolean, whether a linear term is specified in the formulation or not
- `LinDensity`: density of linear term: `(rf30 + rf31 + rf32) / n`
- **!!** `RLinCoeff`: ratio biggest on smallest abs of linear coefficients: `abs(rf34 / rf33)`


**Constraints**

- `ConssDensity`: density of constraints: `(rf35 + rf36 + rf37) / (m * n)`
- `RConssBin`: ratio constraints involving binaries: `rf38 / m`
- `RConssCont`: ratio constraints involving continuous: `rf39 / m`
- `RConssInt`: ratio constraints involving integers: `rf40 / m`
- **!!** `RConssCoeff`: ratio biggest on smallest abs of constraints nnz coefficients: `rf42 / rf41`
- **!!** `RRhsCoeff`: ratio magnitudes smallest on biggest abs of rhs nnz coefficients: `abs(rf44 / rf43)`


**Spectrum**

*Note: some features regarding eigenvalues will not be precise, due to the cut at max 500 eigenvalues extracted.* 

- **!!** `RQTrace`: trace of Q, normalized over n: `rf57 / n`
- **!!** `QSpecNorm`: spectral norm of Q (corrected): `rf58`
- `RQRankEig`: rank of Q ratio (or, ratio of nonzero eigenvalues): `(rf52 + rf53) / n`
- `HardEigenPerc`: percentage of problematic (hard) eigenvalues in objective: `rf53 / n` if `rf1 == 1 (min)`, `rf52 / n` if `rf1 == -1 (max)`
- **!!** `AvgSpecWidth`: width of spectrum, averaged with n: `(rf55 - rf56) / n`
- `RPosEigen`: ratio of positive eigenvalues (corrected): `rf52 / n`
- `RNegEigen`: ratio of negative eigenvalues (corrected): `rf53 / n`
- `RZeroEigen`: ratio of zero eigenvalues (corrected): `rf54 / n`
- **!!** `RAbsEigen`: ratio magnitudes of min over max abs eigenvalues (corrected): `abs(rf56) / abs(rf55)`
- `RNZeroEigenDiff`: ratio of abs difference between original and corrected zero eigenvalues: `abs(rf54 - rf47) / n`
- `HardEigenPercDiff`: abs difference between original and corrected % of problematic eigenvalues in objective: 


**Preprocessing**

- `prep_RelVarsIncL`: relative variables increase for L: `Vars_L / n`
- `prep_RelVarsIncNL`: relative variables increase for NL: `Vars_NL / n`
- `prep_RelConssIncL`: relative constraints increase for L: `Conss_L / m`
- `prep_RelConssIncNL`: relative constraints increase for NL: `Conss_NL / m`
- `prep_RSizesL`: sizes `m / n` ratio in L: `Conss_L / Vars_L`
- `prep_RSizesNL`: sizes `m / n` ratio in NL: `Conss_NL / Vars_NL`
- `prep_ConssDensityL`: density of constraints in L: `Nonzs_L / (Conss_L * Vars_L)`
- `prep_ConssDensityNL`: density of constraints in NL: `Nonzs_NL / (Conss_NL * Vars_NL)`
- `prep_ConssDensityDiff`: density of constraints difference between L and NL: `(difference of two above)`
- `prep_RelConssDensityL`: relative density of constraints in L wrt original: `above for L / ConssDensity`
- `prep_RelConssDensityNL`: relative density of constraints in NL wrt original: `above for NL / ConssDensity`


**Root node**

- `root_RtTimeDiff`: difference of total root times (comprises preprocessing): `RtTime_L - RtTime_NL`
- `root_RLPTimeDiff`: difference of LP root times: `RLPTime_L - RLPTime_NL`
- `root_SignRDBDiff`: sign of dual bounds at root: 1 if L better, -1 if NL better
- `root_RelRDBDiff`: relative difference of bounds at root: 
    `abs(RootDualBound_L - RootDualBound_NL)/(1e-10 + max(abs(RootDualBound_L), abs(RootDualBound_NL)))`
- `root_RelSignRDBDiff`: signed relative difference of bounds at root: `root_SignRDBDiff * root_RelRDBDiff`

