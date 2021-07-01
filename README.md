# A Classifier to Decide on the Linearization of MIQPs in CPLEX
Pierre Bonami, Andrea Lodi and Giulia Zarpellon

---

This code accompanies our paper ***A Classifier to Decide on the Linearization of Mixed-Integer Quadratic Problems in CPLEX***.

In this project, we 
+ translate the algorithmic question of whether to linearize convex Mixed-Integer Quadratic Programming problems (MIQPs) 
  into a classification task, and use machine learning (ML) techniques to tackle it;
+ integrate the obtained prediction function in the commercial optimization solver CPLEX (v12.10.0) to decide on MIQPs linearization.

To achieve these goals, we
+ represent MIQPs and the linearization decision by careful target and feature engineering;
+ design learning experiments and evaluation metrics to incorporate optimization knowledge in the ML pipeline.

Below we provide links and instructions to our paper, source code and datasets.

---

## Paper preprint

* [Optimization Online version](http://www.optimization-online.org/DB_HTML/2020/03/7662.html) 

This work is an extension of our paper 
[Learning a Classification of Mixed-Integer Quadratic Programming Problems](https://link.springer.com/chapter/10.1007/978-3-319-93031-2_43), 
presented at CPAIOR 2018.

Please use the following BibTeX to cite our paper: 

```
@article{BonamiLodiZarpellon20,
  title={A Classifier to Decide on the Linearization of Mixed-Integer Quadratic Problems in {CPLEX}},
  author={Bonami, Pierre and Lodi, Andrea and Zarpellon, Giulia},
  journal={Optimization Online preprint},
  url={http://www.optimization-online.org/DB_HTML/2020/03/7662.html},
  year={2020}
}
```

## Navigating the repository

### `/src`

Contains all Python scripts and code. In particular, 

+ `01_get_benchmark_data.py` reads files produced during MIQP benchmarking to create a DataFrame. 
  Data is subsequently cleaned-up, with consistency and solvability checks applied. 
  Finally, labeling procedures and aggregations over seeds are executed.
  
  **Note**: this script *cannot* be run without proprietary scripts that read benchmark results and compute labels, 
  but we include it for completeness.
  
+ `02_get_raw_features.py` extracts raw attributes from instances in the datasets, as defined in `raw_features.py`
  (refer to `/doc/raw_features.md` for the list of 58 raw attributes).
  
  **Note**: only *static* attributes are extracted, as dynamic features are computed using information from the benchmark, 
  the same used for labeling.

  **Note**: this script requires the CPLEX Python API to read the MIQP instances, but no optimization takes place.

+ `03_get_features.py` joins different batches of data and compute static, preprocessing and root features, as defined in 
  `features.py` (refer to `/doc/features.md` for the list of 60 features).
  
+ `04_learn.py` performs ML on the MIQP data, including regression, feature selection and classification. 
  Main parameters are passed via a `.yaml` file containing the desired configuration for the experiment (see `/configs` and 
  `/doc/config_legend.md`). 
  Experiments are performed with scikit-learn, and managed via the custom class `LearningExperiment` (see `experiment.py`). 
  Models and hyper-parameters are instead specified in `models.py`.
  
+ `utilities.py` and `feature_subsets.py` contain useful routines and the definition of the feature subsets used in our 
  experiments (`/data/fts_subsets.pkl`).
  
Each script contains additional relevant documentation. 

### `/doc`

Basic documentation of raw and processed features, as well as a legend for configuration files. 

### `/data`

Contains the annotated dataset to run ML experiments, as well as intermediate data files for `setD`. See below for more information.

### `/instances`

Contains MIQP models of `setD` and `neos` datasets. See below for more information.

### `/configs`

Configuration files to run learning experiments with `04_learn.py`. For a legend, see `/doc/config_legend.md`.
We include the configurations that were used to run the experiments reported in the paper.


## Data

We work with three datasets of MIQP instances.
+ `setD` is a synthetic dataset which was generated for our first paper. It contains 2640 models, of which 1821 make it 
  to the final dataset. These instances have been contributed to the [MINOA open-source benchmark library](https://minoa-itn.fau.de/?page_id=749), 
  and can also be found in `/instances/setD`.
+ `neos` contains MIQPs that were submitted to the NEOS server between April 2015 and January 2018, with CPLEX as the specified solver. 
  No further selection was done except cleaning duplicates. The 480 models in the final dataset can be found in 
  `/instances/neos`.
+ `miqp` is the CPLEX internal MIQP benchmark, which we cannot open source. While some models in this benchmark are proprietary, 
  note that instances from the scientific literature are also included in this set.
  
The entire (anonymized) annotated dataset (2585 samples, each one annotated with 60 features, labels and weight) can be
found in `/data/anonym_all_data_features_times.pkl`.
  
## Dependencies 

Environment specifications can be found in `environment.yml`. Note that the CPLEX Python API is needed to read instances
when computing raw features (`02_get_raw_features.py`), but not to run ML experiments 
(this is because optimization benchmarks for L and NL were run separately).

## Questions?

Please feel free to submit a GitHub issue if you have any questions or find any bugs. 
We do not guarantee any support, but will do our best if we can help.
