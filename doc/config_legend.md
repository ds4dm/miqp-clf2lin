## MIQPs - Configuration files legend

Legend for configuration `.yaml` files entries. 

Configuration files are used to specify a particular learning setting in `04_learn.py`. 
The directory `../configs` contains the configurations used to run experiments which are reported in our paper. 

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
