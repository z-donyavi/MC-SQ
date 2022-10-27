import argparse
import numpy as np
import pandas as pd
import helpers
from time import localtime, strftime
from getScores import *
from Ensemble_Quantifier import *


# ==============================================================================
# Global Variables
# ==============================================================================

res_path = "results/raw/"

# global data set index
data_set_index = pd.read_csv("data/data_index.csv",
                             sep=";",
                             index_col="dataset")

# global algorithm index
algorithm_index = pd.read_csv("alg_index.csv",
                              sep=";",
                              index_col="algorithm")


algorithms = list(algorithm_index.index)


global_seeds = [4711, 1337, 42, 90210, 666, 879, 1812, 4055, 711, 512]


# train/test ratios to test against
train_test_ratios = [[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3]]
train_test_ratios = [np.array(d) for d in train_test_ratios]

train_distributions = dict()
train_distributions[2] = np.array([[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1], [0.95, 0.05]])
train_distributions[3] = np.array([[0.2, 0.5, 0.3], [0.05, 0.8, 0.15], [0.35, 0.3, 0.35]])
train_distributions[4] = np.array([[0.5, 0.3, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]])
train_distributions[5] = np.array([[0.05, 0.2, 0.1, 0.2, 0.45], [0.05, 0.1, 0.7, 0.1, 0.05], [0.2, 0.2, 0.2, 0.2, 0.2]])


test_distributions = dict()
test_distributions[2] = np.array(
    [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1],
      [0.95, 0.05], [0.99, 0.01], [1, 0]])
test_distributions[3] = np.array(
    [[0.1, 0.7, 0.2], [0.55, 0.1, 0.35], [0.35, 0.55, 0.1], [0.4, 0.25, 0.35], [0., 0.05, 0.95]])
test_distributions[4] = np.array(
    [[0.65, 0.25, 0.05, 0.05], [0.2, 0.25, 0.3, 0.25], [0.45, 0.15, 0.2, 0.2], [0.2, 0, 0, 0.8],
     [0.3, 0.25, 0.35, 0.1]])
test_distributions[5] = np.array(
    [[0.15, 0.1, 0.65, 0.1, 0], [0.45, 0.1, 0.3, 0.05, 0.1], [0.2, 0.25, 0.25, 0.1, 0.2], [0.35, 0.05, 0.05, 0.05, 0.5],
     [0.05, 0.25, 0.15, 0.15, 0.4]])

mc_data = data_set_index.loc[data_set_index.loc[:, "classes"] > 2].index


def parse_args():
    """Parses arguments given to script

    Returns:
        dict-like object -- Dict-like object containing all given arguments
    """
    parser = argparse.ArgumentParser()

    # Test parameters
    parser.add_argument(
        "-a", "--algorithms", nargs="+", type=str,
        choices=algorithms, default=algorithms,
        help="Algorithms used in evaluation."
    )
    parser.add_argument(
        "-d", "--datasets", nargs="*", type=str,
        default=None,
        help="Datasets used in evaluation."
    )
    parser.add_argument(
        "--mc", type=int, default=None,
        help="Running experiments on the all binary or multi-class datasets. "
    )
    parser.add_argument(
        "--cal", type=bool, default=False,
        help="Wheter or not running the experiments with calibrated scores. "
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=global_seeds,
        help="Seeds to be used in experiments. By default, all seeds will be used."
    )
    parser.add_argument(
        "--dt", type=int, nargs="+", default=None,
        help="Index for train/test-splits to be run."
    )
    return parser.parse_args()


def run_synth(data_sets=None,
              algs=None,
              dt_index=None,
              b_mc=None,
              cal=False,
              seeds=global_seeds):
    if data_sets is None:
        df_ind = data_set_index
    else:
        df_ind = data_set_index.loc[data_sets]

    if dt_index is None:
        dt_ratios = train_test_ratios
    else:
        dt_ratios = [train_test_ratios[i] for i in dt_index]

    if b_mc is None:
        data_sets = list(df_ind.index)
    if b_mc == 0:
        df_ind = df_ind.loc[df_ind["classes"] == 2]
        data_sets = list(df_ind.index)
    if b_mc == 1:
        df_ind = df_ind.loc[df_ind["classes"] > 2]
        data_sets = list(df_ind.index)
        
    if algs is None:
        alg_ind = algorithms
    else:
        alg_ind = algs



    for dta_name in data_sets:

        n_classes = df_ind.loc[dta_name, "classes"]

        # build training and test class distributions
        train_ds = train_distributions[n_classes]

        test_ds = test_distributions[n_classes]
        
        columns = [al for al in algs]
        columns.insert(0, 'seed')
        All_AEs = pd.DataFrame(columns = columns)

        for seed in seeds:

            # ----run on unbinned data -----------------------

               Mean_AEs = data_synth_experiments(dta_name, binned=False, algs=alg_ind, dt_ratios=dt_ratios, train_ds=train_ds,
                                       test_ds=test_ds, cal=cal, seed=seed)
               All_AEs = All_AEs.append(pd.Series(Mean_AEs, index=All_AEs.columns[:len(Mean_AEs)]), ignore_index=True)
                
        fnameres = res_path + dta_name + "_All_AEs_" + strftime("%Y-%m-%d_%H-%M-%S", localtime()) + ".csv"
        All_AEs.to_csv(fnameres, index=False, sep=',')

def data_synth_experiments(
        dta_name,
        binned,
        algs,
        dt_ratios,
        train_ds,
        test_ds,
        cal,
        seed=4711):
    if len(algs) == 0 or len(dt_ratios) == 0 or len(train_ds) == 0 or len(test_ds) == 0:
        return

    print(dta_name)
    X, y, N, Y, n_classes, y_cts, y_idx = helpers.get_xy(dta_name, load_from_disk=False, binned=binned)

# transform class labels to a range between 0 and n-1    
    y = class2index(y, tuple(Y)) 
    y_cts = np.unique(y, return_counts=True)
    Y = y_cts[0]
    y_cts = y_cts[1]
        
    n_combs = len(dt_ratios) * len(train_ds) * len(test_ds)
    n_cols = 5 + 4 * n_classes + n_classes * len(algs)+ ((n_classes+1) * len(algs))

    stats_matrix = np.zeros((n_combs, n_cols))
    single_quantifiers = ['HDy', 'EMQ', 'GAC', 'GPAC', 'FM' ]

    i = 0

    for dt_distr in dt_ratios:

        for train_distr in train_ds:

            for test_distr in test_ds:
                
                print('Training and Test dists:')
                print(dt_distr)
                print(train_distr)
                print(test_distr)

                train_index, test_index, stats_vec = helpers.synthetic_draw(N, n_classes, y_cts, y_idx, dt_distr,
                                                                            train_distr, test_distr, seed)
                
                X_train, y_train = X[train_index], y[train_index]
                X_test = X[test_index]
                
                #get train and test scores from a bag of classifires
                
                if cal: #getting scores with calibration
                    train_scores,  test_scores, nmodels = getCalibratedScores(X_train, X_test, y_train, len(Y), algs)
                else: #getting scores without calibration
                    train_scores, test_scores, nmodels = getScores(X_train, X_test, y_train, len(Y), algs)
                    

                
                j = len(stats_vec)
                stats_matrix[i, 0:j] = stats_vec

                for str_alg in algs:
                    print(str_alg)
                    if str_alg in single_quantifiers:
                        tr_scores = train_scores[0]
                        ts_scores = test_scores[0]
                        nmodels = 1
                    else:
                        tr_scores = train_scores
                        ts_scores = test_scores
                        nmodels = 7
                   
                    p = run_setup_Ensemble(tr_scores, ts_scores, y_train, nmodels, len(Y), str_alg)
                    print(p)
                    stats_matrix[i, j:(j + n_classes)] = p
                    j += n_classes
                    
                for AE_QF in range(len(algs)):
                    for n in range(n_classes):
                        stats_matrix[i, j] = abs(stats_matrix[i, len(stats_vec)-n_classes+n]-stats_matrix[i, len(stats_vec)+(AE_QF*n_classes)+n])
                        j += 1
                    stats_matrix[i, j] = sum(stats_matrix[i, j-n_classes: j])
                    j += 1
                i += 1

    col_names = ["Total_Samples_Used", "Training_Size", "Test_Size", "Training_Ratio", "Test_Ratio"]
    col_names += ["Training_Class_" + str(l) + "_Absolute" for l in Y]
    col_names += ["Training_Class_" + str(l) + "_Relative" for l in Y]
    col_names += ["Test_Class_" + str(l) + "_Absolute" for l in Y]
    col_names += ["Test_Class_" + str(l) + "_Relative" for l in Y]

    for alg in algs:
        for li in Y:
            col_names += [alg + "_Prediction_Class_" + str(li)]
            
    for alg in algs:
        for li in Y:
            col_names += [alg + "_AE_Class_" + str(li)]
        col_names += [alg + "_Total_AE"]

    stats_data = pd.DataFrame(data=stats_matrix,
                              columns=col_names)
    Mean_AEs = list(stats_data[[alg+'_Total_AE' for alg in algs]].mean())
    Mean_AEs.insert(0, seed)
    
    
    fname = res_path + dta_name + "_seed_" + str(seed) + "_" + strftime("%Y-%m-%d_%H-%M-%S", localtime()) + ".csv"
    stats_data.to_csv(fname, index=False, sep=',')
    return(Mean_AEs)


def run_setup_Ensemble(train_scores, test_scores, y_train, nmodels, nclasses, QF):

    
    if QF == 'EMQ':
        p = EMQ(test_scores, y_train, nclasses)
        
    # elif QF == 'DyS':
    #     p = EnsembleDyS(train_scores, test_scores, y_train, nmodels, nclasses)  
    
    else:
        p = eval(QF)(train_scores, test_scores, y_train, nmodels, nclasses)


    return p

if __name__ == "__main__":
    args = parse_args()
    run_synth(args.datasets, args.algorithms, args.dt, args.mc, args.cal, args.seeds)
