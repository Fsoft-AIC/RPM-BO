import os
import argparse
from pathlib import Path
import numpy as np
import pickle
import torch
from models.geometry_unaware_model import GuA_NN_model
import test_functions
from utils import ensure_not_1D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
root_dir = str(Path(os.getcwd()))
    
def run_experiemt(args):
    n_trials = args.trial_itr + args.initial_n

    if "Ackley" in args.test_func:
        prefix_init = root_dir+"/initial_file/Ackley/"
    elif "Hyper" in args.test_func:
        prefix_init = root_dir + "/initial_file/Hyper/"
    filename_init = prefix_init + str(args.test_func) + '_results_CS_d' + str(args.effective_dim) + '_D' + str(args.high_dim) + '_n' + str(
                        args.initial_n) + '_rep_1_' + str(args.rep) + '_seed0' + "_initial"
    
    if os.path.exists(filename_init):
        fileOb = open(filename_init, 'rb')
        initial_points_high = pickle.load(fileOb)  # points in high dimension
        initial_value = pickle.load(fileOb)
        fileOb.close()
    else:
        initial_points_high = np.empty((args.rep, args.initial_n, args.high_dim))
    
    if args.test_func == "Ackley_Mix":
        objective = test_functions.ackley_proposed_mix(args.effective_dim, device)
    elif args.test_func == "Hyper_Mix":
        objective = test_functions.hyper_proposed_mix(args.effective_dim, device)
    
    result_f_s = []
    for i in range(args.rep):
        best_min = 1e8
        proposed_model = GuA_NN_model(d_orig = args.high_dim, d_embedding = args.proj_dim, hidden_units=args.hidden_units, 
                                                    initial_points_list = initial_points_high[i], dtype=dtype, device=device)
        list_value_model= []
        # Perform optimization
        print("Training phase of the proposed model")
        print("Running the " + str(i) + ' restart')
      
        for j in range(n_trials):
            X_queries, X_queries_embedded = proposed_model.select_query_point()

            # Ensure not 1D (i.e. size (D,))
            X_queries = ensure_not_1D(X_queries)

            # Evaluate the batch of query points 1-by-1
            for row_idx in range(len(X_queries)):
                X_query = X_queries[row_idx]
                X_query_embedded = X_queries_embedded[row_idx]

                # Ensure no 1D tensors (i.e. expand tensors of size (D,))
                X_query = ensure_not_1D(X_query) 
                X_query_embedded = ensure_not_1D(X_query_embedded) 

                y_query = -objective.evaluate(X_query) 
                list_value_model.append(-y_query.item())
                if (j % args.update_param == 0):
                    update_param = True
                else:
                    update_param = False
                proposed_model.update(X_query, y_query, update_param)
                if list_value_model[j] < best_min:
                    best_min = list_value_model[j]
                print("RPM-BO: iter = ", j, "maxobj = ", best_min)
            print("---------------------")
        result_f_s.append(list_value_model)
    return result_f_s

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_func', type=str, default="Ackley_Mix", choices=["Ackley_Mix", "Hyper_Mix"])
    parser.add_argument('--rep', type=int, default=20)
    parser.add_argument('--trial_itr', type=int, default=300)
    parser.add_argument('--initial_n', type=int, default=10)
    parser.add_argument('--high_dim', type=int, default=500, choices=[500, 1000, 1500])
    parser.add_argument('--effective_dim', type=int, default=15)
    parser.add_argument('--hidden_units', type=int, default=35)
    parser.add_argument('--proj_dim', type=int, default=15)
    parser.add_argument('--update_param', type=int, default=3)
    args = parser.parse_args()
    run_experiemt(args)
