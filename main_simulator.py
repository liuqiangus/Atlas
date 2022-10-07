from difflib import restore
from re import L
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from bayes_opt import UtilityFunction
from scipy.special import kl_div
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor


import numpy as np
import torch.nn.functional as F
import pickle, time, sys, os, copy
import multiprocessing as mp
import matplotlib.pyplot as plt
sys.path.insert(0,'bayesian_torch')
from bayesian_torch.bnn import BNN
from simulator import Simulator
from functions import *
###############################################################################################################################################
# NOTE:  change the PBOUNDS keys manually if you want different number of parameters to be learned

###############################################################################################################################################

def select_next_queries(utils, times=1):
    action = Action() # create a default action
    next_points = []
    
    for t in range(times):
        for key in utils.keys():
            action.modify(**optimizer.suggest(utils[key])) # here, ** is for dynamic inputs
            next_points.append(copy.deepcopy(action))
    
    return next_points

def run_parallel_with_multiprocessing(actions):

    pool = mp.Pool(len(actions))
    results = pool.map(simulator.run_with_sim_para, actions)
    pool.close() 

    return results

###############################################################################################################################################

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='BNN')
parser.add_argument('--aq', type=str, default='ts')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--numUEs', type=int, default=1)
parser.add_argument('--start', type=int, default=100)
parser.add_argument('--percent', type=float, default=0.2)
parser.add_argument('--simtime', type=int, default=30)                  
parser.add_argument('--weight', type=int, default=7)
parser.add_argument('--parallel', type=int, default=16)
parser.add_argument('--seed', type=int, default=1111)
args = parser.parse_args()

assert(args.model in ['GP', 'BNN'])
if args.model == "GP" and args.aq == 'ts': args.aq = 'ei' # attention XXX TODO GP only use EI acq func for baseline 
if args.aq != 'ts' or args.model != 'BNN': args.parallel = 1 # if not ts or not BNN, then every time sample/suggest one action
if args.parallel >= mp.cpu_count(): args.parallel = mp.cpu_count() - 1 # raise ValueError ("Reduce parallel multiprocessing cpu counts!")
assert(args.aq in ['ei', 'ucb', 'pi', 'gpucb', 'ts',])
assert(args.seed != 0) # NS3 does not allow 0

print('MAKE SURE YOU UNDERSTAND THE CODES, AS THERE ARE SOME HARDCODES!')
print('-'*100)
print('simulation with', 'model:', args.model, ', aq:', args.aq, ', epoch:', args.epochs, ', numUEs:', args.numUEs, ', start:', args.start, ', parallel:', args.parallel, ', weight:', args.weight, ', percent:', args.percent, ', seed:', args.seed)

###############################################################################################################################################

simulation_start_time = time.time()
simulator = Simulator(
    simtime=args.simtime, 
    numUEs=args.numUEs,
    seed=args.seed) # the same seed for comparison

PBOUNDS = {
            'loading_time_offset':       (0,                                             int(100*args.percent)),
            'compute_time_mean_offset':  (0,                                             int(100*args.percent)),
            'baseline_loss':             (simulator.baseline_loss*(1-args.percent),      simulator.baseline_loss*(1+args.percent)),
            'backhaul_offset':           (0,                                             int(100*args.percent)), # may reduce this scale to
            'backhaul_delay':            (0,                                             int(50*args.percent)),
            'enb_noise_figure':          (simulator.enb_noise_figure*(1-args.percent),   simulator.enb_noise_figure*(1+args.percent)),
            'ue_noise_figure':           (simulator.ue_noise_figure*(1-args.percent),    simulator.ue_noise_figure*(1+args.percent)),
            # 'enb_antenna_gain':          (simulator.enb_antenna_gain*(1-args.percent),   simulator.enb_antenna_gain*(1+args.percent)),
            # 'enb_tx_power':              (simulator.enb_tx_power*(1-args.percent),       simulator.enb_tx_power*(1+args.percent)),
            # 'ue_antenna_gain':           (simulator.ue_antenna_gain*(1-args.percent),    simulator.ue_antenna_gain*(1+args.percent)),
            # 'ue_tx_power':               (simulator.ue_tx_power*(1-args.percent),        simulator.ue_tx_power*(1+args.percent)),
            # 'edge_bw':                   (simulator.edge_bw*(1-args.percent),            simulator.edge_bw*(1+args.percent)),
            # 'edge_delay':                (0,                                int(50*args.percent)),
            # 'compute_time_std_offset':   (0,                                int(25*args.percent)),
}

DIM = len(PBOUNDS)

REFERENCE = pickle.load(open("app_eval/real_indiv/"+ "measurement_system_performance_slice_0_traffic_"+str(args.numUEs)+".pickle", "rb" ))['performance']

kappa, xi = 2.5, 0.01

All_Utilities = {
    'ei':     UtilityFunction(kind="ei",    kappa=kappa, xi=xi, dim=DIM),
    'pi':     UtilityFunction(kind="poi",   kappa=kappa, xi=xi, dim=DIM),
    'ucb':    UtilityFunction(kind="ucb",   kappa=kappa, xi=xi, dim=DIM),
    'gpucb':  UtilityFunction(kind="gpucb", kappa=kappa, xi=xi, dim=DIM),
    'ts':     UtilityFunction(kind="ts",    kappa=kappa, xi=xi, dim=DIM),
}

utility = {args.aq: All_Utilities[args.aq]}  # get the utility function, only one utility 

util_name = '_'.join(sorted(utility))

if DIM == 1:
    key = list(PBOUNDS.keys())[0]
    savename = os.getcwd()+"/results/simulator/offline_simulator_log_"+str(DIM)+"dim_"+args.model+"_weight_"+str(args.weight)+"_epoch_"+str(args.epochs)+"_numUEs_"+str(args.numUEs)+"_start_"+str(args.start)+"_"+util_name+"_parallel_"+str(args.parallel)+"_percent_"+str(int(args.percent*100))+"_para_"+key+".json"
else:
    savename = os.getcwd()+"/results/simulator/offline_simulator_log_"+str(DIM)+"dim_"+args.model+"_weight_"+str(args.weight)+"_epoch_"+str(args.epochs)+"_numUEs_"+str(args.numUEs)+"_start_"+str(args.start)+"_"+util_name+"_parallel_"+str(args.parallel)+"_percent_"+str(int(args.percent*100))+".json"

# optimizer = pickle.load(open('results/simulator/offline_simulator_'+savename.split('/')[-1]+'_model.pkl', 'rb'))

print("simulation parameters are: ", PBOUNDS)
print("simulation savename is: ", savename)
print('-'*100)

###############################################################################################################################################

if args.model == 'GP':        
    model = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True, n_restarts_optimizer=5, random_state=args.seed,)
elif args.model == 'BNN':
    model = BNN(input_dim=DIM, seed=args.seed, lr=1.0, gamma=0.9996, activation=F.relu, inverse_y=True) # attention, inverse_y make sure positive value for training under relu activation func,  0.996 for 400, 0.9996 for 4000~6000, scheduler is good, but batch queries means time 10~16, so one more scale
else:
    raise ValueError('make sure mode is BNN or GP')

optimizer = BayesianOptimization(
    model=model,
    f=None,
    pbounds=PBOUNDS,
    verbose=2, 
    random_state=args.seed,
)

# New optimizer is loaded with previously seen points
# try: 
#     import shutil
#     shutil.copy(savename, savename.split('.')[0]+savename.split('.')[1]+"_backup."+savename.split('.')[-1])
#     load_logs(optimizer, logs=[savename])
#     print('successfully loaded previous dataset from', savename)
# except:  
#     pass

saver = JSONSaver(path="offline_simulator_dataset_numUEs_"+str(args.numUEs)+".json")

logger = JSONLogger(path=savename)
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

###############################################################################################################################################

num_start_points = args.start
RESULTS = []
start_points = {'next_points':[], 'targets':[]}


def retrieve_results_from_dataset(actions):

    results = []

    for action in actions:
        action_dict = action.to_dict() # convert to dict as it is the input of BNN/GP here, rather than an action object
        next_point = {key:val for key, val in action_dict.items() if key in sorted(PBOUNDS)}

        new_key = generate_key_from_dict(next_point)
        if new_key in saver._data.keys():
            results.append(saver._data[new_key])

    return results

######## main training loop for BO with BNN ############
for ite in range(int(len(optimizer.res)/args.parallel), args.epochs):

    start_time = time.time()

    actions = select_next_queries(utility, times=args.parallel) # if utility is one acq func, then it is one

    results = retrieve_results_from_dataset(actions) # try to get  results from dataset

    if (len(actions) != len(results)):
        # target = black_box_function(**next_point)
        results = run_parallel_with_multiprocessing(actions)
        # results = [simulator.run_with_sim_para(actions[0])]

    assert(len(actions) == len(results)) # make sure the multiprocessing works properly

    if ite == num_start_points: # when exactly is the start point, register these previously saved points into the optimizer buffer
        for i in range(len(start_points['next_points'])):
            optimizer.register(params=start_points['next_points'][i], target=start_points['targets'][i])

    targets, kls, act_dists = [], [], []
    # calculate the target and store in the BO _space (buffer)
    for idx in range(len(results)): 
        action_dict = actions[idx].to_dict() # convert to dict as it is the input of BNN/GP here, rather than an action object
        next_point = {key:val for key, val in action_dict.items() if key in sorted(PBOUNDS)} # remove the other non-optimizing keys
        performance = results[idx]['performance'] # the performance
        kl = calculate_kl_divergence (performance, REFERENCE)
        act_dist = calculate_conf_distance(next_point, PBOUNDS, args.percent)
        target = - kl - args.weight * act_dist  # maxmize, so minus KL divergence
            
        # store temp, save or others
        if ite < num_start_points: # if start point, then temp save them
            start_points['next_points'].append(next_point)
            start_points['targets'].append(target)
        else: 
            optimizer.register(params=next_point, target=target) 

        targets.append(target)
        kls.append(kl)
        act_dists.append(act_dist)
        RESULTS.append([ite, next_point, performance, target, kl, act_dist]) # store the progress

        saver.update({"next_point":next_point, "performance":list(performance), "kl":kl, "act_dists":act_dist, "target":target})

    print("\nite:", ite, "avg. target:", np.mean(targets), "avg. kl:", np.mean(kls), "avg. act_dist:", np.mean(act_dists),"used time:", time.time() - start_time)

###############################################################################################################################################

print('optimal is ', optimizer.max)

# for i, res in enumerate(optimizer.res): print("Iteration {}: \t{}".format(i, res))    

with open('results/simulator/offline_simulator_'+savename.split('/')[-1]+'_progress.pkl', 'wb') as file:
    pickle.dump(RESULTS, file)

with open('results/simulator/offline_simulator_'+savename.split('/')[-1]+'_model.pkl', 'wb') as file:
    pickle.dump(optimizer, file)

###############################################################################################################################################

y = [r[3] for r in RESULTS]
dist_mat = np.reshape(y,(args.epochs, args.parallel))
means = np.mean(dist_mat, axis=1)
tops = np.max(dist_mat, axis=1)
buttons = np.min(dist_mat, axis=1)
x = np.arange(len(means))

plt.fill_between(x, tops, buttons,color='lightgray')
plt.plot(x, means, color='C0')
plt.savefig('results/simulator/Bayesian_optimziation_'+savename.split('/')[-1]+'_training_progress.pdf', format = 'pdf', dpi=300)

print('done with time ', time.time() - simulation_start_time, ' seconds')

