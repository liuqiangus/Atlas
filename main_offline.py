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

def select_next_queries(utils, times=1):
    resource = Resource() # create a default action
    next_points = []
    
    for t in range(times):
        for key in utils.keys():
            resource.modify(**optimizer.suggest(utils[key])) # here, ** is for dynamic inputs XXX  fn is to add the usage as the calcuation of aq
            next_points.append(copy.deepcopy(resource))
    
    return next_points

def run_parallel_with_multiprocessing(resources):

    pool = mp.Pool(len(resources))
    results = pool.map(simulator.run_with_res_para, resources)
    pool.close() 

    return results

def calculate_ymax(X, Y, weight, availability, bounds):
    usage = np.array([np.mean(np.divide(x_try, bounds[:, 1])) for x_try in X])
    values = weight * (Y - availability) - usage
    idx = np.argmax(values)
    ymax = values[idx]

    return ymax
###############################################################################################################################################

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--program', type=str, default='main-mar.cc')
parser.add_argument('--model', type=str, default='BNN')
parser.add_argument('--aq', type=str, default='ts_offline')
parser.add_argument('--numUEs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--start', type=int, default=100)
parser.add_argument('--simtime', type=int, default=30)                  
parser.add_argument('--parallel', type=int, default=15)
parser.add_argument('--threshold', type=float, default=0.9999) # 500 ms for MAR slice
parser.add_argument('--availability', type=float, default=0.90)
parser.add_argument('--weight', type=float, default=10.0)
parser.add_argument('--seed', type=int, default=1111)
args = parser.parse_args()

assert(args.program in ['main-mar.cc', 'main-hvs.cc', 'main-iot.cc'])
assert(args.model in ['GP', 'BNN'])
if args.model == "GP" and args.aq == 'ts_offline': args.aq = 'ei_offline' # attention XXX TODO GP only use EI acq func for baseline
if args.aq != 'ts_offline' or args.model != 'BNN': args.parallel = 1 # if not ts or not BNN, then every time sample/suggest one action
assert(args.aq in ['ei_offline', 'ucb_offline', 'pi_offline', 'gpucb_offline', 'ts_offline',])
assert(args.seed != 0) # NS3 does not allow 0

print('MAKE SURE YOU UNDERSTAND THE CODES, AS THERE ARE SOME HARDCODES!')
print('-'*100)
print('simulation with', 'model:', args.model, ', aq:', args.aq, ', numUEs:', args.numUEs, ', epochs:', args.epochs,', start:', args.start, ', parallel:', args.parallel, ', threshold:', args.threshold,  ', seed:', args.seed)

###############################################################################################################################################
# set the optimal value to the simulator
from parameters import OPTIMAL_PARA_SIM

CONF = OPTIMAL_PARA_SIM
experiment_start_time = time.time()
simulator = Simulator(
            program=args.program,
            simtime = args.simtime,
            numUEs = args.numUEs,
            loading_time_offset = CONF['loading_time_offset'],
            compute_time_mean_offset = CONF['compute_time_mean_offset'],
            baseline_loss = CONF['baseline_loss'],
            backhaul_offset = CONF['backhaul_offset'],
            backhaul_delay = CONF['backhaul_delay'],
            enb_noise_figure = CONF['enb_noise_figure'],
            ue_noise_figure = CONF['ue_noise_figure'],
            seed = args.seed
            ) # the same seed for comparison,

# results = simulator.step()

# results = simulator.run_with_sim_para()

# opt = {'bandwidth_ul': 25.74, 'mcs_offset_ul': 2.35, 'bandwidth_dl': 39.3, 'mcs_offset_dl': 0.02, 'backhaul_bw': 14.59, 'cpu_ratio': 0.9}
# results = simulator.run_with_res_para(opt)

# # with open('Bayesian_optimziation_offline_evaluation_performance_numUEs_'+str(simulator.numUEs)+'.pkl', 'wb') as file:
# #     pickle.dump(results, file)

# REFERENCE = pickle.load(open("app_eval/real_indiv/"+ "measurement_system_performance_slice_0_traffic_"+str(simulator.numUEs)+".pickle", "rb" ))['performance']

# kl = calculate_kl_divergence(results['performance'], REFERENCE)

# print('simulated kl with optimal parameter is', kl, ', should be ~0.12')

# print('done')
# raise ValueError('completed successfully.')


## attention, XXX make sure this matches the setting in the simulator.py
PBOUNDS = {
            'bandwidth_ul':       (0,      40 ),
            'mcs_offset_ul':      (0,      10 ),
            'bandwidth_dl':       (0,      36 ),
            'mcs_offset_dl':      (0,      10 ),
            'backhaul_bw':        (0,      90 ),
            'cpu_ratio':          (0,      1  ),
}

DIM = len(PBOUNDS)

kappa, xi = 2.5, 0.01
weight = 0 if args.model == 'BNN' else args.weight # this will be updated attention XXX

All_Utilities = {
    'ei_offline':     UtilityFunction(kind="ei_offline",    kappa=kappa, xi=xi, dim=DIM, weight=weight),
    'pi_offline':     UtilityFunction(kind="poi_offline",   kappa=kappa, xi=xi, dim=DIM, weight=weight),
    'ucb_offline':    UtilityFunction(kind="ucb_offline",   kappa=kappa, xi=xi, dim=DIM, weight=weight),
    'gpucb_offline':  UtilityFunction(kind="gpucb_offline", kappa=kappa, xi=xi, dim=DIM, weight=weight),
    'ts_offline':     UtilityFunction(kind="ts_offline",    kappa=kappa, xi=xi, dim=DIM, weight=weight),
}

utility = {args.aq: All_Utilities[args.aq]}  # get the utility function, only one utility 

util_name = '_'.join(sorted(utility)) + '_' + args.program[:-3]

if DIM == 1:
    the_key = list(PBOUNDS.keys())[0]
    savename = os.getcwd()+"/results/offline/log-offline_training-"+str(DIM)+"dim-"+args.model+"_epoch_"+str(args.epochs)+"_numUEs_"+str(args.numUEs)+"_start_"+str(args.start)+"_"+util_name+"_parallel_"+str(args.parallel)+"_threshold_"+str(args.threshold)+"_availability_"+str(int(args.availability*100))+"_weight_"+str(int(args.weight*100))+"_para_"+the_key+".json"
else:
    savename = os.getcwd()+"/results/offline/log-offline_training-"+str(DIM)+"dim-"+args.model+"_epoch_"+str(args.epochs)+"_numUEs_"+str(args.numUEs)+"_start_"+str(args.start)+"_"+util_name+"_parallel_"+str(args.parallel)+"_threshold_"+str(args.threshold)+"_availability_"+str(int(args.availability*100))+"_weight_"+str(int(args.weight*100))+".json"

# optimizer = pickle.load(open('Bayesian_optimzier_'+savename.split('/')[-1]+'_model.pkl', 'rb'))

print("simulation parameters are: ", PBOUNDS)
print("simulation savename is: ", savename)
print('-'*100)

###############################################################################################################################################

if args.model == 'GP':        
    model = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True, n_restarts_optimizer=5, random_state=args.seed,)
elif args.model == 'BNN':
    model = BNN(input_dim=DIM, seed=args.seed, lr=1.0, gamma=0.999, activation=None) # 0.999 for 100 epochs, scheduler is good, but batch queries means time 10~16, so one more scale
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
#     shutil.copy(savename, savename.split('.')[0]+savename.split('.')[1]+"_backup"+savename.split('.')[-1])
#     load_logs(optimizer, logs=[savename])
# except:  
#     pass

saver = JSONSaver(path="offline_training_dataset_numUEs_"+str(args.numUEs)+util_name+".json")

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
        # results = [simulator.run_with_res_para(actions[0])]

    assert(len(actions) == len(results)) # make sure the multiprocessing works properly

    if ite == num_start_points: # when exactly is the start point, register these previously saved points into the optimizer buffer
        for i in range(len(start_points['next_points'])):
            optimizer.register(params=start_points['next_points'][i], target=start_points['targets'][i])

    targets, usages, qoes = [], [], []
    # calculate the target and store in the BO _space (buffer)
    for idx in range(len(results)): 
        action_dict = actions[idx].to_dict() # convert to dict as it is the input of BNN/GP here, rather than an action object
        next_point = {key:val for key, val in action_dict.items() if key in sorted(PBOUNDS)} # remove the other non-optimizing keys
        res = results[idx] # the performance

        usage = calculate_usage(action_dict, PBOUNDS)
        if args.program == 'main-mar.cc':
            qoe, perf = calculate_qoe_mar(res, args.threshold)
        elif args.program == 'main-hvs.cc':
            qoe, perf = calculate_qoe_hvs(res, args.threshold)
        elif args.program == 'main-iot.cc':
            qoe, perf = calculate_qoe_iot(res, args.threshold)
        else: raise ValueError('wrong program')

        target = - usage + weight * (qoe - args.availability)  # maxmize, so minus KL divergence
    
        # store temp, save or others
        if ite < num_start_points: # if start point, then temp save them
            start_points['next_points'].append(next_point)
            start_points['targets'].append(qoe)
        else: 
            optimizer.register(params=next_point, target=qoe) # attention XXX we learn qoe function directly, not the total target

        targets.append(target)
        qoes.append(qoe)
        usages.append(usage)
        RESULTS.append([ite, next_point, perf, target, usage, qoe]) # store the progress

        saver.update({"next_point":next_point, "perf":list(perf), "usage":usage, "qoe":qoe})
    
    # Lagrangian dual update with step size 0.1 TBD XXX TODO 
    if ite > num_start_points and args.model == 'BNN': 
        weight = np.clip(weight - 0.1 * (np.mean(qoes) - args.availability), 0, None)

        # attention, XXX we calculate the real ymax here for aqusition function, as weights are changing
        optimizer.ymax = calculate_ymax(optimizer._space.params, optimizer._space.target, weight, args.availability, optimizer._space.bounds)

        for util in utility.values(): # need to do everytime is changes
            util.weight = weight
            util.availability = args.availability

    print("\nite:", ite, "avg. target:", np.mean(targets), "avg. qoe:", np.mean(qoes), "avg. usage:", np.mean(usages), "weight", weight, "used time:", time.time() - start_time)

###############################################################################################################################################

print('optimal is ', optimizer.max)

# for i, res in enumerate(optimizer.res): print("Iteration {}: \t{}".format(i, res))    

with open('results/offline/offline_training_'+savename.split('/')[-1]+'_progress.pkl', 'wb') as file:
    pickle.dump(RESULTS, file)

with open('results/offline/offline_training_'+savename.split('/')[-1]+'_model.pkl', 'wb') as file:
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
plt.savefig('results/offline/offline_training_'+savename.split('/')[-1]+'_training_progress.pdf', format = 'pdf', dpi=300)


print('done with time ', time.time() - experiment_start_time, ' seconds')

