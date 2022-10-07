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
import pickle, time, sys, os, copy, math
from system import SYS
import matplotlib.pyplot as plt
sys.path.insert(0,'bayesian_torch')
from bayesian_torch.bnn import BNN
from simulator import Simulator
from functions import *
from parameters import *
###############################################################################################################################################
# NOTE:  change the PBOUNDS keys manually if you want different number of parameters to be learned

def select_next_queries(utility):
    resource = Resource() # create a default action
    
    resource.modify(**optimizer.suggest(utility)) # here, ** is for dynamic inputs XXX  fn is to add the usage as the calcuation of aq
    
    return resource

def calculate_ymax(X, Y, weight, availability, bounds):
    usage = np.array([np.mean(np.divide(x_try, bounds[:, 1])) for x_try in X])
    values = weight * (Y - availability) - usage
    idx = np.argmax(values)
    ymax = values[idx]

    return ymax

def retrieve_results_from_dataset(saver, action):

    action_dict = action.to_dict() # convert to dict as it is the input of OUR/GP here, rather than an action object
    next_point = {key:val for key, val in action_dict.items() if key in sorted(PBOUNDS)}

    new_key = generate_key_from_dict(next_point)
    if new_key in saver._data.keys():
        return saver._data[new_key]
    else:
        return None

def post_process_data(action, pounds, result, program, threshold):
        # calculate the target and store in the BO _space (buffer)
    action_dict = action.to_dict() # convert to dict as it is the input of OUR/GP here, rather than an action object
    next_point = {key:val for key, val in action_dict.items() if key in sorted(pounds)} # remove the other non-optimizing keys

    usage = calculate_usage(action_dict, pounds)
    if program == 'main-mar.cc':
        qoe, perf = calculate_qoe_mar(result, threshold)
    elif program == 'main-hvs.cc':
        qoe, perf = calculate_qoe_hvs(result, threshold)
    elif program == 'main-iot.cc':
        qoe, perf = calculate_qoe_iot(result, threshold)
    else: raise ValueError('wrong program')

    return next_point, usage, qoe, perf

###############################################################################################################################################

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--program', type=str, default='main-mar.cc')
parser.add_argument('--numUEs', type=int, default=1)
parser.add_argument('--model', type=str, default='GP')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--simtime', type=int, default=20)                  
parser.add_argument('--interval', type=int, default=10)                  
parser.add_argument('--threshold', type=float, default=300.0) # 500 ms for MAR slice
parser.add_argument('--availability', type=float, default=0.90)
parser.add_argument('--continued', type=bool, default=False)
parser.add_argument('--seed', type=int, default=1111)
args = parser.parse_args()

assert(args.program in ['main-mar.cc', 'main-hvs.cc', 'main-iot.cc'])
assert(args.seed != 0) # NS3 does not allow 0

print('MAKE SURE YOU UNDERSTAND THE CODES, AS THERE ARE SOME HARDCODES!')
print('-'*100)
print('simulation with, numUEs:', args.numUEs, ', epochs:', args.epochs, ', threshold:', args.threshold,  ', seed:', args.seed)

###############################################################################################################################################
# set the optimal value to the simulator
from parameters import OPTIMAL_PARA_SIM
experiment_start_time = time.time()

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

CONF = OPTIMAL_PARA_SIM
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

if args.program == 'main-mar.cc':
    sys = SYS(idx=0, traffic=args.numUEs, num_vars=DIM)
elif args.program == 'main-hvs.cc':
    sys = SYS(idx=1, traffic=args.numUEs, num_vars=DIM)
elif args.program == 'main-iot.cc':
    sys = SYS(idx=2, traffic=args.numUEs, num_vars=DIM)
else:
    raise ValueError('wrong program')

kappa, xi = 0, 0.01 # XXX kappa starts from zero, very much conservative

if args.numUEs == 1 and args.threshold == 300:
    offline_action = OPTIMIAL_RES_OUR
    weight = 0.2 # this will be updated attention XXX TODO set the weight manually from the last offline training
elif args.numUEs == 1 and args.threshold == 500:
    offline_action = OPTIMIAL_RES_OUR
    weight = 0.2 # this will be updated attention XXX TODO set the weight manually from the last offline training
elif args.numUEs == 2 and args.threshold == 500:
    offline_action = OPTIMIAL_RES_OUR_500_UE_2
    weight = 0.12 # this will be updated attention XXX TODO set the weight manually from the last offline training
elif args.numUEs == 3 and args.threshold == 500:
    offline_action = OPTIMIAL_RES_OUR_500_UE_3
    weight = 0.17 # this will be updated attention XXX TODO set the weight manually from the last offline training
elif args.numUEs == 4 and args.threshold == 500:
    offline_action = OPTIMIAL_RES_OUR_500_UE_4
    weight = 0.25 # this will be updated attention XXX TODO set the weight manually from the last offline training
else:
    raise ValueError('no optimal parameter') 
utility = UtilityFunction(kind="dcb", kappa=kappa, xi=xi, dim=DIM, weight=weight, availability=args.availability)  # get the utility function, only one utility 

savename = os.getcwd()+"/results/online/log-online_learning-"+str(DIM)+"dim-"+"_epoch_"+str(args.epochs)+"_numUEs_"+str(args.numUEs)+"_threshold_"+str(args.threshold)+"_availability_"+str(int(args.availability*100))+"_"+args.model+".json"

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

# attention XXX make sure this is the learned offline OUR model
offline_optimizer = pickle.load(open('results/offline/offline_training_log-offline_training-6dim-BNN_epoch_1000_numUEs_'+str(args.numUEs)+'_start_100_ts_offline_parallel_15_threshold_'+str(args.threshold)+'_availability_90_weight_1000.json_model.pkl', 'rb'))
BNN = offline_optimizer._model # get the BNN # predict with BNN.predict(X)
# BNN = copy.deepcopy(model) # enable this to evaluate "omitting offline training"

if args.continued: 
    model = BNN # if continued from previous offline model, then use this
    print('use continued model from offline')

optimizer = BayesianOptimization(
    model=model,
    f=None,
    pbounds=PBOUNDS,
    verbose=2, 
    random_state=args.seed,
    bnn=BNN, # attention, XXX we add BNN model here, which will be used when maximizing the aq functions in acq_max()
    n_warmup = 20000,
)

# load_logs(optimizer, logs=[savename])
offline_saver = JSONSaver(path="offline_training_dataset_numUEs_"+str(args.numUEs)+".json")
online_saver = JSONSaver(path="online_learning_dataset_numUEs_"+str(args.numUEs)+".json")

logger = JSONLogger(path=savename)
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

###############################################################################################################################################
RESULTS = []

for ite in range(len(optimizer.res), int(args.epochs*args.interval)):

    start_time = time.time()
    ###################################### select action  ########################################

    if ite == 0: # this avoid the random selection when the buffer in GP is empty
        action = offline_action # use the last offline searched optimal actions
    else:
        action = select_next_queries(utility) # if utility is one acq func, then it is one

    ######################################  get sim perf #########################################

    result_sim = retrieve_results_from_dataset(offline_saver, action) # try to get results from dataset
    if not result_sim:
        result_sim = simulator.run_with_res_para(action)

    next_point, usage, qoe_sim, perf_sim = post_process_data(action, PBOUNDS, result_sim, args.program, args.threshold)

    offline_saver.update({"next_point":next_point, "perf":list(perf_sim), "usage":usage, "qoe":qoe_sim})

    ######################################  get sys perf #########################################

    if ite%args.interval==0:
        result_sys = retrieve_results_from_dataset(online_saver, action) # try to get results from dataset
        if not result_sys:
            result_sys, relia_ul, relia_dl = sys.step(action)

        next_point, usage, qoe_sys, perf_sys = post_process_data(action, PBOUNDS, result_sys, args.program, args.threshold)

        gap = qoe_sys - qoe_sim # the gap under this action, define this gap here, acquisition function needs to be the same XXX

        optimizer.register(params=next_point, target=gap) # attention XXX we learn qoe function directly, not the total target
        
        # attention, XXX we calculate the real ymax here for some aqusition functions (not ours), as weights are changing
        optimizer.ymax = calculate_ymax(optimizer._space.params, optimizer._space.target, weight, args.availability, optimizer._space.bounds)

        online_saver.update({"next_point":next_point, "perf":list(perf_sys), "usage":usage, "qoe":qoe_sys})

        RESULTS.append([ite, next_point, perf_sys, weight, usage, qoe_sys]) # store the progress, only for online interactions
        print('-'*100)
        print("|SYS| ite:", ite, "avg. qoe:", qoe_sys, "avg. usage:", usage, "weight", utility.weight, "kappa", utility.kappa, "used time:", time.time() - start_time)
        print('-'*100)
        esti_qoe_sys = qoe_sys
    else:
        try:
            esti_gap = optimizer._model.predict([action.to_numpy()])[0]
        except:
            esti_gap = optimizer._model.predict([action.to_numpy()])
        esti_qoe_sys = qoe_sim + esti_gap # see above sys definition
        
        ###################################### update Lagrangian if needed ####################################################
        # Lagrangian dual update with step size 0.01 TBD 

        # XXX update weight kappa for dynamic confidence bound, only for sim, sys is more to contribute information, for the gap GP model
        if esti_qoe_sys > args.availability:
            weight = np.clip(weight - 0.06 * (esti_qoe_sys - args.availability), 0, None)
        else:
            weight = np.clip(weight - 0.02 * (esti_qoe_sys - args.availability), 0, None)
        utility.weight = weight   # TODO update method for kappa XXX TODO XXX

        theta = 0.1
        # beta = 2*np.log(np.power(int(ite/args.interval),DIM/2)+2*np.square(math.pi)/(3*theta)) # delta is 0.1 here
        k = np.log((np.square(int(ite/args.interval))+1)/np.sqrt(2)*math.pi)/np.log(1+theta/2)
        beta = np.clip(np.random.gamma(scale=k,shape=theta,size=1), 0, 10)[0] # XXX limit to 10 or less
        utility.kappa = np.sqrt(beta) # GP-UCB update method TODO XXX

        print("|SIM| ite:", ite, "avg. qoe:", qoe_sim, "avg. usage:", usage, "weight", utility.weight, "kappa", utility.kappa, "used time:", time.time() - start_time)

    ###############################################################################################################################################

    with open('results/online/online_learning_'+savename.split('/')[-1]+'_progress.pkl', 'wb') as file:
        pickle.dump(RESULTS, file)

    with open('results/online/online_learning_'+savename.split('/')[-1]+'_model.pkl', 'wb') as file:
        pickle.dump(optimizer, file)

###############################################################################################################################################

usages = [r[-2] for r in RESULTS]
qoes = [r[-1] for r in RESULTS]
plt.plot(usages, color='C0', label='Usage')
plt.plot(qoes, color='C1', label='QoE')
plt.legend()
plt.savefig('results/online/online_learning_'+savename.split('/')[-1]+'_training_progress.pdf', format = 'pdf', dpi=300)


print('done with time ', time.time() - experiment_start_time, ' seconds')

