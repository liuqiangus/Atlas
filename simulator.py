from ast import Assert
from tkinter import E
import numpy as np
import glob, copy
import matplotlib.pyplot as plt
import subprocess, pickle
import matplotlib, time, os
from sklearn.model_selection import GridSearchCV, ParameterGrid
import multiprocessing as mp
from functions import *

NS3_PATH = "/home/"+ os.getlogin()+"/workspace/ns-3-allinone/ns-3-dev"


def read_stats_txt(filename, skip_first_line=True, character=' '):
        stats = []
        # try:
        with open(NS3_PATH + "/" + filename) as f:
            if skip_first_line: f.readline() # the first line is notation, not data
            lines = f.readlines()
        for line in lines:
            stats.append(np.array([float(x) for x in line.split(character) if x != '\n']))
        # except:
        #     raise ValueError("file is broken and not read")
            # latencies = [int(x) for x in outputs.split()]

        # remove duplicate stats //  no need as the duplicates are from different users
        # stats = np.array(stats)
        # _, indices = np.unique(stats[:,0],return_index=True) # use the first column to determine uniqueness
        # stats = stats[indices, :]  # only count the unique stats

        return np.array(stats)

class Simulator:
    def __init__(self, 
                program = 'main-mar.cc',
                simtime = 10,
                numUEs = 1,
                filename = "stats.txt",
                bandwidth_ul = 50,
                bandwidth_dl = 50,
                mcs_offset_ul = 0,
                mcs_offset_dl = 0,
                backhaul_bw = 100, # this will times 10Mbps
                cpu_ratio = 1.0,
                baseline_loss = 38.57,
                enb_antenna_gain = 5.0,
                enb_tx_power = 30.0,
                enb_noise_figure = 5.0,
                ue_antenna_gain = 5.0,
                ue_tx_power = 30.0,
                ue_noise_figure = 9.0,
                backhaul_offset = 0,
                backhaul_delay = 0,
                edge_bw = 22300000000,
                edge_delay = 0,
                compute_time_mean_offset = 0,
                compute_time_std_offset = 0,
                loading_time_offset = 0,
                seed=None,
                ):

        self.program = program
        self.simtime = simtime
        self.numUEs = numUEs
        self.filename = filename
        self.savename = filename # temp
        self.bandwidth_ul = bandwidth_ul
        self.bandwidth_dl = bandwidth_dl
        self.mcs_offset_ul = mcs_offset_ul
        self.mcs_offset_dl = mcs_offset_dl
        self.backhaul_bw = backhaul_bw
        self.cpu_ratio = cpu_ratio
        self.baseline_loss = baseline_loss
        self.enb_antenna_gain = enb_antenna_gain
        self.enb_tx_power = enb_tx_power
        self.enb_noise_figure = enb_noise_figure
        self.ue_antenna_gain = ue_antenna_gain
        self.ue_tx_power = ue_tx_power
        self.ue_noise_figure = ue_noise_figure
        self.backhaul_offset = backhaul_offset
        self.backhaul_delay = backhaul_delay
        self.edge_bw = edge_bw
        self.edge_delay = edge_delay
        self.compute_time_mean_offset = compute_time_mean_offset
        self.compute_time_std_offset = compute_time_std_offset
        self.loading_time_offset = loading_time_offset

        self.savename = "measurement_app_perf_sim_slice_" + self.program[-6:-3] + '_traffic_' + str(self.numUEs) + '_pid_' + str(mp.current_process().pid) + self.filename[-4:]

        if seed is None: self.seed = mp.current_process().pid
        else: self.seed = seed

        # remove previous temp files
        try:
            fileList = glob.glob(NS3_PATH+'/DlRlcStats*')
            [os.remove(filePath) for filePath in fileList]
            fileList = glob.glob(NS3_PATH+'/UlRlcStats*')
            [os.remove(filePath) for filePath in fileList]
            fileList = glob.glob(NS3_PATH+'/measurement_app_perf_sim_slice_*')
            [os.remove(filePath) for filePath in fileList]
        except OSError:
            pass

    def step(self, action = np.ones(6)):

        if len(action) != 6: raise ValueError("make sure the action is 6-dim!")
        action = np.clip(action, 0, 1)

        # URBG, UMCS, DRBG, DMCS, TNBW, CCPU
        #  0     1     2     3     4     5  
        resource = Resource(
                            bandwidth_ul    =      int(action[0]*40  + 6  ), 
                            mcs_offset_ul   = 10 - int(action[1]*10  + 0  ),
                            bandwidth_dl    =      int(action[2]*36  + 3  ), 
                            mcs_offset_dl   = 10 - int(action[3]*10  + 0  ),
                            backhaul_bw     =      int(action[4]*90  + 3  ), # bw is 10Mbits/s as unit, see self.run_with_res_para
                            cpu_ratio       =      float(action[5]   + 0.3),
                  ) ## from the observation, the cpu ratio is not linear or 1/x relation, so TODO

        return self.run_with_res_para(resource)
    
    def run_with_res_para(self, resource):

        self.savename = "measurement_app_perf_sim_slice_" + self.program[-6:-3] + '_traffic_' + str(self.numUEs) + '_pid_' + str(mp.current_process().pid) + self.filename[-4:]
        if isinstance(resource, dict):
            command = NS3_PATH + "/ns3 run \" scratch/" + self.program + \
                            " --simtime=" + str(self.simtime) + \
                            " --numUEs=" + str(self.numUEs) + \
                            " --filename=" + str(self.savename) + \
                            " --bandwidth_ul=" + str(resource['bandwidth_ul']) + \
                            " --bandwidth_dl=" + str(resource['bandwidth_dl']) + \
                            " --mcs_offset_ul=" + str(resource['mcs_offset_ul']) + \
                            " --mcs_offset_dl=" + str(resource['mcs_offset_dl']) + \
                            " --cpu_ratio=" + str(resource['cpu_ratio']) + \
                            " --baseline_loss=" + str(self.baseline_loss) + \
                            " --enb_antenna_gain=" + str(self.enb_antenna_gain) + \
                            " --enb_tx_power=" + str(self.enb_tx_power) + \
                            " --enb_noise_figure=" + str(self.enb_noise_figure) + \
                            " --ue_antenna_gain=" + str(self.ue_antenna_gain) + \
                            " --ue_tx_power=" + str(self.ue_tx_power) + \
                            " --ue_noise_figure=" + str(self.ue_noise_figure) + \
                            " --backhaul_bw=" + str(int(resource['backhaul_bw'] * 10000000)) + \
                            " --backhaul_offset=" + str(int(self.backhaul_offset * 10000000)) + \
                            " --backhaul_delay=" + str(self.backhaul_delay) + \
                            " --edge_bw=" + str(self.edge_bw) + \
                            " --edge_delay=" + str(self.edge_delay) + \
                            " --compute_time_mean_offset=" + str(self.compute_time_mean_offset) + \
                            " --compute_time_std_offset=" + str(self.compute_time_std_offset) + \
                            " --loading_time_offset=" + str(self.loading_time_offset) + \
                            " --random_seed=" + str(mp.current_process().pid) + \
                            "\""

        else:
            command = NS3_PATH + "/ns3 run \" scratch/" + self.program + \
                            " --simtime=" + str(self.simtime) + \
                            " --numUEs=" + str(self.numUEs) + \
                            " --filename=" + str(self.savename) + \
                            " --bandwidth_ul=" + str(resource.bandwidth_ul) + \
                            " --bandwidth_dl=" + str(resource.bandwidth_dl) + \
                            " --mcs_offset_ul=" + str(resource.mcs_offset_ul) + \
                            " --mcs_offset_dl=" + str(resource.mcs_offset_dl) + \
                            " --cpu_ratio=" + str(resource.cpu_ratio) + \
                            " --baseline_loss=" + str(self.baseline_loss) + \
                            " --enb_antenna_gain=" + str(self.enb_antenna_gain) + \
                            " --enb_tx_power=" + str(self.enb_tx_power) + \
                            " --enb_noise_figure=" + str(self.enb_noise_figure) + \
                            " --ue_antenna_gain=" + str(self.ue_antenna_gain) + \
                            " --ue_tx_power=" + str(self.ue_tx_power) + \
                            " --ue_noise_figure=" + str(self.ue_noise_figure) + \
                            " --backhaul_bw=" + str(int(resource.backhaul_bw * 10000000)) + \
                            " --backhaul_offset=" + str(int(self.backhaul_offset * 10000000)) + \
                            " --backhaul_delay=" + str(self.backhaul_delay) + \
                            " --edge_bw=" + str(self.edge_bw) + \
                            " --edge_delay=" + str(self.edge_delay) + \
                            " --compute_time_mean_offset=" + str(self.compute_time_mean_offset) + \
                            " --compute_time_std_offset=" + str(self.compute_time_std_offset) + \
                            " --loading_time_offset=" + str(self.loading_time_offset) + \
                            " --random_seed=" + str(mp.current_process().pid) + \
                            "\""

        results = self.run_with_command(command, self.savename)

        return results

    def run_with_sim_para(self, action=None): ## attention XXX this order matters, as sometime we run this func with array inputs, not dict

        # attention here, XXX the scale of backhaul is reduced for better BNN/GP approximation, cause its orginal scale is too large

        self.savename = "measurement_app_perf_sim_slice_" + self.program[-6:-3] + '_traffic_' + str(self.numUEs) + '_pid_' + str(mp.current_process().pid) + self.filename[-4:]

        if action is None:
            command = NS3_PATH + "/ns3 run \" scratch/" + self.program + \
                                " --simtime=" + str(self.simtime) + \
                                " --numUEs=" + str(self.numUEs) + \
                                " --filename=" + str(self.savename) + \
                                " --bandwidth_ul=" + str(self.bandwidth_ul) + \
                                " --bandwidth_dl=" + str(self.bandwidth_dl) + \
                                " --mcs_offset_ul=" + str(self.mcs_offset_ul) + \
                                " --mcs_offset_dl=" + str(self.mcs_offset_dl) + \
                                " --cpu_ratio=" + str(self.cpu_ratio) + \
                                " --baseline_loss=" + str(self.baseline_loss) + \
                                " --enb_antenna_gain=" + str(self.enb_antenna_gain) + \
                                " --enb_tx_power=" + str(self.enb_tx_power) + \
                                " --enb_noise_figure=" + str(self.enb_noise_figure) + \
                                " --ue_antenna_gain=" + str(self.ue_antenna_gain) + \
                                " --ue_tx_power=" + str(self.ue_tx_power) + \
                                " --ue_noise_figure=" + str(self.ue_noise_figure) + \
                                " --backhaul_bw=" + str(int(self.backhaul_bw * 10000000)) + \
                                " --backhaul_offset=" + str(int(self.backhaul_offset * 10000000)) + \
                                " --backhaul_delay=" + str(self.backhaul_delay) + \
                                " --edge_bw=" + str(self.edge_bw) + \
                                " --edge_delay=" + str(self.edge_delay) + \
                                " --compute_time_mean_offset=" + str(self.compute_time_mean_offset) + \
                                " --compute_time_std_offset=" + str(self.compute_time_std_offset) + \
                                " --loading_time_offset=" + str(self.loading_time_offset) + \
                                " --random_seed=" + str(self.seed) + \
                                "\""
        else:
            command = NS3_PATH + "/ns3 run \" scratch/" + self.program + \
                                " --simtime=" + str(self.simtime) + \
                                " --numUEs=" + str(self.numUEs) + \
                                " --filename=" + str(self.savename) + \
                                " --bandwidth_ul=" + str(self.bandwidth_ul) + \
                                " --bandwidth_dl=" + str(self.bandwidth_dl) + \
                                " --mcs_offset_ul=" + str(self.mcs_offset_ul) + \
                                " --mcs_offset_dl=" + str(self.mcs_offset_dl) + \
                                " --cpu_ratio=" + str(self.cpu_ratio) + \
                                " --baseline_loss=" + str(action.baseline_loss) + \
                                " --enb_antenna_gain=" + str(action.enb_antenna_gain) + \
                                " --enb_tx_power=" + str(action.enb_tx_power) + \
                                " --enb_noise_figure=" + str(action.enb_noise_figure) + \
                                " --ue_antenna_gain=" + str(action.ue_antenna_gain) + \
                                " --ue_tx_power=" + str(action.ue_tx_power) + \
                                " --ue_noise_figure=" + str(action.ue_noise_figure) + \
                                " --backhaul_bw=" + str(self.backhaul_bw * 10000000) + \
                                " --backhaul_offset=" + str(action.backhaul_offset * 10000000) + \
                                " --backhaul_delay=" + str(action.backhaul_delay) + \
                                " --edge_bw=" + str(action.edge_bw) + \
                                " --edge_delay=" + str(action.edge_delay) + \
                                " --compute_time_mean_offset=" + str(action.compute_time_mean_offset) + \
                                " --compute_time_std_offset=" + str(action.compute_time_std_offset) + \
                                " --loading_time_offset=" + str(action.loading_time_offset) + \
                                " --random_seed=" + str(self.seed) + \
                                "\""

        results = self.run_with_command(command, self.savename)

        return results

    def run_with_command(self, command, filename): 

        # print("call the following command: ", command)

        outputs = subprocess.check_output(command,shell=True)
        # print("the command outputs: ", outputs)

        stats = read_stats_txt(filename) # the stats we record with callback functions, tracesources
        
        try: # if this fails, basically means the setting are too weak to have one packet to be completed during the simulation time
            Performance, Details, Sizes, Queued_Size = np.array(stats[:,0]), np.array(stats[:,1:5]), np.array(stats[:, 5:-1]), np.array(stats[:,-1])
            computetime = Details[:,2]

            stats_lte_ul = read_stats_txt("UlRlcStats"+filename+".txt", character='\t') # % start	end	CellId	IMSI	RNTI	LCID	nTxPDUs	TxBytes	nRxPDUs	RxBytes	delay	stdDev	min	max	PduSize	stdDev	min	max
            per_ul = np.clip(1 - np.sum(stats_lte_ul[:,9])/np.sum(stats_lte_ul[:,7]), 0, 1)
            datarate_ul = np.max(8*np.sum(stats_lte_ul[:,9])/self.simtime/1000000, 0) # 1 byte = 8 bits, eventually Mbits

            stats_lte_dl = read_stats_txt("DlRlcStats"+filename+".txt", character='\t') # % start	end	CellId	IMSI	RNTI	LCID	nTxPDUs	TxBytes	nRxPDUs	RxBytes	delay	stdDev	min	max	PduSize	stdDev	min	max
            per_dl = np.clip(1 - np.sum(stats_lte_dl[:,9])/np.sum(stats_lte_dl[:,7]), 0, 1)
            datarate_dl = np.max(8*np.sum(stats_lte_dl[:,9])/self.simtime/1000000, 0) # 1 byte = 8 bits, eventually Mbits

            PER_UL, Rate_UL, PER_DL, Rate_DL = per_ul, datarate_ul, per_dl, datarate_dl

        except:
            Performance, Details, Sizes, Queued_Size = np.array([10000]), np.array([0,0,0,0]), np.array([0,0]), np.array([0])
            computetime = [0]
            PER_UL, Rate_UL, PER_DL, Rate_DL = 0, 1, 0, 1

        results = {
            "performance": Performance,
            "fps": np.array([len(Performance)/self.simtime/self.numUEs]),
            "reliability": np.array([1 - 0.5*(PER_UL + PER_DL)]),
            "sizes": Sizes,
            "computetime": computetime,
            "queuesz": Queued_Size,
            "reliability_ul": np.array([1 - PER_UL]),
            "rate_ul": np.array([Rate_UL]),
            "reliability_dl": np.array([1 - PER_DL]),
            "rate_dl": np.array([Rate_DL]),
        }

        # remove temp files
        try:
            os.remove(NS3_PATH+'/'+filename)
            os.remove(NS3_PATH+'/'+"UlRlcStats"+filename+".txt")
            os.remove(NS3_PATH+'/'+"DlRlcStats"+filename+".txt")
        except OSError:
            pass

        return results 

    def grid_search_parameter(self, program='main-mar.cc', stage='offline'):
        MAP = dict(URBG=0, UMCS=1, DRBG=2, DMCS=3, TNBW=4, CCPU=5, )
        VARIABLES = { 'main-mar.cc': {"URBG":np.arange(0.1, 1.01, 0.2), \
                                    "CCPU":np.arange(0.1, 1.01, 0.2), \
                                    }, # MAR
                    'main-hvs.cc': {"DRBG":np.arange(0.1, 1.01, 0.2), \
                                    "TNBW":np.arange(0.1, 1.01, 0.2), \
                                    }, # Video
                    'main-iot.cc':  {"UMCS":np.arange(0.1, 1.01, 0.2), \
                                    "DMCS":np.arange(0.1, 1.01, 0.2), \
                                    }  # IoT
                    }
        
        grid = ParameterGrid(VARIABLES[program])
        print("grid search length:", len(grid))
        
        ##########################################################################
        RESULTS = []
        num_parallel = int(mp.cpu_count()) # use multiprocess to get more data, because PER and fps are single value for each simulation
        

        for idx in range(len(grid)):

            # get params
            params = grid[idx]
            # init action, single vector
            action = np.ones(len(MAP)) ## TODO XXX
            # assign to action
            for key, val in params.items():
                action[MAP[key]] = val

            print('grid idx', idx, 'action', action)

            # run the system
            start_time = time.time()
            # results = simulator.step(action)
            # performance, queuesz, computetime, reliability_ul, reliability_dl, fps = simulator.step(action)
            pool = mp.Pool(num_parallel)
            results = pool.map(simulator.step, np.repeat(np.array([action]),num_parallel, axis=0))
            pool.close()
            
            print("simualtion time is ", time.time() - start_time)
            print('-'*40)
            # print(params, np.mean(performance), np.mean(reliability_ul), np.mean(reliability_dl))
            print('-'*40)

            RESULTS.append(results)
            # RESULTS.append({"performance":performance, "queuesz":queuesz, "computetime":computetime, "reliability_ul":reliability_ul, "reliability_dl":reliability_dl, "fps": fps})

            pickle.dump(RESULTS, open("app_eval/measurement_simulator_"+stage+"_grid_search_sim_slice_"+program+".pickle", "wb" ))

    def grid_search_resource(self, program='main-mar.cc', threshold=300, numUE=1):

        times = 4 # 3**6 = 729
        VARIABLES = {
                    "bandwidth_ul": np.arange(0.1, 40, 40/times), 
                    "mcs_offset_ul":np.arange(0.1, 10, 10/times), 
                    "bandwidth_dl": np.arange(0.1, 36, 36/times), 
                    "mcs_offset_dl":np.arange(0.1, 10, 10/times), 
                    "backhaul_bw":  np.arange(0.1, 90, 90/times), 
                    "cpu_ratio":    np.arange(0.1, 1,  1 /times), 
                }

        PBOUNDS = {
            'bandwidth_ul':       (0,      40 ),
            'mcs_offset_ul':      (0,      10 ),
            'bandwidth_dl':       (0,      36 ),
            'mcs_offset_dl':      (0,      10 ),
            'backhaul_bw':        (0,      90 ),
            'cpu_ratio':          (0,      1  ),
        }
        
        grid = ParameterGrid(VARIABLES)
        print("grid search length:", len(grid))
        
        resource = Resource()
        resources = []
        for params in grid:
            resource.modify(
                bandwidth_ul  = params['bandwidth_ul'],
                mcs_offset_ul = params['mcs_offset_ul'],
                bandwidth_dl  = params['bandwidth_dl'],
                mcs_offset_dl = params['mcs_offset_dl'],
                backhaul_bw   = params['backhaul_bw'],
                cpu_ratio     = params['cpu_ratio'],
                )
            resources.append(copy.deepcopy(resource))

        ##########################################################################
        saver = JSONSaver(path="offline_training_dataset_numUEs_"+str(numUE)+"_grid_search.json")
        from tqdm import tqdm
        parallel = 16
        for idx in tqdm(range(int(len(resources)/parallel))):
            ress = resources[idx*parallel:(idx+1)*parallel]

            ## multiprocessing to accelerate the data collection
            pool = mp.Pool(len(ress))
            results = pool.map(self.run_with_res_para, ress)
            pool.close() 
            
            for i in range(len(results)):
                action_dict = resources[idx*parallel+i].to_dict() # convert to dict as it is the input of BNN/GP here, rather than an action object
                next_point = {key:val for key, val in action_dict.items() if key in sorted(PBOUNDS)} # remove the other non-optimizing keys

                usage = calculate_usage(action_dict, PBOUNDS)
                if program == 'main-mar.cc':
                    qoe, perf = calculate_qoe_mar(results[i], threshold)
                elif program == 'main-hvs.cc':
                    qoe, perf = calculate_qoe_hvs(results[i], threshold)
                elif program == 'main-iot.cc':
                    qoe, perf = calculate_qoe_iot(results[i], threshold)
                else: raise ValueError('wrong program')
                    
                saver.update({"next_point":next_point, "perf":list(perf), "usage":usage, "qoe":qoe})


if __name__ == "__main__": 

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--program', type=str, default='main-mar.cc')
    parser.add_argument('--stage', type=str, default='offline')
    parser.add_argument('--mode', type=str, default="indiv")
    parser.add_argument('--simtime', type=int, default=60)                  # simulation time in NS3
    parser.add_argument('--numUEs', type=int, default=1)                    # number of users, follow the trace
    parser.add_argument('--filename', type=str, default="Stats.txt")        # the name of the file to record the latencies, which is also output to terminal and captured then
    parser.add_argument('--bandwidth_ul', type=int, default=50)             # // number of PRBs, e.g., 25, 50, or 100  // # action parameters of slicing
    parser.add_argument('--bandwidth_dl', type=int, default=50)             # // number of PRBs, e.g., 25, 50, or 100  // # action parameters of slicing
    parser.add_argument('--mcs_offset_ul', type=int, default=0)             # // mcs offset  // # action parameters of slicing
    parser.add_argument('--mcs_offset_dl', type=int, default=0)             # // mcs offset  // # action parameters of slicing
    parser.add_argument('--backhaul_bw', type=int, default=100)             # // backhual bandwidth, 10Mbits/s // # action parameters of slicing
    parser.add_argument('--cpu_ratio', type=float, default=1.0)             # // the allocated CPU ratio in edge server // # action parameters of slicing
    parser.add_argument('--baseline_loss', type=float, default=38.57)       #  // baseline loss, as the distrance is fixed, so log attenuation model "becomes" baseline gain
    parser.add_argument('--enb_antenna_gain', type=float, default=5.0)      # // antenna gain
    parser.add_argument('--enb_tx_power', type=float, default=30.0)         #  // enb tx power in dB
    parser.add_argument('--enb_noise_figure', type=float, default=5.0)      # // enb tx noise figure (gain loss by hardware) in dB
    parser.add_argument('--ue_antenna_gain', type=float, default=5.0)       # // antenna gain
    parser.add_argument('--ue_tx_power', type=float, default=30.0)          # // ue tx power in dB
    parser.add_argument('--ue_noise_figure', type=float, default=9.0)       # // ue tx noise figure (gain loss by hardware) in dB
    parser.add_argument('--backhaul_offset', type=float, default=0)         #  // backhual bandwidth, bits/s
    parser.add_argument('--backhaul_delay', type=float, default=0)          # // backhual delay in milliseconds
    parser.add_argument('--edge_bw', type=int, default=22300000000)         # // edge bandwidth , bits/s
    parser.add_argument('--edge_delay', type=int, default=0)                # // edge delay in milliseconds
    parser.add_argument('--compute_time_mean_offset', type=int, default=0)             # // factor of compute time for task computation in edge server, in millisecond (currently is exp distribution)
    parser.add_argument('--compute_time_std_offset', type=int, default=0)             # // factor of compute time for task computation in edge server, in millisecond (currently is exp distribution)
    parser.add_argument('--loading_time_offset', type=int, default=0)             # // factor of compute time for task computation in edge server, in millisecond (currently is exp distribution)
    parser.add_argument('--seed', type=int, default=1111)                # // seed for simulator,i.e., NS3
    args = parser.parse_args()
    print(args)
    
    ##########################################################################
    if args.stage == "offline":
        simulator = Simulator(
                    program = args.program,
                    simtime = args.simtime,
                    numUEs = args.numUEs,
                    filename = args.filename,
                    bandwidth_ul = args.bandwidth_ul,
                    bandwidth_dl = args.bandwidth_dl,
                    mcs_offset_ul = args.mcs_offset_ul,
                    mcs_offset_dl = args.mcs_offset_dl,
                    backhaul_bw = args.backhaul_bw,
                    cpu_ratio = args.cpu_ratio,
                    baseline_loss = args.baseline_loss,
                    enb_antenna_gain = args.enb_antenna_gain,
                    enb_tx_power = args.enb_tx_power,
                    enb_noise_figure = args.enb_noise_figure,
                    ue_antenna_gain = args.ue_antenna_gain,
                    ue_tx_power = args.ue_tx_power,
                    ue_noise_figure = args.ue_noise_figure,
                    backhaul_offset = args.backhaul_offset,
                    backhaul_delay = args.backhaul_delay,
                    edge_bw = args.edge_bw,
                    edge_delay = args.edge_delay,
                    compute_time_mean_offset = args.compute_time_mean_offset,
                    compute_time_std_offset = args.compute_time_std_offset,
                    loading_time_offset = args.loading_time_offset,
                    seed=args.seed,
                    )
    elif args.stage == "online":
        from parameters import *
        CONF = OPTIMAL_PARA_SIM # [str(args.numUEs)]

        simulator = Simulator(
            simtime = args.simtime, 
            numUEs = args.numUEs,
            program = args.program,
            filename = args.filename,
            bandwidth_ul = args.bandwidth_ul,
            bandwidth_dl = args.bandwidth_dl,
            mcs_offset_ul = args.mcs_offset_ul,
            mcs_offset_dl = args.mcs_offset_dl,
            loading_time_offset = CONF['loading_time_offset'],
            compute_time_mean_offset = CONF['compute_time_mean_offset'],
            baseline_loss = CONF['baseline_loss'],
            backhaul_offset = CONF['backhaul_offset'],
            backhaul_delay = CONF['backhaul_delay'],
            enb_noise_figure = CONF['enb_noise_figure'],
            ue_noise_figure = CONF['ue_noise_figure'],
            seed = args.seed
            ) # the same seed for comparison,
    else:
        raise ValueError('make sure you set the correct stage, e.g., offline or online.')

    # simulator.grid_search_resource()
    # raise ValueError('All grid search is completed correctly.')
    ##########################################################################
    if args.mode == "indiv":
        ### XXX for simualtor run single configuration, uncomment this
        results = simulator.run_with_sim_para()

        with open('app_eval/measurement_simulator_evaluation_'+args.stage+'_'+args.mode+'_performance_numUEs_'+str(simulator.numUEs)+'.pkl', 'wb') as file:
            pickle.dump(results, file)

        print('indiviudal measurement is done')

    ##########################################################################
    ### XXX for grid search, run these codes
    if args.mode == "grid_para":
        simulator.grid_search_parameter(args.program, args.stage)        

    if args.mode == "grid_res":
        simulator.grid_search_resource(args.program, threshold=500, numUE=args.numUEs)        

    print(args.stage+': grid search measurement completed!')
