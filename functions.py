import numpy as np
from scipy.special import kl_div


def handle_response(response):

    if response.status<200 or response.status>299: 
        # print(",", end="")
        idx = False
    else:
        # print(".", end="")
        idx = True
    return idx


async def PUT(session, addr, payload):

    async with session.put(addr, data=payload) as response:

        idx = handle_response(response)

        html = await response.text()

        return html, idx

async def POST( session, addr, payload):

    async with session.post(addr, data=payload) as response:
        
        idx = handle_response(response)

        html = await response.text()

        return html, idx

async def GET(session, addr):

    async with session.get(addr) as response:

        idx = handle_response(response)

        html = await response.text()

        return html


async def DEL(session, addr):

    async with session.delete(addr) as response:

        idx = handle_response(response)

        html = await response.text()

        return html
###############################################################################################################################################
class Action:
    def __init__(self, 
                loading_time_offset = 0,
                compute_time_std_offset = 0,
                baseline_loss = 38.57,
                backhaul_offset = 0,
                backhaul_delay = 0,
                enb_noise_figure = 5.0,
                ue_noise_figure = 9.0,
                enb_antenna_gain = 5.0,
                enb_tx_power = 30.0,
                ue_antenna_gain = 5.0,
                ue_tx_power = 30.0,
                edge_bw = 22300000000,
                edge_delay = 0,
                compute_time_mean_offset = 0,
                ):
        self.loading_time_offset = loading_time_offset
        self.compute_time_std_offset = compute_time_std_offset
        self.baseline_loss = baseline_loss
        self.backhaul_offset = backhaul_offset
        self.backhaul_delay = backhaul_delay
        self.enb_noise_figure = enb_noise_figure
        self.ue_noise_figure = ue_noise_figure
        self.enb_antenna_gain = enb_antenna_gain
        self.enb_tx_power = enb_tx_power
        self.ue_antenna_gain = ue_antenna_gain
        self.ue_tx_power = ue_tx_power
        self.edge_bw = edge_bw
        self.edge_delay = edge_delay
        self.compute_time_mean_offset = compute_time_mean_offset
    
    def modify(self, 
                loading_time_offset = 0,
                compute_time_mean_offset = 0,
                baseline_loss = 38.57,
                backhaul_offset = 0,
                backhaul_delay = 0,
                enb_noise_figure = 5.0,
                ue_noise_figure = 9.0,
                enb_antenna_gain = 5.0,
                enb_tx_power = 30.0,
                ue_antenna_gain = 5.0,
                ue_tx_power = 30.0,
                edge_bw = 22300000000,
                edge_delay = 0,
                compute_time_std_offset = 0,
              ):

        self.loading_time_offset = loading_time_offset
        self.compute_time_std_offset = compute_time_std_offset
        self.baseline_loss = baseline_loss
        self.backhaul_offset = backhaul_offset
        self.backhaul_delay = backhaul_delay
        self.enb_noise_figure = enb_noise_figure
        self.ue_noise_figure = ue_noise_figure
        self.enb_antenna_gain = enb_antenna_gain
        self.enb_tx_power = enb_tx_power
        self.ue_antenna_gain = ue_antenna_gain
        self.ue_tx_power = ue_tx_power
        self.edge_bw = edge_bw
        self.edge_delay = edge_delay
        self.compute_time_mean_offset = compute_time_mean_offset
    
    def to_dict(self,):
        dict = {
            'loading_time_offset': self.loading_time_offset,
            'compute_time_std_offset': self.compute_time_std_offset,
            'baseline_loss': self.baseline_loss,
            'backhaul_offset': self.backhaul_offset,
            'backhaul_delay': self.backhaul_delay,
            'enb_noise_figure':self.enb_noise_figure,
            'ue_noise_figure': self.ue_noise_figure,
            'enb_antenna_gain': self.enb_antenna_gain,
            'enb_tx_power': self.enb_tx_power,
            'ue_antenna_gain': self.ue_antenna_gain,
            'ue_tx_power': self.ue_tx_power,
            'edge_bw': self.edge_bw,
            'edge_delay': self.edge_delay,
            'compute_time_mean_offset': self.compute_time_mean_offset,
        }
        return dict

class Resource:
    def __init__(self, 
                bandwidth_ul = 0,
                mcs_offset_ul = 0,
                bandwidth_dl = 0,
                mcs_offset_dl = 0,
                backhaul_bw = 0,
                cpu_ratio = 0,
                ):
        self.bandwidth_ul = bandwidth_ul
        self.mcs_offset_ul = mcs_offset_ul
        self.bandwidth_dl = bandwidth_dl
        self.mcs_offset_dl = mcs_offset_dl
        self.backhaul_bw = backhaul_bw
        self.cpu_ratio = cpu_ratio
    
    def modify(self, 
                bandwidth_ul = 0,
                mcs_offset_ul = 0,
                bandwidth_dl = 0,
                mcs_offset_dl = 0,
                backhaul_bw = 0,
                cpu_ratio = 0,
              ):

        self.bandwidth_ul  = int((bandwidth_ul  + 6)*100)/100
        self.mcs_offset_ul = int((mcs_offset_ul + 0)*100)/100
        self.bandwidth_dl  = int((bandwidth_dl  + 3)*100)/100
        self.mcs_offset_dl = int((mcs_offset_dl + 0)*100)/100
        self.backhaul_bw   = int((backhaul_bw   + 3)*100)/100
        self.cpu_ratio     = int((cpu_ratio     + 0.3)*100)/100
    
    def to_dict(self,):
        dict = {
            'bandwidth_ul': float(self.bandwidth_ul),
            'mcs_offset_ul': float(self.mcs_offset_ul),
            'bandwidth_dl': float(self.bandwidth_dl),
            'mcs_offset_dl': float(self.mcs_offset_dl),
            'backhaul_bw': float(self.backhaul_bw),
            'cpu_ratio':float(self.cpu_ratio),
        }
        return dict

    def to_numpy(self,):
        dict = {
            'bandwidth_ul': self.bandwidth_ul,
            'mcs_offset_ul': self.mcs_offset_ul,
            'bandwidth_dl': self.bandwidth_dl,
            'mcs_offset_dl': self.mcs_offset_dl,
            'backhaul_bw': self.backhaul_bw,
            'cpu_ratio':self.cpu_ratio,
        }
        dict = self.to_dict()
        res = []
        for k, d in sorted(dict.items()):
            res.append(d)
        return res

# original_conf = {   'baseline_loss':             simulator.baseline_loss, 
#                     'enb_antenna_gain':          simulator.enb_antenna_gain,
#                     'enb_tx_power':              simulator.enb_tx_power,
#                     'enb_noise_figure':          simulator.enb_noise_figure,
#                     'ue_antenna_gain':           simulator.ue_antenna_gain,
#                     'ue_tx_power':               simulator.ue_tx_power,
#                     'ue_noise_figure':           simulator.ue_noise_figure,
#                     'backhaul_offset':           simulator.backhaul_offset,
#                     'backhaul_delay':            simulator.backhaul_delay,
#                     'edge_bw':                   simulator.edge_bw,
#                     'edge_delay':                simulator.edge_delay,
#                     'compute_time_mean_offset':  simulator.compute_time_mean_offset,
#                     'compute_time_std_offset':   simulator.compute_time_std_offset,
#                     'loading_time_offset':       simulator.loading_time_offset}

def calculate_kl_divergence(x, y):

    # here, the max(y) select the maximum y as the maximum and min(y) as the minimum
    # then split them into 20 ranges. because we compare x TO y
    xcounts, xbins = np.histogram(x, bins=20, range=(min(y), max(y))) 
    ycounts, ybins = np.histogram(y, bins=20, range=(min(y), max(y))) 
    
    xcounts = np.clip(xcounts, 1e-9, None)
    ycounts = np.clip(ycounts, 1e-9, None)

    xpdf = xcounts/np.sum(xcounts) # for normalization to 1, PDF
    ypdf = ycounts/np.sum(ycounts) # for normalization to 1, PDF
    
    kl = sum(kl_div(ypdf, xpdf)) # this order matters, shows how X is different from Y

    return kl

def calculate_conf_distance(x, y, percent):

    distance = []
    for key in y.keys():
        if y.get(key, 0)[0] == 0:
            dist = x[key] / y.get(key, 0)[-1]  # take the devation, as compared to the maximum
        else:
            # take the mean, which is the default baseline value, TODO if PBOUNDS [-x, x], then mean is zero
            dist = np.abs(np.mean(y.get(key, 0)) - x[key]) / ( y.get(key, 0)[-1] - np.mean(y.get(key, 0))) 
        distance.append(dist)
         
    return np.linalg.norm(percent * np.array(distance)) # , the maximum is scaled to PERCENT


def calculate_qoe_mar(x, threshold=300):

    if isinstance(x, np.ndarray):
        perf = x
    else:
        try:
            perf = np.array(x['performance'])
        except:
            perf = np.array(x['perf'])

    qoe = np.sum(perf <= threshold)/len(perf) # here is latency, so less equal

    return qoe, perf


def calculate_qoe_hvs(x, threshold_fps=30):

    perf = x['fps']
    # qoe = np.sum(perf >= threshold_fps)/len(perf)
    qoe = min(np.mean(perf)/threshold_fps, 1.0)

    return qoe, perf


def calculate_qoe_iot(x, threshold_rel=0.9999):

    perf = np.clip(x['reliability'] -0.99, 0, 0.01)*100
    threshold_rel = np.clip(threshold_rel - 0.99, 0, 0.01)*100
    qoe = np.clip(perf/threshold_rel, 0, 1)

    return qoe, perf


def calculate_usage(action_dict, PBOUNDS):

    usage = np.mean([val/PBOUNDS.get(key)[-1] for key, val in action_dict.items()])

    return usage


def calculate_log_barrier(qoe, availability):
    
    if qoe < availability:
        return - np.log(qoe - availability + 1) #  here plus one, make sure log is on positive value
    else:
        return 0


def generate_key_from_dict(point):

    vals = list(point.values()) ## need sorted() to make sure order is the same!! next time...
    vals = [str(int(100*v)/100) for v in vals]
    new_key = '_'.join(vals)
    return new_key

import os, json

class JSONSaver():
    def __init__(self, path):
        self._path = path if path[-5:] == ".json" else path + ".json"

        self._data = {}

        if not os.path.exists(self._path):
            with open(self._path, 'w') as f:
                print("The json file is created")
        else:
            self.load() # load the data from file
        
    def update(self, instance):
        assert('next_point' in instance.keys())
        point = instance['next_point']
        
        new_key = generate_key_from_dict(point)

        if new_key not in self._data.keys():
            with open(self._path, "a") as f:
                f.write(json.dumps({new_key: instance}) + "\n")

    def load(self,):
        end = False
        with open(self._path, 'r') as f:
            while (not end):
                str_data = f.readline()
                if not str_data: break
                data = json.loads(str_data)
                for k, d in data.items():
                    self._data[k] = d
            
            print('load dataset done!')


