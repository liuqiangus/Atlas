from cProfile import label
from matplotlib import pyplot as plt
import pickle
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import seaborn as sns
from functions import *

font_size = 18
fig_size = (5,3.5)
linesize = 3
font = {'size'   : font_size}
matplotlib.rc('font', **font)

FOLDER = "app_eval/"


def fix_hist_step_vertical_line_at_end(ax):
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, matplotlib.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])

#########################################################################################################
LENGTH = 500
PARALLEL = 15
Yname = "Discrepancy"

##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
### show convergence of BNN under 4-dim 100 num 40 start weight 2 aq ts parallel 16

def plot_convergence():
    folder = "results/simulator/"
    savename = "offline_simulator_offline_simulator_log_7dim_BNN_weight_7_epoch_500_numUEs_1_start_100_ts_parallel_15_percent_20.json_progress.pkl"
    results = pickle.load(open(folder+savename, 'rb'))
    dist = [-r[3] for r in results] # negative distance we optimized
    rem = len(dist)%PARALLEL
    if rem != 0: dist = dist[:-rem]
    dist_mat = np.reshape(dist,(-1, PARALLEL))
    means = np.mean(dist_mat, axis=1)
    tops = np.max(dist_mat, axis=1)
    buttons = np.min(dist_mat, axis=1)
    x = np.arange(len(means))

    best_idx = np.argmin(buttons)
    best = np.min(buttons)

    fig, ax = plt.subplots(figsize=fig_size)
    plt.fill_between(x, tops, buttons,color='lightgray')
    plt.plot(x, means, linewidth=linesize, color='C0')
    plt.scatter(best_idx, best, marker='*', s=150, c='C3', label='Best')
    plt.legend(prop=dict(size=font_size-4),loc='upper right')
    plt.xlabel('Number of iterations',fontsize=font_size)
    plt.ylabel(Yname,fontsize=font_size)
    plt.tight_layout()
    plt.grid(which='both',linestyle='--',axis='both',color='gray')
    plt.ylim(0.9,4)
    plt.subplots_adjust(left=0.14, bottom=0.19, right=1-0.06, top=1-0.03)
    plt.xlim(left=0, right=LENGTH)

    # plt.show()
    plt.savefig("figures/result_simulator_convergence.pdf", format = 'pdf', dpi=300)
    plt.show()
    print('done')
    
plot_convergence()


def get_average_performance(savename, parallel=1):
    folder = "results/simulator/"
    results = pickle.load(open(folder+savename, 'rb'))
    dist = [-r[3] for r in results] # negative distance we optimized
    rem = len(dist)%15 # make sure this matches the data above
    if rem != 0: dist = dist[:-rem]
    dist_mat = np.reshape(dist,(-1, parallel))
    means = np.mean(dist_mat, axis=1)
    tops = np.max(dist_mat, axis=1)
    buttons = np.min(dist_mat, axis=1)
    
    best_idx = np.argmin(buttons)
    best = np.min(buttons)

    return means, best_idx, best

avg_data_bnn, best_idx_bnn, best_bnn = get_average_performance(savename = "offline_simulator_offline_simulator_log_7dim_BNN_weight_7_epoch_500_numUEs_1_start_100_ts_parallel_15_percent_20.json_progress.pkl", parallel=15)
avg_data_gp, best_idx_gp, best_gp = get_average_performance(savename = "offline_simulator_offline_simulator_log_7dim_GP_weight_7_epoch_500_numUEs_1_start_100_ei_parallel_1_percent_20.json_progress.pkl", parallel=1)

def plot_convergence_compariosn(avg_data_bnn, best_idx_bnn, best_bnn, avg_data_gp, best_idx_gp, best_gp):
    fig, ax = plt.subplots(figsize=fig_size)
    x = np.arange(len(avg_data_gp))
    plt.plot(x, avg_data_gp, linewidth=linesize-1, color='C0')
    plt.scatter(best_idx_gp, best_gp, marker='*', s=350, c='C0', edgecolor='r', label='GP-Best')
    x = np.arange(len(avg_data_bnn))
    plt.plot(x, avg_data_bnn, linewidth=linesize-1, color='C1')
    plt.scatter(best_idx_bnn, best_bnn, marker='*', s=350, c='C1', edgecolor='r', label='Ours-Best')    

    plt.legend(prop=dict(size=font_size-4),loc='upper right')
    plt.xlabel('Number of iterations',fontsize=font_size)
    plt.ylabel('Avg. discrepancy',fontsize=font_size)
    plt.tight_layout()
    plt.grid(which='both',linestyle='--',axis='both',color='gray')
    # plt.ylim(0.9,4)
    plt.subplots_adjust(left=0.14, bottom=0.19, right=1-0.06, top=1-0.03)
    plt.xlim(left=0, right=LENGTH)

    # plt.show()
    plt.savefig("figures/result_simulator_convergence_comparison.pdf", format = 'pdf', dpi=300)
    plt.show()
    print('done')


plot_convergence_compariosn(avg_data_bnn, best_idx_bnn, best_bnn, avg_data_gp, best_idx_gp, best_gp)

##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################


def plot_cdf_comparison():

    folder = "results/simulator/"
    savename = "offline_simulator_offline_simulator_log_7dim_BNN_weight_7_epoch_500_numUEs_1_start_100_ts_parallel_15_percent_20.json_progress.pkl"
    results = pickle.load(open(folder+savename, 'rb'))
    dist = [-r[3] for r in results] # negative distance we optimized
    idx = np.argmin(dist)
    performance_mar_aug_sim = results[idx][2]

    savename = "offline_simulator_offline_simulator_log_7dim_GP_weight_7_epoch_500_numUEs_1_start_100_ei_parallel_1_percent_20.json_progress.pkl"
    results = pickle.load(open(folder+savename, 'rb'))
    dist = [-r[3] for r in results] # negative distance we optimized
    idx = np.argmin(dist)
    performance_mar_aug_sim_gp = results[idx][2]

    performance_mar_sim = pickle.load(open("app_eval/sim_indiv/measurement_simulator_evaluation_performance_numUEs_1.pkl", "rb" ))['performance']

    performance_mar_sys = pickle.load(open("app_eval/real_indiv/measurement_system_performance_slice_0_traffic_1.pickle", "rb" ))['performance']

    #########################################################################################################
    fig, ax = plt.subplots(figsize=fig_size)
    plt.hist(performance_mar_aug_sim_gp, bins=100,cumulative=True, density=True, label='Simulator, GP', histtype='step',  color='C1', linestyle='solid', linewidth=linesize)
    plt.hist(performance_mar_sys, bins=100,cumulative=True, density=True, label='System', histtype='step',  color='C0',linestyle='dashed', linewidth=linesize)
    plt.hist(performance_mar_aug_sim, bins=100,cumulative=True, density=True, label='Simulator, Ours', histtype='step',  color='C3',linestyle='dotted', linewidth=linesize)

    fix_hist_step_vertical_line_at_end(ax)

    plt.legend(prop=dict(size=font_size-1),loc='lower right')
    plt.xlabel('Latency (ms)',fontsize=font_size)
    plt.ylabel('CDF',fontsize=font_size)
    plt.tight_layout()
    plt.grid(which='both',linestyle='--',axis='both',color='gray')
    plt.ylim(0,1)
    plt.subplots_adjust(left=0.17, bottom=0.19, right=1-0.06, top=1-0.03)
    plt.xlim(left=100, right=650)

    plt.savefig("figures/result_simulator_learned_cdf_comparison.pdf", format = 'pdf', dpi=300)
    plt.show()

    print('done')


plot_cdf_comparison()

# ##################################################################################################################################################################################################################
# ##################################################################################################################################################################################################################
# ##################################################################################################################################################################################################################
# from scipy.special import kl_div

# def calculate_kl_divergence(x, y):

#     # here, the max(y) select the maximum y as the maximum and min(y) as the minimum
#     # then split them into 20 ranges. because we compare x TO y
#     xcounts, xbins = np.histogram(x, bins=20, range=(min(y), max(y))) 
#     ycounts, ybins = np.histogram(y, bins=20, range=(min(y), max(y))) 
    
#     xcounts = np.clip(xcounts, 1e-9, None)
#     ycounts = np.clip(ycounts, 1e-9, None)

#     xpdf = xcounts/np.sum(xcounts) # for normalization to 1, PDF
#     ypdf = ycounts/np.sum(ycounts) # for normalization to 1, PDF
    
#     kl = sum(kl_div(ypdf, xpdf)) # this order matters, shows how X is different from Y

#     return kl

# def calculate_conf_distance(x, y, percent):

#     distance = []
#     for key in y.keys():
#         if y.get(key, 0)[0] == 0:
#             dist = x[key] / y.get(key, 0)[-1]  # take the devation, as compared to the maximum
#         else:
#             # take the mean, which is the default baseline value, TODO if PBOUNDS [-x, x], then mean is zero
#             dist = np.abs(np.mean(y.get(key, 0)) - x[key]) / ( y.get(key, 0)[-1] - np.mean(y.get(key, 0))) 
#         distance.append(dist)
         
#     return np.linalg.norm(percent * np.array(distance)) # , the maximum is scaled to PERCENT

PERCENT = 0.2
from simulator import Simulator
simulator = Simulator()
PBOUNDS = {
            'loading_time_offset':       (0,                                        int(250*PERCENT)),
            'compute_time_mean_offset':  (0,                                        int(250*PERCENT)),
            'baseline_loss':             (simulator.baseline_loss*(1-PERCENT),      simulator.baseline_loss*(1+PERCENT)),
            'backhaul_offset':           (0,                                        int(100*PERCENT)), # may reduce this scale to
            'backhaul_delay':            (0,                                int(50*PERCENT)),
            'enb_noise_figure':          (simulator.enb_noise_figure*(1-PERCENT),   simulator.enb_noise_figure*(1+PERCENT)),
            'ue_noise_figure':           (simulator.ue_noise_figure*(1-PERCENT),    simulator.ue_noise_figure*(1+PERCENT)),
            # 'enb_antenna_gain':          (simulator.enb_antenna_gain*(1-PERCENT),   simulator.enb_antenna_gain*(1+PERCENT)),
            # 'enb_tx_power':              (simulator.enb_tx_power*(1-PERCENT),       simulator.enb_tx_power*(1+PERCENT)),
            # 'ue_antenna_gain':           (simulator.ue_antenna_gain*(1-PERCENT),    simulator.ue_antenna_gain*(1+PERCENT)),
            # 'ue_tx_power':               (simulator.ue_tx_power*(1-PERCENT),        simulator.ue_tx_power*(1+PERCENT)),
            # 'edge_bw':                   (simulator.edge_bw*(1-PERCENT),            simulator.edge_bw*(1+PERCENT)),
            # 'edge_delay':                (0,                                int(50*PERCENT)),
            # 'compute_time_std_offset':   (0,                                int(25*PERCENT)),
}

REFERENCE = pickle.load(open("app_eval/real_indiv/"+ "measurement_system_performance_slice_0_traffic_1.pickle", "rb" ))['performance']

def show_perf(name='BNN'):

    folder = "results/simulator/"
    if name == 'BNN':
        savename = "offline_simulator_offline_simulator_log_7dim_BNN_weight_7_epoch_500_numUEs_1_start_100_ts_parallel_15_percent_20.json_progress.pkl"
    else:
        savename = "offline_simulator_offline_simulator_log_7dim_GP_weight_7_epoch_500_numUEs_1_start_100_ei_parallel_1_percent_20.json_progress.pkl"
    results = pickle.load(open(folder+savename, 'rb'))
    dist = [-r[3] for r in results] # negative distance we optimized
    idx = np.argmin(dist)
    overall_min_dist = results[idx][3]
    performance_sim = results[idx][2]
    action_sim = results[idx][1]
    counts = len(dist)
    kl =  calculate_kl_divergence (performance_sim, REFERENCE)  
    act_dist = calculate_conf_distance(action_sim, PBOUNDS, PERCENT)

    print('action', action_sim, 'overall_min_dist', overall_min_dist, 'kl', kl, 'act_dist', act_dist, 'counts', counts)


print('-'*100)
show_perf(name='BNN')
show_perf(name='GP')

performan_org_sim = pickle.load(open("app_eval/sim_indiv/measurement_simulator_evaluation_performance_numUEs_1.pkl", "rb" ))['performance']
kl =  calculate_kl_divergence (performan_org_sim, REFERENCE)  
print('kl of original simulator is ', kl)
print('-'*100)


# ## plot result_motivation_sim_to_real_mar_traffic_1

# performance_mar_sim = read_sim_perf_from_file(FOLDER + "sim_indiv/"+ "measurement_app_perf_sim_slice_mar_traffic_1.pkl")

# performance_mar_sys = read_sys_perf_from_file(FOLDER + "real_indiv/"+ "measurement_app_perf_slice_0_traffic_1.pickle")

# #########################################################################################################
# fig, ax = plt.subplots(figsize=fig_size)
# plt.hist(performance_mar_sim, bins=100,cumulative=True, density=True, label='Simulator', histtype='step',  color='C0', linestyle='solid', linewidth=linesize)
# plt.hist(performance_mar_sys, bins=100,cumulative=True, density=True, label='System', histtype='step',  color='C1',linestyle='dashed', linewidth=linesize)

# fix_hist_step_vertical_line_at_end(ax)

# plt.legend(prop=dict(size=font_size),loc='lower right')
# plt.xlabel('Latency (ms)',fontsize=font_size)
# plt.ylabel('CDF',fontsize=font_size)
# plt.tight_layout()
# plt.grid(which='both',linestyle='--',axis='both',color='gray')
# plt.ylim(0,1)
# plt.subplots_adjust(left=0.17, bottom=0.19, right=1-0.06, top=1-0.03)
# plt.xlim(left=0, right=500)

# # plt.show()
# plt.savefig("figures/result_motivation_sim_to_real_mar_traffic_1.pdf", format = 'pdf', dpi=300)

# print('done')

# ##################################################################################################################################################################################################################
# ##################################################################################################################################################################################################################
# ##################################################################################################################################################################################################################
# # plot result_motivation_sim_to_real_mar_traffic_1

# performance_mar_sim_traffic_1 = read_sim_perf_from_file(FOLDER + "sim_indiv/"+ "measurement_app_perf_sim_slice_mar_traffic_1.pkl")
# performance_mar_sim_traffic_2 = read_sim_perf_from_file(FOLDER + "sim_indiv/"+ "measurement_app_perf_sim_slice_mar_traffic_2.pkl")
# performance_mar_sim_traffic_3 = read_sim_perf_from_file(FOLDER + "sim_indiv/"+ "measurement_app_perf_sim_slice_mar_traffic_3.pkl")
# performance_mar_sim_traffic_4 = read_sim_perf_from_file(FOLDER + "sim_indiv/"+ "measurement_app_perf_sim_slice_mar_traffic_4.pkl")
# # performance_mar_sim_traffic_5 = read_sim_perf_from_file(FOLDER + "sim_indiv/"+ "measurement_app_perf_sim_slice_mar_traffic_5.pkl")
# # performance_mar_sim_traffic_10 = read_sim_perf_from_file(FOLDER + "sim_indiv/"+ "measurement_app_perf_sim_slice_mar_traffic_10.pkl")
# performance_mar_sim = np.array([performance_mar_sim_traffic_1, performance_mar_sim_traffic_2, performance_mar_sim_traffic_3, performance_mar_sim_traffic_4, ])
# performance_mar_sim_mean = np.array([np.mean(x) for x in performance_mar_sim])

# performance_mar_sys_traffic_1 = read_sys_perf_from_file(FOLDER + "real_indiv/"+ "measurement_app_perf_slice_0_traffic_1.pickle")
# performance_mar_sys_traffic_2 = read_sys_perf_from_file(FOLDER + "real_indiv/"+ "measurement_app_perf_slice_0_traffic_2.pickle")
# performance_mar_sys_traffic_3 = read_sys_perf_from_file(FOLDER + "real_indiv/"+ "measurement_app_perf_slice_0_traffic_3.pickle")
# performance_mar_sys_traffic_4 = read_sys_perf_from_file(FOLDER + "real_indiv/"+ "measurement_app_perf_slice_0_traffic_4.pickle")
# # performance_mar_sys_traffic_5 = read_sys_perf_from_file(FOLDER + "real_indiv/"+ "measurement_app_perf_slice_0_traffic_5.pickle")
# # performance_mar_sys_traffic_10 = read_sys_perf_from_file(FOLDER + "real_indiv/"+ "measurement_app_perf_slice_0_traffic_10.pickle")
# performance_mar_sys = np.array([performance_mar_sys_traffic_1, performance_mar_sys_traffic_2, performance_mar_sys_traffic_3, performance_mar_sys_traffic_4,])
# performance_mar_sys_mean = np.array([np.mean(x) for x in performance_mar_sys])

# #########################################################################################################
# ind = np.arange(1,5,1)
# width = 0.3

# fig, ax2 = plt.subplots(figsize=fig_size)    

# ax2.boxplot(performance_mar_sim,widths=0.3)
# ax2.bar(ind,performance_mar_sim_mean,linewidth=linesize, width=width, label="Simulator", linestyle='solid', color='C0')

# ax2.boxplot(performance_mar_sys, positions=ind+width, widths=0.3)
# ax2.bar(ind+width,performance_mar_sys_mean,linewidth=linesize, width=width, label="System", linestyle='solid', color='C1')

# plt.xticks(ind + width / 2, ('1','2','3','4'))

# plt.legend(prop=dict(size=font_size),loc='upper left')
# plt.xlabel('User traffic',fontsize=font_size)
# plt.ylabel('Latency (ms)',fontsize=font_size)
# plt.tight_layout()
# plt.grid(which='both',linestyle='--',axis='y',color='gray')
# plt.ylim(0,1500)

# plt.tight_layout()
# plt.subplots_adjust(left=0.22, bottom=0.20, right=0.980, top=0.970) # adjust when plt.show() and copy to here

# # plt.show()
# plt.savefig("figures/result_motivation_sim_to_real_mar_traffic_comparison.pdf", format = 'pdf', dpi=300)

# print('done')

# ##################################################################################################################################################################################################################
# ##################################################################################################################################################################################################################
# #################################################################################################################################################################################################################

print("done")
