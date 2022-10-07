from cProfile import label
from matplotlib import pyplot as plt
import pickle
import numpy as np
import matplotlib
import seaborn as sns
from functions import *

font_size = 18
fig_size = (5,3.5)
linesize = 3
font = {'size'   : font_size}
matplotlib.rc('font', **font)


def fix_hist_step_vertical_line_at_end(ax):
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, matplotlib.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])

# ##################################################################################################################################################################################################################
# ##################################################################################################################################################################################################################
# ##################################################################################################################################################################################################################
from scipy.special import kl_div

#########################################################################################################
LENGTH = 100

PBOUNDS = {
            'bandwidth_ul':       (0,      40 ),
            'mcs_offset_ul':      (0,      10 ),
            'bandwidth_dl':       (0,      36 ),
            'mcs_offset_dl':      (0,      10 ),
            'backhaul_bw':        (0,      90 ),
            'cpu_ratio':          (0,      1  ),
}

REFERENCE = pickle.load(open("app_eval/real_indiv/measurement_system_performance_slice_0_traffic_1.pickle", "rb" ))['performance']

##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
def read_performance_from_file(folder, savename, threshold=300, availability=0.9, rangex=None, parallel=15):
    results = pickle.load(open(folder+savename, 'rb'))
    if rangex is None:
        results = np.array(results)
    else:
        results = np.array(results[:rangex*parallel])
    length = len(results)
    qoes, usages = np.zeros(length), np.zeros(length)
    for i in range(length):
        perf = results[i][2]
        qoes[i] = np.sum(perf <= threshold)/len(perf) # attention XXX this less equal is for MAR slice latency
        action_dict = results[i][1]
        usages[i] = calculate_usage(action_dict, PBOUNDS)

    selected_idxs = qoes >= availability
    selected_qoes = qoes[selected_idxs]
    selected_usages = usages[selected_idxs]
    selected_actions = results[selected_idxs]

    best_idx = np.argmin(selected_usages)
    best_usage = selected_usages[best_idx]
    best_qoe = selected_qoes[best_idx]
    best_action = selected_actions[best_idx][1]

    return best_usage, best_qoe, best_action

def plot_pareto(usages, qoes, threshold=0.9, labels=['Ours', 'GP-EI', 'GP-PI', 'GP-UCB', 'DLDA']):

    fig, ax = plt.subplots(figsize=fig_size)

    plt.axhline(y=threshold, color='black', linestyle='--')
    for i in range(len(labels)):
        x, y = usages[i]*100, qoes[i]
        plt.scatter(x, y, marker='o',s=150, color='C'+str(i))
        if i==3: plt.annotate(labels[i],(x-1.5, y+0.005), color='C'+str(i))
        elif i==4: plt.annotate(labels[i],(x-3, y+0.005), color='C'+str(i))
        else: plt.annotate(labels[i],(x+1.5, y-0.005), color='C'+str(i))

    # plt.arrow(25, 0.975, -5, -0.02, shape='full', width=0.002, color='green', edgecolor='green', head_width=0.01, head_length=0.3)
    # plt.annotate('Better',(20, 0.95), color='green')

    plt.xlabel('Resource Usage (%)',fontsize=font_size)
    plt.ylabel('QoE',fontsize=font_size)
    
    plt.tight_layout()
    plt.grid(which='both',linestyle='--',axis='both',color='gray')
    plt.ylim(0.89,1.0)
    plt.subplots_adjust(left=0.24, bottom=0.19, right=1-0.06, top=1-0.03)
    plt.xlim(left=10, right=50)

    plt.savefig("figures/result_offline_training_pareto.pdf", format = 'pdf', dpi=300)
    plt.show()
    print('done')
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
# usages = np.zeros(4)
# qoes = np.zeros(4)

# usages[0], qoes[0], _ = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-BNN_epoch_500_numUEs_1_start_100_ts_offline_parallel_15_threshold_300.0_availability_90_weight_100.json_progress.pkl", rangex=200)
# usages[1], qoes[1], _ = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-BNN_epoch_500_numUEs_1_start_100_ts_offline_parallel_15_threshold_300.0_availability_90_weight_100.json_progress.pkl", rangex=500)
# usages[2], qoes[2], _ = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-BNN_epoch_1000_numUEs_1_start_100_ts_offline_parallel_15_threshold_300.0_availability_90_weight_100.json_progress.pkl", rangex=1000)
# usages[3], qoes[3], _ = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-BNN_epoch_2000_numUEs_1_start_100_ts_offline_parallel_15_threshold_300.0_availability_90_weight_100.json_progress.pkl", rangex=2000)


##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
results = pickle.load(open("results/offline/offline_training_log-offline_training-6dim-BNN_epoch_1000_numUEs_1_start_100_ts_offline_parallel_15_threshold_300.0_availability_90_weight_100.json_progress.pkl", "rb" ))

targets, qoes, usages, ws = [], [], [], []
for res in results:
    targets.append(res[-3])
    usages.append(res[-2])
    qoes.append(res[-1])
    ws.append(  (res[-3] + res[-2]) / (res[-1] - 0.9) ) # 0.9 is availability

# process for the parallel
def averaging(y):
    dist_mat = np.reshape(y,(-1, 15)) # parallel = 15
    means = np.mean(dist_mat, axis=1)
    return means

# targets = averaging(np.array(targets))
# usages = averaging(np.array(usages))
# qoes = averaging(np.array(qoes))


# fig, ax = plt.subplots(figsize=fig_size)
# ax.plot(usages*100, label='Avg. usage (%)', color='C0')
# ax.set_ylabel('Avg.usage (%)', color='C0', fontsize=font_size-2)

# ax2 = ax.twinx()
# ax2.plot(qoes, label='QoE', color='C1')
# ax2.set_ylabel('Avg. QoE', color='C1', fontsize=font_size-2)
# ax.set_xlabel('Number of iterations')

# plt.tight_layout()
# plt.grid(which='both',linestyle='--',axis='both',color='gray')

# plt.subplots_adjust(left=0.16, bottom=0.19, right=1-0.16, top=1-0.03)

# plt.savefig("figures/result_offline_training_convergence.pdf", format = 'pdf', dpi=300)
# plt.show()
# print('done')

##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
### show convergence of BNN under 4-dim 100 num 40 start weight 2 aq ts parallel 16



usages = np.zeros(5)
qoes = np.zeros(5)
actions = []
usages[0], qoes[0], act_dict = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-BNN_epoch_1000_numUEs_1_start_100_ts_offline_parallel_15_threshold_300.0_availability_90_weight_100.json_progress.pkl", threshold=300)
actions.append(act_dict)
usages[1], qoes[1], act_dict = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-GP_epoch_1000_numUEs_1_start_100_ei_offline_parallel_1_threshold_300.0_availability_90_weight_1000.json_progress.pkl", threshold=300)
actions.append(act_dict)
usages[2], qoes[2], act_dict = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-GP_epoch_1000_numUEs_1_start_100_pi_offline_parallel_1_threshold_300.0_availability_90_weight_1000.json_progress.pkl", threshold=300)
actions.append(act_dict)
usages[3], qoes[3], act_dict = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-GP_epoch_1000_numUEs_1_start_100_ucb_offline_parallel_1_threshold_300.0_availability_90_weight_1000.json_progress.pkl", threshold=300)
actions.append(act_dict)
usages[4], qoes[4] = 0.2687096699894768, 0.9849624 # this is obtained by executing dadl.py, with offline mode
actions.append([11.35193436,  3.36348804, 11.10917298,  1.09510244,  0.51434964,  2.68625713])

# Attention XXX this code is for MAR slice only 
plot_pareto(usages, qoes)
print('done')


##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################

usages = np.zeros(3)
qoes = np.zeros(3)

usages[0], qoes[0], _ = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-BNN_epoch_1000_numUEs_1_start_100_ts_offline_parallel_15_threshold_300.0_availability_90_weight_100.json_progress.pkl", threshold=300)
# usages[1], qoes[1], _ = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-BNN_epoch_1000_numUEs_1_start_100_ts_offline_parallel_15_threshold_350.0_availability_90_weight_100.json_progress.pkl", threshold=350)
usages[1], qoes[1], _ = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-BNN_epoch_1000_numUEs_1_start_100_ts_offline_parallel_15_threshold_400.0_availability_90_weight_100.json_progress.pkl", threshold=400)
# usages[3], qoes[3], _ = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-BNN_epoch_1000_numUEs_1_start_100_ts_offline_parallel_15_threshold_450.0_availability_90_weight_100.json_progress.pkl", threshold=450)
usages[2], qoes[2], _ = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-BNN_epoch_1000_numUEs_1_start_100_ts_offline_parallel_15_threshold_500.0_availability_90_weight_100.json_progress.pkl", threshold=500)


# threshold 300 ms: 0.2687096699894768 0.9849624060150376 [11.35193436  3.36348804 11.10917298  1.09510244  0.51434964  2.68625713]
# threshold 350 ms: 0.2687096699894768 1.0 [11.35193436  3.36348804 11.10917298  1.09510244  0.51434964  2.68625713]
# threshold 400 ms: 0.23848992530722468 0.990990990990991 [10.0640635   5.47843868 19.77625206  0.88221298  0.41065301  0.32641837]
# threshold 450 ms: 0.13555657244045047 0.9493670886075949 [13.64129136  5.07563606  9.80803876  0.39568937  0.02736494  0.161832  ]
# threshold 500 ms: 0.13555657244045047 0.9466666666666667 [13.64129136  5.07563606  9.80803876  0.39568937  0.02736494  0.161832  ]

usages_dlda = np.array([0.2687, 0.238, 0.1355])
qoes_dlda = np.array([0.98, 0.999, 0.946])

#########################################################################################################
ind = np.arange(1,4,1)
width = 0.3

fig, ax = plt.subplots(figsize=fig_size)    

ax.bar(ind,usages*100,linewidth=linesize, width=width, label="Ours", linestyle='solid', color='C0')

ax.bar(ind+width,usages_dlda*100,linewidth=linesize, width=width, label="DLDA", linestyle='solid', color='C1')

plt.xticks(ind + width / 2, ('300','400','500'))

plt.legend(prop=dict(size=font_size),loc='lower left')
plt.xlabel('Threshold (ms)',fontsize=font_size)
plt.ylabel('Avg. usage (%)',fontsize=font_size)
plt.tight_layout()
plt.grid(which='both',linestyle='--',axis='y',color='gray')
# plt.ylim(0,800)

plt.tight_layout()
plt.subplots_adjust(left=0.16, bottom=0.20, right=0.980, top=0.970) # adjust when plt.show() and copy to here


plt.savefig("figures/result_offline_training_threshold.pdf", format = 'pdf', dpi=300)
plt.show()
print('done')


##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################

usages = np.zeros(6)
qoes = np.zeros(6)

usages[0], qoes[0], _ = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-BNN_epoch_1000_numUEs_1_start_100_ts_offline_parallel_15_threshold_300_availability_95_weight_1000.json_progress.pkl", availability=0.95)
usages[1], qoes[1], _ = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-BNN_epoch_1000_numUEs_1_start_100_ts_offline_parallel_15_threshold_300.0_availability_90_weight_100.json_progress.pkl", availability=0.9)
usages[2], qoes[2], _ = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-BNN_epoch_1000_numUEs_1_start_100_ts_offline_parallel_15_threshold_300_availability_80_weight_1000.json_progress.pkl", availability=0.8)
usages[3], qoes[3], _ = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-BNN_epoch_1000_numUEs_1_start_100_ts_offline_parallel_15_threshold_300_availability_70_weight_1000.json_progress.pkl", availability=0.7)
usages[4], qoes[4], _ = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-BNN_epoch_1000_numUEs_1_start_100_ts_offline_parallel_15_threshold_300_availability_60_weight_1000.json_progress.pkl", availability=0.6)
usages[5], qoes[5], _ = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-BNN_epoch_1000_numUEs_1_start_100_ts_offline_parallel_15_threshold_300_availability_50_weight_1000.json_progress.pkl", availability=0.5)


usages_gp = np.zeros(6)
qoes_gp = np.zeros(6)
usages_gp[0], qoes_gp[0], act_dict = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-GP_epoch_1000_numUEs_1_start_100_ei_offline_parallel_1_threshold_300.0_availability_90_weight_1000.json_progress.pkl", availability=0.95)
usages_gp[1], qoes_gp[1], act_dict = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-GP_epoch_1000_numUEs_1_start_100_ei_offline_parallel_1_threshold_300.0_availability_90_weight_1000.json_progress.pkl", availability=0.9)
usages_gp[2], qoes_gp[2], act_dict = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-GP_epoch_1000_numUEs_1_start_100_ei_offline_parallel_1_threshold_300.0_availability_90_weight_1000.json_progress.pkl", availability=0.8)
usages_gp[3], qoes_gp[3], act_dict = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-GP_epoch_1000_numUEs_1_start_100_ei_offline_parallel_1_threshold_300.0_availability_90_weight_1000.json_progress.pkl", availability=0.7)
usages_gp[4], qoes_gp[4], act_dict = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-GP_epoch_1000_numUEs_1_start_100_ei_offline_parallel_1_threshold_300.0_availability_90_weight_1000.json_progress.pkl", availability=0.6)
usages_gp[5], qoes_gp[5], act_dict = read_performance_from_file(folder = "results/offline/", savename="offline_training_log-offline_training-6dim-GP_epoch_1000_numUEs_1_start_100_ei_offline_parallel_1_threshold_300.0_availability_90_weight_1000.json_progress.pkl", availability=0.5)



# availability 0.99: 0.29391343511976453 0.9925925925925926 [6.73171141 5.51509302 8.09651787 1.22638673 0.46177783 3.84120525]
# availability 0.95: 0.2850552553317098 1.0 [17.43157026  7.71404181 10.31753746  1.17175724  0.05023521  1.94429815]
# availability 0.9:  0.2687096699894768 0.976 [11.35193436  3.36348804 11.10917298  1.09510244  0.51434964  2.68625713]
# availability 0.8:  0.26349558356966835 0.9396551724137931 [21.57048893  3.00731795  6.92258441  0.87729352  0.36407098  4.10182812]
# availability 0.7:  0.23668740664987306 0.8918918918918919 [22.44463538  3.07818671  8.25538616  0.73012353  1.51985498  2.06772508]
# availability 0.6:  0.23668740664987306 0.9117647058823529 [22.44463538  3.07818671  8.25538616  0.73012353  1.51985498  2.06772508]
# availability 0.5:  0.13555657244045047 0.33783783783783783 [13.64129136  5.07563606  9.80803876  0.39568937  0.02736494  0.161832  ]

usages_dlda = np.array([0.285, 0.2687, 0.263, 0.236, 0.236, 0.1355])
qoes_dlda = np.array([1.0, 0.976, 0.939, 0.89, 0.91, 0.337])



fig, ax = plt.subplots(figsize=fig_size)
plt.scatter(usages*100, qoes, marker='o',s=120, color='C0',)
plt.plot(usages*100, qoes, linewidth=linesize, color='C0',label='Ours')

plt.scatter(usages_dlda*100, qoes_dlda, marker='*',s=120,color='C1',)
plt.plot(usages_dlda*100, qoes_dlda, linewidth=linesize, color='C1',label='DLDA')

plt.scatter(usages_gp*100, qoes_gp, marker='^',s=120,color='C3',)
plt.plot(usages_gp*100, qoes_gp, linewidth=linesize, color='C3',label='GP-EI')

plt.arrow(16, 0.8, -2, 0.1, shape='full', width=0.008, head_width = 0.03, head_length=0.4, color='green', edgecolor='green', )
plt.annotate('Better',(14, 0.92), color='green')

plt.xlabel('Avg. usage (%)',fontsize=font_size)
plt.ylabel('Availability',fontsize=font_size)

plt.legend(prop=dict(size=font_size),loc='lower right')

plt.tight_layout()
plt.grid(which='both',linestyle='--',axis='both',color='gray')
# plt.ylim(0,0.4)
plt.subplots_adjust(left=0.18, bottom=0.19, right=1-0.04, top=1-0.03)
plt.xlim(left=12, right=30)

plt.savefig("figures/result_offline_training_availability.pdf", format = 'pdf', dpi=300)
plt.show()
print('done')


print('done')