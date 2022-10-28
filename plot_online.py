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
def read_performance_from_file(savename, threshold=300, availability=0.9,rangex=100 ):
    results = pickle.load(open(savename, 'rb'))
    results = np.array(results[:rangex])
    length = len(results)
    qoes, usages = np.zeros(length), np.zeros(length)
    for i in range(length):
        perf = np.array(results[i][2])
        if len(perf) <= 1:
            qoes[i] = results[i][-1]
            usages[i] = results[i][-2]
        else:
            qoes[i] = np.sum(perf <= threshold)/len(perf) # attention XXX this less equal is for MAR slice latency
            action_dict = results[i][1]
            usages[i] = calculate_usage(action_dict, PBOUNDS)

    selected_idxs = qoes >= availability
    selected_qoes = qoes[selected_idxs]
    selected_usages = usages[selected_idxs]
    selected_actions = results[selected_idxs]

    if len(selected_usages)==0:
        best_idx = -1
        best_usage = 1.0
        best_qoe = 1.0
        best_action = [1.0]
    else:
        best_idx = np.argmin(selected_usages)
        best_usage = selected_usages[best_idx]
        best_qoe = selected_qoes[best_idx]
        best_action = selected_actions[best_idx][1]

    return usages, qoes, best_usage, best_qoe, best_action

# def plot_pareto(usages, qoes, threshold=0.9, labels=['Ours', 'GP-EI', 'GP-PI', 'GP-UCB', 'DLDA']):

#     fig, ax = plt.subplots(figsize=fig_size)

#     plt.axhline(y=threshold, color='black', linestyle='--')
#     for i in range(len(labels)):
#         x, y = usages[i]*100, qoes[i]
#         plt.scatter(x, y, marker='o',s=150, color='C'+str(i))
#         if i==3: plt.annotate(labels[i],(x-1.5, y+0.005), color='C'+str(i))
#         elif i==4: plt.annotate(labels[i],(x-3, y+0.005), color='C'+str(i))
#         else: plt.annotate(labels[i],(x+1.5, y-0.005), color='C'+str(i))

#     # plt.arrow(25, 0.975, -5, -0.02, shape='full', width=0.002, color='green', edgecolor='green', head_width=0.01, head_length=0.3)
#     # plt.annotate('Better',(20, 0.95), color='green')

#     plt.xlabel('Resource Usage (%)',fontsize=font_size)
#     plt.ylabel('QoE',fontsize=font_size)
    
#     plt.tight_layout()
#     plt.grid(which='both',linestyle='--',axis='both',color='gray')
#     plt.ylim(0.89,1.0)
#     plt.subplots_adjust(left=0.24, bottom=0.19, right=1-0.06, top=1-0.03)
#     plt.xlim(left=10, right=50)

#     plt.savefig("figures/result_offline_training_pareto.pdf", format = 'pdf', dpi=300)
#     plt.show()
#     print('done')



# ##################################################################################################################################################################################################################
# ##################################################################################################################################################################################################################
# ##################################################################################################################################################################################################################

usage_ours, qoe_ours, best_usage_ours, best_qoe_ours, best_action_ours = read_performance_from_file(savename="results/online/our/online_learning_log-online_learning-6dim-_epoch_100_numUEs_1_threshold_300_availability_90_GP.json_progress.pkl")
usage_org_sim, qoe_org_sim, best_usage_org_sim, best_qoe_org_sim, best_action_org_sim = read_performance_from_file(savename="results/online/our_org_sim/online_learning_log-online_learning-6dim-_epoch_100_numUEs_1_threshold_300_availability_90_GP_org_sim.json_progress.pkl")
usage_no_offline, qoe_no_offline, best_usage_no_offline, best_qoe_no_offline, best_action_no_offline = read_performance_from_file(savename="results/online/no_offline/online_learning_log-online_learning-6dim-_epoch_100_numUEs_1_threshold_300_availability_90_GP_no_offline.json_progress.pkl")
usage_no_online, qoe_no_online, best_usage_no_online, best_qoe_no_online, best_action_no_online = read_performance_from_file(savename="results/online/online_eval/online_learning_log-online_learning-6dim-_epoch_50_numUEs_1_threshold_300_availability_90_GP_eval.json_progress.pkl")


################################################################################################################
best_usages                 = [best_usage_ours, best_usage_org_sim, best_usage_no_offline, best_usage_no_online]
best_qoes                   = [best_qoe_ours, best_qoe_org_sim, best_qoe_no_offline, best_qoe_no_online]
idx                         = np.argmin(best_usages)
best_of_best_usage          = best_usages[idx] 
best_of_best_qoe            = best_qoes[idx]

regret_usage_ours           = np.sum(usage_ours - best_of_best_usage)
regret_usage_org_sim        = np.sum(usage_org_sim - best_of_best_usage)
regret_usage_no_offline     = np.sum(usage_no_offline - best_of_best_usage)
regret_usage_no_online      = np.sum(usage_no_online - best_of_best_usage)

regret_qoe_ours             = np.sum(np.clip(best_of_best_qoe - qoe_ours, 0, None))
regret_qoe_org_sim          = np.sum(np.clip(best_of_best_qoe - qoe_org_sim, 0, None))
regret_qoe_no_offline       = np.sum(np.clip(best_of_best_qoe - qoe_no_offline, 0, None) )
regret_qoe_no_online        = np.sum(np.clip(best_of_best_qoe - qoe_no_online, 0, None))

print('ours', regret_usage_ours, regret_qoe_ours)
print('org_sim', regret_usage_org_sim, regret_qoe_org_sim)
print('no_offline', regret_usage_no_offline, regret_qoe_no_offline)
print('no_online', regret_usage_no_online, regret_qoe_no_online)
################################################################################################################

fig, ax = plt.subplots(figsize=fig_size)

plt.scatter(usage_no_online*100, qoe_no_online, color='C3', marker='1', label='No stage 3')

plt.scatter(usage_no_offline*100, qoe_no_offline, color='C2',marker='d', label='No stage 2')

plt.scatter(usage_org_sim*100, qoe_org_sim, color='C1',marker='s', label='No stage 1')

plt.scatter(usage_ours*100, qoe_ours, color='C0',marker='o', label='Ours')

# plt.arrow(16, 0.8, -2, 0.1, shape='full', width=0.008, head_width = 0.03, head_length=0.4, color='green', edgecolor='green', )
# plt.annotate('Better',(14, 0.92), color='green')

plt.xlabel('Usage (%)',fontsize=font_size)
plt.ylabel('QoE',fontsize=font_size)

plt.legend(prop=dict(size=font_size-2),loc='lower right')

plt.tight_layout()
plt.grid(which='both',linestyle='--',axis='both',color='gray')
# plt.ylim(0,0.4)
plt.subplots_adjust(left=0.20, bottom=0.19, right=1-0.04, top=1-0.03)
plt.xlim(left=12, right=45)

plt.savefig("figures/result_online_components.pdf", format = 'pdf', dpi=300)
plt.show()
print('done')

# ##################################################################################################################################################################################################################
# ##################################################################################################################################################################################################################
# ##################################################################################################################################################################################################################
usage_ours, qoe_ours, best_usage_ours, best_qoe_ours, best_action_ours = read_performance_from_file(savename="results/online/our/online_learning_log-online_learning-6dim-_epoch_100_numUEs_1_threshold_300_availability_90_GP.json_progress.pkl")
usage_ours_bnn, qoe_ours_bnn, best_usage_ours_bnn, best_qoe_ours_bnn, best_action_ours_bnn = read_performance_from_file(savename="results/online/gap-model-bnn/online_learning_log-online_learning-6dim-_epoch_100_numUEs_1_threshold_300_availability_90_BNN.json_progress.pkl")
usage_ours_bnnc, qoe_ours_bnnc, best_usage_ours_bnnc, best_qoe_ours_bnnc, best_action_ours_bnnc = read_performance_from_file(savename="results/online/continue_bnn/online_learning_log-online_learning-6dim-_epoch_100_numUEs_1_threshold_300_availability_90_BNN.json_progress.pkl")
usage_ours_full_online, qoe_ours_full_online, best_usage_ours_full_online, best_qoe_ours_full_online, best_action_ours_full_online = read_performance_from_file(savename="results/online/full_online/online_learning_log-online_learning-6dim-_epoch_100_numUEs_1_threshold_300_availability_90_GP.json_progress.pkl")

################################################################################################################
best_usages                 = [best_usage_ours, best_usage_ours_bnn, best_usage_ours_bnnc, best_usage_ours_full_online]
best_qoes                   = [best_qoe_ours, best_qoe_ours_bnn, best_qoe_ours_bnnc, best_qoe_ours_full_online]
idx                         = np.argmin(best_usages)
best_of_best_usage          = best_usages[idx] 
best_of_best_qoe            = best_qoes[idx]

regret_usage_ours           = np.sum(usage_ours - best_of_best_usage)
regret_usage_bnn            = np.sum(usage_ours_bnn - best_of_best_usage)
regret_usage_bnnc           = np.sum(usage_ours_bnnc - best_of_best_usage)
regret_usage_full_online    = np.sum(usage_ours_full_online - best_of_best_usage)

regret_qoe_ours             = np.sum(np.clip(best_of_best_qoe - qoe_ours, 0, None))
regret_qoe_bnn              = np.sum(np.clip(best_of_best_qoe - qoe_ours_bnn, 0, None))
regret_qoe_bnnc             = np.sum(np.clip(best_of_best_qoe - qoe_ours_bnnc, 0, None) )
regret_qoe_full_online      = np.sum(np.clip(best_of_best_qoe - qoe_ours_full_online, 0, None) )

print('ours', regret_usage_ours, regret_qoe_ours)
print('bnn', regret_usage_bnn, regret_qoe_bnn)
print('bnnc', regret_usage_bnnc, regret_qoe_bnnc)
print('full_online', regret_usage_full_online, regret_qoe_full_online)
################################################################################################################

#########################################################################################################
fig, ax = plt.subplots(figsize=fig_size)

plt.scatter(regret_usage_full_online, regret_qoe_full_online/100, color='C3',marker='d', s=200, label="No Offline Acc.")

plt.scatter(regret_usage_bnnc, regret_qoe_bnnc/100, color='C2',marker='d', s=200, label="BNN-Cont'd")

plt.scatter(regret_usage_bnn, regret_qoe_bnn/100, color='C1',marker='s', s=200, label='BNN')

plt.scatter(regret_usage_ours, regret_qoe_ours/100, color='C0',marker='o',s=200, label='Ours')

plt.arrow(-2.5, 0.3, -5, -0.2, shape='full', width=0.01, head_width = 0.03, head_length=0.6, color='green', edgecolor='green', )
plt.annotate('Better',(-7, 0.05), color='green')

plt.xlabel('Avg. Usage Regret (%)',fontsize=font_size)
plt.ylabel('Avg. QoE Regret',fontsize=font_size)

plt.legend(prop=dict(size=font_size),loc='upper right')

plt.tight_layout()
plt.grid(which='both',linestyle='--',axis='both',color='gray')
# plt.ylim(0,0.4)
plt.subplots_adjust(left=0.20, bottom=0.19, right=1-0.04, top=1-0.03)
# plt.xlim(left=12, right=30)

plt.savefig("figures/result_online_gap_model.pdf", format = 'pdf', dpi=300)
plt.show()
print('done')



# ##################################################################################################################################################################################################################
# ##################################################################################################################################################################################################################
# ##################################################################################################################################################################################################################



usage_ours, qoe_ours, best_usage_ours, best_qoe_ours, best_action_ours = read_performance_from_file(savename="results/online/our/online_learning_log-online_learning-6dim-_epoch_100_numUEs_1_threshold_300_availability_90_GP.json_progress.pkl")
usage_dlda, qoe_dlda, best_usage_dlda, best_qoe_dlda, best_action_dlda = read_performance_from_file(savename="results/online/dlda/online_learninglog-online_learning-6dim-DNN_epoch_1000_numUEs_1_threshold_300_availability_90.json_progress.pkl")
usage_virtualedge, qoe_virtualedge, best_usage_virtualedge, best_qoe_virtualedge, best_action_virtualedge = read_performance_from_file(savename="results/online/virtualedge/online_learninglog-online_learning-6dim-BNN_epoch_100_numUEs_1_threshold_300_availability_90_virtualedge.json_progress_virtualedge.pkl")
usage_gp, qoe_gp, best_usage_gp, best_qoe_gp, best_action_gp = read_performance_from_file(savename="results/online/gp/online_learning_log-online_learning-6dim-_epoch_100_numUEs_1_threshold_300_availability_90_GP_ei_offline.json_progress.pkl")

################ usage curve ###################
fig, ax = plt.subplots(figsize=fig_size)
plt.plot(usage_gp*100, linewidth=linesize-1, color='gray', linestyle='dotted',label='Baseline')
plt.plot(usage_virtualedge*100, linewidth=linesize-1, color='C2', linestyle='dashed', label='VirtualEdge')
plt.plot(usage_dlda*100, linewidth=linesize-1, color='C1', linestyle='dashdot',label='DLDA')
plt.plot(usage_ours*100, linewidth=linesize, color='C0',linestyle='solid', label='Ours')

plt.legend(prop=dict(size=font_size-4),loc='upper left')
plt.xlabel('Number of iterations',fontsize=font_size)
plt.ylabel('Avg. Usage (%)',fontsize=font_size)
plt.tight_layout()
plt.grid(which='both',linestyle='--',axis='both',color='gray')
plt.ylim(10,100)
plt.subplots_adjust(left=0.18, bottom=0.19, right=1-0.06, top=1-0.03)
plt.xlim(left=0, right=100)

# plt.show()
plt.savefig("figures/result_online_convergence_usage.pdf", format = 'pdf', dpi=300)
plt.show()


fig, ax = plt.subplots(figsize=fig_size)
plt.plot(qoe_gp, linewidth=linesize-1, color='gray', linestyle='dotted', label='Baseline')
plt.plot(qoe_virtualedge, linewidth=linesize-1, color='C2', linestyle='dashed', label='VirtualEdge')
plt.plot(qoe_dlda, linewidth=linesize-1, color='C1',linestyle='dashdot', label='DLDA')
plt.plot(qoe_ours, linewidth=linesize, color='C0',linestyle='solid', label='Ours')

plt.legend(prop=dict(size=font_size-4),loc='lower right')
plt.xlabel('Number of iterations',fontsize=font_size)
plt.ylabel('Avg. QoE',fontsize=font_size)
plt.tight_layout()
plt.grid(which='both',linestyle='--',axis='both',color='gray')
# plt.ylim(0.9,4)
plt.subplots_adjust(left=0.20, bottom=0.19, right=1-0.06, top=1-0.03)
plt.xlim(left=0, right=100)

# plt.show()
plt.savefig("figures/result_online_convergence_qoe.pdf", format = 'pdf', dpi=300)
plt.show()

print('done')

################################################################################################################
best_usages                 = [best_usage_ours, best_usage_dlda, best_usage_virtualedge, best_usage_gp]
best_qoes                   = [best_qoe_ours, best_qoe_dlda, best_qoe_virtualedge, best_qoe_gp]
idx                         = np.argmin(best_usages)
best_of_best_usage          = best_usages[idx] 
best_of_best_qoe            = best_qoes[idx]

regret_usage_ours           = np.sum(usage_ours - best_of_best_usage)
regret_usage_dlda           = np.sum(usage_dlda - best_of_best_usage)
regret_usage_virtualedge    = np.sum(usage_virtualedge - best_of_best_usage)
regret_usage_gp             = np.sum(usage_gp - best_of_best_usage)

regret_qoe_ours             = np.sum(np.clip(best_of_best_qoe - qoe_ours, 0, None))
regret_qoe_dlda             = np.sum(np.clip(best_of_best_qoe - qoe_dlda, 0, None))
regret_qoe_virtualedge      = np.sum(np.clip(best_of_best_qoe - qoe_virtualedge, 0, None) )
regret_qoe_gp               = np.sum(np.clip(best_of_best_qoe - qoe_gp, 0, None))

print('ours', regret_usage_ours, regret_qoe_ours)
print('dlda', regret_usage_dlda, regret_qoe_dlda)
print('virtualedge', regret_usage_virtualedge, regret_qoe_virtualedge)
print('gp', regret_usage_gp, regret_qoe_gp)
################################################################################################################

#########################################################################################################
usage_dlda, qoe_dlda, best_usage_dlda, best_qoe_dlda, best_action_dlda = read_performance_from_file(savename="results/online/dlda/online_learninglog-online_learning-6dim-DNN_epoch_1000_numUEs_1_threshold_300_availability_90.json_progress.pkl",rangex=1000)


fig, ax = plt.subplots(figsize=fig_size)

plt.scatter(usage_dlda*100, qoe_dlda, color='C0', marker='o',s=50)

plt.axhline(y=0.9, color='red', linestyle='--')
plt.annotate('QoE threshold',(40.5, 0.82), color='red')

plt.xlabel('Usage (%)',fontsize=font_size)
plt.ylabel('QoE',fontsize=font_size)

# plt.legend(prop=dict(size=font_size),loc='lower right')

plt.tight_layout()
plt.grid(which='both',linestyle='--',axis='both',color='gray')
# plt.ylim(0,0.4)
plt.subplots_adjust(left=0.16, bottom=0.19, right=1-0.02, top=1-0.03)
plt.xlim(left=25, right=53)

plt.savefig("figures/result_online_motivation_dlda.pdf", format = 'pdf', dpi=300)
plt.show()
print('done')


# ##################################################################################################################################################################################################################
# ##################################################################################################################################################################################################################
# ##################################################################################################################################################################################################################

usage_ours, qoe_ours, best_usage_ours, best_qoe_ours, best_action_ours = read_performance_from_file(savename="results/online/our/online_learning_log-online_learning-6dim-_epoch_100_numUEs_1_threshold_300_availability_90_GP.json_progress.pkl")
usage_ucb, qoe_ucb, best_usage_ucb, best_qoe_ucb, best_action_ucb = read_performance_from_file(savename="results/online/gp-ucb/online_learning_log-online_learning-6dim-_epoch_100_numUEs_1_threshold_300_availability_90_GP.json_progress.pkl")
usage_ei, qoe_ei, best_usage_ei, best_qoe_ei, best_action_ei = read_performance_from_file(savename="results/online/ei/online_learning_log-online_learning-6dim-_epoch_100_numUEs_1_threshold_300_availability_90_GP_ei_offline.json_progress.pkl")
usage_pi, qoe_pi, best_usage_pi, best_qoe_pi, best_action_pi = read_performance_from_file(savename="results/online/pi/online_learning_log-online_learning-6dim-_epoch_100_numUEs_1_threshold_300_availability_90_GP_poi_offline.json_progress.pkl")


################################################################################################################
best_usages                 = [best_usage_ours, best_usage_dlda, best_usage_virtualedge, best_usage_gp]
best_qoes                   = [best_qoe_ours, best_qoe_dlda, best_qoe_virtualedge, best_qoe_gp]
idx                         = np.argmin(best_usages)
best_of_best_usage          = best_usages[idx] 
best_of_best_qoe            = best_qoes[idx]

regret_usage_ours           = np.sum(usage_ours - best_of_best_usage)
regret_usage_ucb            = np.sum(usage_ucb - best_of_best_usage)
regret_usage_ei             = np.sum(usage_ei - best_of_best_usage)
regret_usage_pi             = np.sum(usage_pi - best_of_best_usage)

regret_qoe_ours             = np.sum(np.clip(best_of_best_qoe - qoe_ours, 0, None))
regret_qoe_ucb              = np.sum(np.clip(best_of_best_qoe - qoe_ucb, 0, None))
regret_qoe_ei               = np.sum(np.clip(best_of_best_qoe - qoe_ei, 0, None) )
regret_qoe_pi               = np.sum(np.clip(best_of_best_qoe - qoe_pi, 0, None))

print('ours', regret_usage_ours, regret_qoe_ours)
print('ucb', regret_usage_ucb, regret_qoe_ucb)
print('ei', regret_usage_ei, regret_qoe_ei)
print('pi', regret_usage_pi, regret_qoe_pi)
################################################################################################################

fig, ax = plt.subplots(figsize=fig_size)

plt.scatter(usage_pi*100, qoe_pi, color='C3', marker='1', label='PI')

plt.scatter(usage_ei*100, qoe_ei, color='C2',marker='d', label='EI')

plt.scatter(usage_ucb*100, qoe_ucb, color='C1',marker='s', label='GP-UCB')

plt.scatter(usage_ours*100, qoe_ours, color='C0',marker='o', label='Ours')

# plt.arrow(16, 0.8, -2, 0.1, shape='full', width=0.008, head_width = 0.03, head_length=0.4, color='green', edgecolor='green', )
# plt.annotate('Better',(14, 0.92), color='green')

plt.xlabel('Usage (%)',fontsize=font_size)
plt.ylabel('QoE',fontsize=font_size)

plt.legend(prop=dict(size=font_size),loc='lower right')

plt.tight_layout()
plt.grid(which='both',linestyle='--',axis='both',color='gray')
# plt.ylim(0,0.4)
plt.subplots_adjust(left=0.20, bottom=0.19, right=1-0.04, top=1-0.03)
# plt.xlim(left=12, right=30)

plt.savefig("figures/result_online_acq_function.pdf", format = 'pdf', dpi=300)
plt.show()
print('done')








# print('done')
