
ADDR_CN = 'http://192.168.17.4:8888/'
ADDR_TN = 'http://127.0.0.1:7777/'
ADDR_AN = 'http://127.0.0.1:9999/'

# SLICE_REQUIREMENT = [1, 1, 1] # we have utility function already in smart phone app, [0,2], 1.0 is considered as good



####################################################################################################################################
####################################################################################################################################
################################### offline simulator parameters ###################################################################
####################################################################################################################################


# configure the optimal search parameters from main_offline.py and come back here
# this is optimal for MAR slice under traffic 1, 2, 3, 4
OPTIMAL_PARA_SIM = { 'loading_time_offset': 3.10,
            'baseline_loss': 38.76, 
            'backhaul_offset': 0.69, 
            'backhaul_delay': 6.17, 
            'enb_noise_figure': 5.04, 
            'ue_noise_figure': 8.93,
            'compute_time_mean_offset': 2.16
}

# this all to all is not used
OPTIMAL_ALL_TO_ALL = {
'1':{       'loading_time_offset': 3.10,
            'baseline_loss': 38.76, 
            'backhaul_offset': 0.69, 
            'backhaul_delay': 6.17, 
            'enb_noise_figure': 5.04, 
            'ue_noise_figure': 8.93,
            'compute_time_mean_offset': 2.16
     },
'2': {      'backhaul_delay': 9.10, 
            'backhaul_offset': 0.69, 
            'baseline_loss': 39.28, 
            'compute_time_mean_offset': 12.27, 
            'enb_noise_figure': 5.12, 
            'loading_time_offset': 5.28, 
            'ue_noise_figure': 8.94
     },
'3':{       'backhaul_delay': 9.63, 
            'backhaul_offset': 0.82, 
            'baseline_loss': 41.86, 
            'compute_time_mean_offset': 13.87, 
            'enb_noise_figure': 5.05, 
            'loading_time_offset': 10.24, 
            'ue_noise_figure': 8.81
     },
'4': {      'backhaul_delay': 0.18, 
            'backhaul_offset': 0.99, 
            'baseline_loss': 39.56, 
            'compute_time_mean_offset': 17.35, 
            'enb_noise_figure': 5.06, 
            'loading_time_offset': 1.64, 
            'ue_noise_figure': 8.78
     },
}

# GP model on traffic 2, 3, 4 is not well trained but just copy from traffic 1 TODO 

GPS = {     'loading_time_offset': 6.47, 
            'baseline_loss': 38.58, 
            'backhaul_offset': 1.44, 
            'backhaul_delay': 7.48, 
            'enb_noise_figure': 5.08, 
            'ue_noise_figure': 9.23, 
            'compute_time_mean_offset': 6.02 
}













####################################################################################################################################
####################################################################################################################################
################################### offline training parameters ###################################################################
####################################################################################################################################
from functions import Resource

OPTIMIAL_RES_OUR = Resource(
    bandwidth_ul = 8.69,
    mcs_offset_ul = 0.08,
    bandwidth_dl = 3.06,
    mcs_offset_dl = 0.1,
    backhaul_bw = 6.2,
    cpu_ratio = 0.8
)

OPTIMIAL_RES_DLDA = Resource(
    bandwidth_ul = 11.1,
    mcs_offset_ul = 0.51,
    bandwidth_dl = 3.36,
    mcs_offset_dl = 2.68,
    backhaul_bw = 11.35,
    cpu_ratio = 1.09
)

OPTIMIAL_RES_GPEI = Resource(
    bandwidth_ul = 12.25,
    mcs_offset_ul = 0.89,
    bandwidth_dl = 5.29,
    mcs_offset_dl = 1.24,
    backhaul_bw = 14.94,
    cpu_ratio = 0.87
)

OPTIMIAL_RES_GPPI = Resource(
    bandwidth_ul = 16.61,
    mcs_offset_ul = 0.2,
    bandwidth_dl = 7.89,
    mcs_offset_dl = 5.43,
    backhaul_bw = 3.57,
    cpu_ratio = 1.02
)

OPTIMIAL_RES_GPUCB = Resource(
    bandwidth_ul = 16.61,
    mcs_offset_ul = 0.2,
    bandwidth_dl = 7.89,
    mcs_offset_dl = 5.43,
    backhaul_bw = 3.57,
    cpu_ratio = 1.02
)


#####################################
OPTIMIAL_RES_OUR_500_UE_1 = Resource(
    bandwidth_ul = 6.38,
    mcs_offset_ul = 0.31,
    bandwidth_dl = 3.55,
    mcs_offset_dl = 0.2,
    backhaul_bw = 3.2,
    cpu_ratio = 0.4
)

OPTIMIAL_RES_OUR_500_UE_2 = Resource(
    bandwidth_ul = 8.08,
    mcs_offset_ul = 0.04,
    bandwidth_dl = 3.69,
    mcs_offset_dl = 0.01,
    backhaul_bw = 4.5,
    cpu_ratio = 0.53
)

OPTIMIAL_RES_OUR_500_UE_3 = Resource(
    bandwidth_ul = 9.5,
    mcs_offset_ul = 0.01,
    bandwidth_dl = 3.7,
    mcs_offset_dl = 0.01,
    backhaul_bw = 5.2,
    cpu_ratio = 0.68
)

OPTIMIAL_RES_OUR_500_UE_4 = Resource(
    bandwidth_ul = 11.3,
    mcs_offset_ul = 0.41,
    bandwidth_dl = 3.7,
    mcs_offset_dl = 0.01,
    backhaul_bw = 3.96,
    cpu_ratio = 0.85
)

# OPTIMAL_PARA_SIM = { 'loading_time_offset': 3.0951466029044883,
#             'baseline_loss': 38.764166890059116, 
#             'backhaul_offset': 0.6865789989980509, 
#             'backhaul_delay': 6.170900356040347, 
#             'enb_noise_figure': 5.035094440428589, 
#             'ue_noise_figure': 8.934521317912903,
#             'compute_time_mean_offset': 2.1648477692974932
# }

# # this all to all is not used
# OPTIMAL_ALL_TO_ALL = {
# '1':{       'loading_time_offset': 3.0951466029044883,
#             'baseline_loss': 38.764166890059116, 
#             'backhaul_offset': 0.6865789989980509, 
#             'backhaul_delay': 6.170900356040347, 
#             'enb_noise_figure': 5.035094440428589, 
#             'ue_noise_figure': 8.934521317912903,
#             'compute_time_mean_offset': 2.1648477692974932
#      },
# '2': {      'backhaul_delay': 9.103666447237028, 
#             'backhaul_offset': 0.6915908347748712, 
#             'baseline_loss': 39.2792934251853, 
#             'compute_time_mean_offset': 12.273172315638305, 
#             'enb_noise_figure': 5.115133865845213, 
#             'loading_time_offset': 5.284485552632656, 
#             'ue_noise_figure': 8.944379801496094
#      },
# '3':{       'backhaul_delay': 9.632668311360574, 
#             'backhaul_offset': 0.822114989229199, 
#             'baseline_loss': 41.85559800562795, 
#             'compute_time_mean_offset': 13.870466558245848, 
#             'enb_noise_figure': 5.054106940757463, 
#             'loading_time_offset': 10.243421185405136, 
#             'ue_noise_figure': 8.810253073006951
#      },
# '4': {      'backhaul_delay': 0.18191410434746258, 
#             'backhaul_offset': 0.99811865373165, 
#             'baseline_loss': 39.557099951163366, 
#             'compute_time_mean_offset': 17.34960949109833, 
#             'enb_noise_figure': 5.055040759811046, 
#             'loading_time_offset': 1.640947957446366, 
#             'ue_noise_figure': 8.783752304874607
#      },
# }

# # GP model on traffic 2, 3, 4 is not well trained but just copy from traffic 1 TODO 
# GPS = {     'loading_time_offset': 6.467577468481858, 
#             'baseline_loss': 38.57876516810983, 
#             'backhaul_offset': 1.4372455897451153, 
#             'backhaul_delay': 7.4832834826544214, 
#             'enb_noise_figure': 5.075294218901057, 
#             'ue_noise_figure': 9.227299651318228, 
# }





