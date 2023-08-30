CS_STAT_PATH = 'cs_stat'
CS_TOP_PATH = 'cs_top'
CS_RECORD_PATH = 'cs_record'
GS_STAT_PATH = 'gs_stat'
GS_TOP_PATH = 'gs_top'
GS_RECORD_PATH = 'gs_record'
KERNEL_RECORD_FILENAME = 'kernel_record.json'
CS_AVG_OVERHEAD_FILENAME = 'cs_avg_overhead.json'
CS_AVG_ELBOW_POINT_FILENAME = 'cs_avg_elbow_point.json'
PERF_RECORD = 'perf_record'
CS = 'CS'
GS = 'GS'
COMBINED_SYMBOL = 'combined_symbol'
ATTACK_TIME_DIFF_FILENAME = 'attack_time_diff.txt'
NORMAL_TIME_DIFF_FILENAME = 'normal_time_diff.txt'
CS_ID_PID_FILENAME = 'cs_id_pid.csv'
SIM_SEC_ATTACK_FILENAME = 'sim_sec_attack.txt'
SIM_SEC_NORMAL_FILENAME = 'sim_sec_normal.txt'
TYPE = 'type'
BRANCH = 'branch'
CYCLES = 'cycles'
INSTRUCTIONS = 'instructions'
TYPE_LIST = [BRANCH, CYCLES, INSTRUCTIONS]
BRANCH_MISSES = 'branch-misses'
TIME_DIFF_INIT = 'T'
TIME_DIFF_F1_SUPPORT = 'TIME_DIFF_F1_SUPPORT'
GS_ID = 'GS_1'
CS_ALL_FEATURE_SIZE_FILENAME = 'cs_all_feature_size.json'
GS_ALL_FEATURE_SIZE_FILENAME = 'gs_all_feature_size.json'
ATTACK = 'attack'
NORMAL = 'normal'
ATTACK_EV_NUMBER = 'attack_ev_number'
NORMAL_EV_NUMBER = 'normal_ev_number'
TOTAL_ATTACK_COUNT = 'total_attack_count'
TOTAL_NORMAL_COUNT = 'total_normal_count'
TOTAL_GS_AUTH_COUNT = 'total_auth_count'
CHARGING_SESSION = 'Charging_Session'
NOT_COUNTED = '<not counted>'
GIGA_HERTZ = 'GHz'
MEGA_HERTZ = 'MHz'
KERNEL = '[kernel]'
DATA_POINT = 'data_point'
EXCLUSIVE = 'exclusive'
COMMON = 'common'
ALL = 'all'
SAMPLING_RESOLUTION = 'sampling_resolution'
SAMPLING_COUNT = 'sampling_count'
SIMULATION_TIME = 'simulation_time'
SIMULATION_TIME_DIFF_AVG = 'simulation_time_diff_avg'
AVERAGE_SAMPLING_RESOLUTION = 'average_sampling_resolution'
AVERAGE_SAMPLING_RES_DIFF = 'average_sampling_res_diff'
COMBINED_SAMPLING_RESOLUTION = 'combined_sampling_resolution'
FULL_FEATURE_INFORMATION = 'full_feature_information'
FEATURE_COMBINATION = 'feature_combination'
FINAL_DATASET = 'final_dataset'
FEATURE_SIZE = 'feature_size'
CSR_PROOF = 'csr_proof'
CSR_RATIO = 'csr_ratio'
ORIGINAL_CSR_RATIO = 'original_csr_ratio'
PROCESSED_CSR_RATIO = 'processed_csr_ratio'
CSR_ANALYSIS = 'csr_analysis'
STATISTICS = 'statistics'
P_VALUE = 'p_value'
SUPPORT_AVERAGE = 'support_average'
CSR_AVERAGE = 'csr_average'
CORRELATION_ANALYSIS_FILENAME = 'correlation_analysis.json'
OVERHEAD_INDEX = 'overhead_index'
CS_OVERHEAD_INDEX_FILENAME = 'cs_overhead_index.json'
GS_OVERHEAD_INDEX_FILENAME = 'gs_overhead_index.json'
CS_RECORD_SIZE_FILENAME = 'cs_record_size.json'
GS_RECORD_SIZE_FILENAME = 'gs_record_size.json'
RECORD_SIZE = 'record_size'
CS_K_SIZE_FILENAME = 'cs_k_size.json'
GS_K_SIZE_FILENAME = 'gs_k_size.json'
COMBINATION_LIST = 'combination_list'
COMBINED_LOSS_RATE = 'combination_loss_rate'
MIN_COMBINED_LOSS_RATE = 'min_combination_loss_rate'
NORMALITY = 'NORMALITY'
DUMMY_DATA = -1.0
ATTACK_LABEL = 1
NORMAL_LABEL = 0
TIME_DIFF = 'time_diff'
CID_RCOFF_GOFF = 'cid_rcoff_goff'
CID_RCOFF_GON = 'cid_rcoff_gon'
CID_RCON_GOFF = 'cid_rcon_goff'
CID_RCON_GON = 'cid_rcon_gon'
WCT_RCOFF_GOFF = 'wct_rcoff_goff'
WCT_RCOFF_GON = 'wct_rcoff_gon'
WCT_RCON_GOFF = 'wct_rcon_goff'
WCT_RCON_GON = 'wct_rcon_gon'
WET_RCOFF_GOFF = 'wet_rcoff_goff'
WET_RCOFF_GON = 'wet_rcoff_gon'
WET_RCON_GOFF = 'wet_rcon_goff'
WET_RCON_GON = 'wet_rcon_gon'
WID_RCOFF_GOFF = 'wid_rcoff_goff'
WID_RCOFF_GON = 'wid_rcoff_gon'
WID_RCON_GOFF = 'wid_rcon_goff'
WID_RCON_GON = 'wid_rcon_go'
RAW_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_OFF_ATTACK_PATH = 'Dataset/Raw_Data/Correct_ID/Random_CS_Off/Gaussian_Off/Attack'
RAW_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_OFF_NORMAL_PATH = 'Dataset/Raw_Data/Correct_ID/Random_CS_Off/Gaussian_Off/Normal'
RAW_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_ON_ATTACK_PATH = 'Dataset/Raw_Data/Correct_ID/Random_CS_Off/Gaussian_On/Attack'
RAW_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_ON_NORMAL_PATH = 'Dataset/Raw_Data/Correct_ID/Random_CS_Off/Gaussian_On/Normal'
RAW_CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_OFF_ATTACK_PATH = 'Dataset/Raw_Data/Correct_ID/Random_CS_On/Gaussian_Off/Attack'
RAW_CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_OFF_NORMAL_PATH = 'Dataset/Raw_Data/Correct_ID/Random_CS_On/Gaussian_Off/Normal'
RAW_CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_ON_ATTACK_PATH = 'Dataset/Raw_Data/Correct_ID/Random_CS_On/Gaussian_On/Attack'
RAW_CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_ON_NORMAL_PATH = 'Dataset/Raw_Data/Correct_ID/Random_CS_On/Gaussian_On/Normal'
RAW_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_OFF_ATTACK_PATH = \
    'Dataset/Raw_Data/Wrong_CS_TS/Random_CS_Off/Gaussian_Off/Attack'
RAW_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_OFF_NORMAL_PATH = \
    'Dataset/Raw_Data/Wrong_CS_TS/Random_CS_Off/Gaussian_Off/Normal'
RAW_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_ON_ATTACK_PATH = 'Dataset/Raw_Data/Wrong_CS_TS/Random_CS_Off/Gaussian_On/Attack'
RAW_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_ON_NORMAL_PATH = 'Dataset/Raw_Data/Wrong_CS_TS/Random_CS_Off/Gaussian_On/Normal'
RAW_WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_OFF_ATTACK_PATH = 'Dataset/Raw_Data/Wrong_CS_TS/Random_CS_On/Gaussian_Off/Attack'
RAW_WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_OFF_NORMAL_PATH = 'Dataset/Raw_Data/Wrong_CS_TS/Random_CS_On/Gaussian_Off/Normal'
RAW_WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_ON_ATTACK_PATH = 'Dataset/Raw_Data/Wrong_CS_TS/Random_CS_On/Gaussian_On/Attack'
RAW_WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_ON_NORMAL_PATH = 'Dataset/Raw_Data/Wrong_CS_TS/Random_CS_On/Gaussian_On/Normal'
RAW_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_OFF_ATTACK_PATH = \
    'Dataset/Raw_Data/Wrong_EV_TS/Random_CS_Off/Gaussian_Off/Attack'
RAW_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_OFF_NORMAL_PATH = \
    'Dataset/Raw_Data/Wrong_EV_TS/Random_CS_Off/Gaussian_Off/Normal'
RAW_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_ON_ATTACK_PATH = 'Dataset/Raw_Data/Wrong_EV_TS/Random_CS_Off/Gaussian_On/Attack'
RAW_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_ON_NORMAL_PATH = 'Dataset/Raw_Data/Wrong_EV_TS/Random_CS_Off/Gaussian_On/Normal'
RAW_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_OFF_ATTACK_PATH = 'Dataset/Raw_Data/Wrong_EV_TS/Random_CS_On/Gaussian_Off/Attack'
RAW_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_OFF_NORMAL_PATH = 'Dataset/Raw_Data/Wrong_EV_TS/Random_CS_On/Gaussian_Off/Normal'
RAW_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_ON_ATTACK_PATH = 'Dataset/Raw_Data/Wrong_EV_TS/Random_CS_On/Gaussian_On/Attack'
RAW_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_ON_NORMAL_PATH = 'Dataset/Raw_Data/Wrong_EV_TS/Random_CS_On/Gaussian_On/Normal'
RAW_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_OFF_ATTACK_PATH = 'Dataset/Raw_Data/Wrong_ID/Random_CS_Off/Gaussian_Off/Attack'
RAW_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_OFF_NORMAL_PATH = 'Dataset/Raw_Data/Wrong_ID/Random_CS_Off/Gaussian_Off/Normal'
RAW_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_ON_ATTACK_PATH = 'Dataset/Raw_Data/Wrong_ID/Random_CS_Off/Gaussian_On/Attack'
RAW_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_ON_NORMAL_PATH = 'Dataset/Raw_Data/Wrong_ID/Random_CS_Off/Gaussian_On/Normal'
RAW_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_OFF_ATTACK_PATH = 'Dataset/Raw_Data/Wrong_ID/Random_CS_On/Gaussian_Off/Attack'
RAW_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_OFF_NORMAL_PATH = 'Dataset/Raw_Data/Wrong_ID/Random_CS_On/Gaussian_Off/Normal'
RAW_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_ON_ATTACK_PATH = 'Dataset/Raw_Data/Wrong_ID/Random_CS_On/Gaussian_On/Attack'
RAW_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_ON_NORMAL_PATH = 'Dataset/Raw_Data/Wrong_ID/Random_CS_On/Gaussian_On/Normal'
RAW_DATA_DICT = {CID_RCOFF_GOFF: {ATTACK: RAW_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_OFF_ATTACK_PATH,
                                  NORMAL: RAW_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_OFF_NORMAL_PATH},
                 CID_RCOFF_GON: {ATTACK: RAW_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_ON_ATTACK_PATH,
                                 NORMAL: RAW_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_ON_NORMAL_PATH},
                 CID_RCON_GOFF: {ATTACK: RAW_CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_OFF_ATTACK_PATH,
                                 NORMAL: RAW_CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_OFF_NORMAL_PATH},
                 CID_RCON_GON: {ATTACK: RAW_CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_ON_ATTACK_PATH,
                                NORMAL: RAW_CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_ON_NORMAL_PATH},
                 WCT_RCOFF_GOFF: {ATTACK: RAW_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_OFF_ATTACK_PATH,
                                  NORMAL: RAW_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_OFF_NORMAL_PATH},
                 WCT_RCOFF_GON: {ATTACK: RAW_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_ON_ATTACK_PATH,
                                 NORMAL: RAW_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_ON_NORMAL_PATH},
                 WCT_RCON_GOFF: {ATTACK: RAW_WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_OFF_ATTACK_PATH,
                                 NORMAL: RAW_WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_OFF_NORMAL_PATH},
                 WCT_RCON_GON: {ATTACK: RAW_WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_ON_ATTACK_PATH,
                                NORMAL: RAW_WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_ON_NORMAL_PATH},
                 WET_RCOFF_GOFF: {ATTACK: RAW_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_OFF_ATTACK_PATH,
                                  NORMAL: RAW_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_OFF_NORMAL_PATH},
                 WET_RCOFF_GON: {ATTACK: RAW_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_ON_ATTACK_PATH,
                                 NORMAL: RAW_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_ON_NORMAL_PATH},
                 WET_RCON_GOFF: {ATTACK: RAW_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_OFF_ATTACK_PATH,
                                 NORMAL: RAW_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_OFF_NORMAL_PATH},
                 WET_RCON_GON: {ATTACK: RAW_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_ON_ATTACK_PATH,
                                NORMAL: RAW_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_ON_NORMAL_PATH},
                 WID_RCOFF_GOFF: {ATTACK: RAW_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_OFF_ATTACK_PATH,
                                  NORMAL: RAW_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_OFF_NORMAL_PATH},
                 WID_RCOFF_GON: {ATTACK: RAW_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_ON_ATTACK_PATH,
                                 NORMAL: RAW_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_ON_NORMAL_PATH},
                 WID_RCON_GOFF: {ATTACK: RAW_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_OFF_ATTACK_PATH,
                                 NORMAL: RAW_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_OFF_NORMAL_PATH},
                 WID_RCON_GON: {ATTACK: RAW_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_ON_ATTACK_PATH,
                                NORMAL: RAW_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_ON_NORMAL_PATH}}
AUTHENTICATION_RESULTS = 'authentication_results'
DATASET_PROCESSED_DATA_PATH = 'Dataset/Processed_Data'
STAT_PATH = 'STAT'
TOP_PATH = 'TOP'
TOP = TOP_PATH
STAT = STAT_PATH
RATIO_INFORMATION_FILE_NAME = 'ratio_information.json'
ELBOW_POINT_INFORMATION_FILE_NAME = 'elbow_point_information.json'
K = '[k]'
SILHOUETTE_SCORE = 'silhouette_score'
BEST_K = 'best_k'
BEST_SILHOUETTE_SCORE = 'best_silhouette_score'
RESULT = 'Result'
CS_BEST_FILENAME = 'cs_best.json'
GS_BEST_FILENAME = 'gs_best.json'
OTHER_SILHOUETTE_SCORES = 'other_silhouette_scores'
RECORD_INFORMATION = 'record_information'
PERCENT = '%'
OVERHEAD_RATIO = 'overhead_ratio'
OVERHEAD_RATIO_MEAN = 'overhead_ratio_mean'
BEST_SCORE_FILE_NAME = 'best_score'
SYMBOL_INDEX = 'symbol_index'
SYMBOL_INDEX_MEAN = 'symbol_index_mean'
EXPERIMENT_DATA = 'Experiment_Data'
STATION_ID = 'Station_ID'
RAW_DATA = 'Raw_Data'
ADA_BOOST = 'AB'
AGGLOMERATIVE_CLUSTERING = 'AC'
DB_SCAN = 'DS'
DNN = 'DNN'
DECISION_TREE = 'DT'
GAUSSIAN_MIXTURE = 'GM'
GAUSSIAN_NB = 'GN'
GRADIENT_BOOST = 'GB'
KNN = 'KNN'
KMEANS = 'KM'
LINEAR_REGRESSION = 'LR'
LINEAR_REGRESSION_RIDGE = 'LRR'
LINEAR_REGRESSION_LASSO = 'LRL'
LINEAR_REGRESSION_ELASTIC = 'LRE'
LOGISTIC_REGRESSION = 'LOG_R'
RANDOM_FOREST = 'RF'
SVM = 'SVM'
ML_LIST = [ADA_BOOST, AGGLOMERATIVE_CLUSTERING, DB_SCAN, DECISION_TREE, GAUSSIAN_MIXTURE, GAUSSIAN_NB,
           GRADIENT_BOOST, KNN, KMEANS, LINEAR_REGRESSION, LINEAR_REGRESSION_RIDGE, LINEAR_REGRESSION_LASSO,
           LINEAR_REGRESSION_ELASTIC, LOGISTIC_REGRESSION, RANDOM_FOREST, SVM, DNN]  # dnn 빠짐
TRAINING_SET_RATIO = 0.7
ACCURACY = 'accuracy'
PRECISION = 'precision'
RECALL = 'recall'
F1_SCORE = 'f1_score'
SUPPORT = 'support'
ML_SCORE_DIR_PATH = 'ML/Score'
RESULT_STAT_DIR_PATH = 'Result/' + STAT_PATH
RESULT_TOP_DIR_PATH = 'Result/' + TOP_PATH
ML_DATASET_PATH = 'ML/Dataset'
COMBINATION_TYPE = 'combination_type'
ML_TYPE = 'ml_type'
ML = 'ML'

STAT_TIME_DIFF = 'STAT_TIME_DIFF'
FINAL_SUGGESTION = 'FINAL_SUGGESTION'
RESULT_FINAL_SUGGESTION_PATH = 'Result/' + FINAL_SUGGESTION
BEST_FEATURE_PATH = 'Sub_Best_Feature'
F1_SCORE_PATH = 'F1_Score'
SCORE = 'Score'
BEST_FEATURE = 'Best_Feature'
CSR = 'CSR'
STATISTICS_PATH = BEST_FEATURE
BASIS_SYMBOL = 'basis_symbol'
F1_SCORE_AVERAGE = 'f1_score_average'
F1_SCORE_MEDIAN = 'f1_score_median'
SAM_RES_DIFF_AND_MEAN = 'sam_res_diff_and_mean'
UNIQUE_INTERSECTION = 'unique_intersection'
MIN_SAMPLE_COUNT = 'min_sample_count'
TRAINING_FEATURE = 'training_feature'
TRAINING_LABEL = 'training_label'
TESTING_FEATURE = 'testing_feature'
TESTING_LABEL = 'testing_label'
CID_RCOFF_GOFF = 'cid_rcoff_goff'
CID_RCOFF_GON = 'cid_rcoff_gon'
CID_RCON_GOFF = 'cid_rcon_goff'
CID_RCON_GON = 'cid_rcon_gon'
WCT_RCOFF_GOFF = 'wct_rcoff_goff'
WCT_RCOFF_GON = 'wct_rcoff_gon'
WCT_RCON_GOFF = 'wct_rcon_goff'
WCT_RCON_GON = 'wct_rcon_gon'
WET_RCOFF_GOFF = 'wet_rcoff_goff'
WET_RCOFF_GON = 'wet_rcoff_gon'
WET_RCON_GOFF = 'wet_rcon_goff'
WET_RCON_GON = 'wet_rcon_gon'
WID_RCOFF_GOFF = 'wid_rcoff_goff'
WID_RCOFF_GON = 'wid_rcoff_gon'
WID_RCON_GOFF = 'wid_rcon_goff'
WID_RCON_GON = 'wid_rcon_go'
INITIAL_SCENARIO_NAME_LIST = [CID_RCOFF_GOFF, CID_RCOFF_GON, CID_RCON_GOFF, CID_RCON_GON, WCT_RCOFF_GOFF, WCT_RCOFF_GON,
                              WCT_RCON_GOFF, WCT_RCON_GON, WET_RCOFF_GOFF, WET_RCOFF_GON, WET_RCON_GOFF, WET_RCON_GON,
                              WID_RCOFF_GOFF, WID_RCOFF_GON, WID_RCON_GOFF, WID_RCON_GON]
CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_OFF = 'Correct_ID_Random_CS_Off_Gaussian_Off'
CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_ON = 'Correct_ID_Random_CS_Off_Gaussian_On'
CORRECT_ID_RANDOM_CS_ON_GAUSSAIN_OFF = 'Correct_ID_Random_CS_On_Gaussian_Off'
CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_ON = 'Correct_ID_Random_CS_On_Gaussian_On'
WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_OFF = 'Wrong_CS_TS_Random_CS_Off_Gaussian_Off'
WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_ON = 'Wrong_CS_TS_Random_CS_Off_Gaussian_On'
WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_OFF = 'Wrong_CS_TS_Random_CS_On_Gaussian_Off'
WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_ON = 'Wrong_CS_TS_Random_CS_On_Gaussian_On'
WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_OFF = 'Wrong_EV_TS_Random_CS_Off_Gaussian_Off'
WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_ON = 'Wrong_EV_TS_Random_CS_Off_Gaussian_On'
WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_OFF = 'Wrong_EV_TS_Random_CS_On_Gaussian_Off'
WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_ON = 'Wrong_EV_TS_Random_CS_On_Gaussian_On'
WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_OFF = 'Wrong_ID_Random_CS_Off_Gaussian_Off'
WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_ON = 'Wrong_ID_Random_CS_Off_Gaussian_On'
WRONG_ID_RANDOM_CS_ON_GAUSSIAN_OFF = 'Wrong_ID_Random_CS_On_Gaussian_Off'
WRONG_ID_RANDOM_CS_ON_GAUSSIAN_ON = 'Wrong_ID_Random_CS_On_Gaussian_On'
FULL_SCENARIO_NAME_LIST = [CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_OFF, CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_ON,
                           CORRECT_ID_RANDOM_CS_ON_GAUSSAIN_OFF, CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_ON,
                           WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_OFF, WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_ON,
                           WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_OFF, WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_ON,
                           WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_OFF, WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_ON,
                           WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_OFF, WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_ON,
                           WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_OFF, WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_ON,
                           WRONG_ID_RANDOM_CS_ON_GAUSSIAN_OFF, WRONG_ID_RANDOM_CS_ON_GAUSSIAN_ON]
PROCESSED_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_OFF_PATH = 'Dataset/Processed_Data/Correct_ID/Random_CS_Off/Gaussian_Off'
PROCESSED_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_ON_PATH = 'Dataset/Processed_Data/Correct_ID/Random_CS_Off/Gaussian_On'
PROCESSED_CORRECT_ID_RANDOM_CS_ON_GAUSSAIN_OFF_PATH = 'Dataset/Processed_Data/Correct_ID/Random_CS_On/Gaussian_Off'
PROCESSED_CORRECT_ID_RANDOM_CS_ON_GAUSSAIN_ON_PATH = 'Dataset/Processed_Data/Correct_ID/Random_CS_On/Gaussian_On'
PROCESSED_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSAIN_OFF_PATH = 'Dataset/Processed_Data/Wrong_CS_TS/Random_CS_Off/Gaussian_Off'
PROCESSED_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSAIN_ON_PATH = 'Dataset/Processed_Data/Wrong_CS_TS/Random_CS_Off/Gaussian_On'
PROCESSED_WRONG_CS_TS_RANDOM_CS_ON_GAUSSAIN_OFF_PATH = 'Dataset/Processed_Data/Wrong_CS_TS/Random_CS_On/Gaussian_Off'
PROCESSED_WRONG_CS_TS_RANDOM_CS_ON_GAUSSAIN_ON_PATH = 'Dataset/Processed_Data/Wrong_CS_TS/Random_CS_On/Gaussian_On'
PROCESSED_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_OFF_PATH = 'Dataset/Processed_Data/Wrong_EV_TS/Random_CS_Off/Gaussian_Off'
PROCESSED_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_ON_PATH = 'Dataset/Processed_Data/Wrong_EV_TS/Random_CS_Off/Gaussian_On'
PROCESSED_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_OFF_PATH = 'Dataset/Processed_Data/Wrong_EV_TS/Random_CS_On/Gaussian_Off'
PROCESSED_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_ON_PATH = 'Dataset/Processed_Data/Wrong_EV_TS/Random_CS_On/Gaussian_On'
PROCESSED_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_OFF_PATH = 'Dataset/Processed_Data/Wrong_ID/Random_CS_Off/Gaussian_Off'
PROCESSED_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_ON_PATH = 'Dataset/Processed_Data/Wrong_ID/Random_CS_Off/Gaussian_On'
PROCESSED_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_OFF_PATH = 'Dataset/Processed_Data/Wrong_ID/Random_CS_On/Gaussian_Off'
PROCESSED_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_ON_PATH = 'Dataset/Processed_Data/Wrong_ID/Random_CS_On/Gaussian_On'
PROCESSED_DATASET_PATH_DICT = {CID_RCOFF_GOFF: PROCESSED_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_OFF_PATH,
                               CID_RCOFF_GON: PROCESSED_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_ON_PATH,
                               CID_RCON_GOFF: PROCESSED_CORRECT_ID_RANDOM_CS_ON_GAUSSAIN_OFF_PATH,
                               CID_RCON_GON: PROCESSED_CORRECT_ID_RANDOM_CS_ON_GAUSSAIN_ON_PATH,
                               WCT_RCOFF_GOFF: PROCESSED_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSAIN_OFF_PATH,
                               WCT_RCOFF_GON: PROCESSED_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSAIN_ON_PATH,
                               WCT_RCON_GOFF: PROCESSED_WRONG_CS_TS_RANDOM_CS_ON_GAUSSAIN_OFF_PATH,
                               WCT_RCON_GON: PROCESSED_WRONG_CS_TS_RANDOM_CS_ON_GAUSSAIN_ON_PATH,
                               WET_RCOFF_GOFF: PROCESSED_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_OFF_PATH,
                               WET_RCOFF_GON: PROCESSED_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_ON_PATH,
                               WET_RCON_GOFF: PROCESSED_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_OFF_PATH,
                               WET_RCON_GON: PROCESSED_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_ON_PATH,
                               WID_RCOFF_GOFF: PROCESSED_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_OFF_PATH,
                               WID_RCOFF_GON: PROCESSED_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_ON_PATH,
                               WID_RCON_GOFF: PROCESSED_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_OFF_PATH,
                               WID_RCON_GON: PROCESSED_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_ON_PATH}
ML_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_OFF_PATH = 'ML/Dataset/Correct_ID/Random_CS_Off/Gaussian_Off'
ML_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_ON_PATH = 'ML/Dataset/Correct_ID/Random_CS_Off/Gaussian_On'
ML_CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_OFF_PATH = 'ML/Dataset/Correct_ID/Random_CS_On/Gaussian_Off'
ML_CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_ON_PATH = 'ML/Dataset/Correct_ID/Random_CS_On/Gaussian_On'
ML_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_OFF = 'ML/Dataset/Wrong_CS_TS/Random_CS_Off/Gaussian_Off'
ML_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_ON = 'ML/Dataset/Wrong_CS_TS/Random_CS_Off/Gaussian_On'
ML_WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_OFF = 'ML/Dataset/Wrong_CS_TS/Random_CS_On/Gaussian_Off'
ML_WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_ON = 'ML/Dataset/Wrong_CS_TS/Random_CS_On/Gaussian_On'
ML_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_OFF = 'ML/Dataset/Wrong_EV_TS/Random_CS_Off/Gaussian_Off'
ML_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_ON = 'ML/Dataset/Wrong_EV_TS/Random_CS_Off/Gaussian_On'
ML_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_OFF = 'ML/Dataset/Wrong_EV_TS/Random_CS_On/Gaussian_Off'
ML_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_ON = 'ML/Dataset/Wrong_EV_TS/Random_CS_On/Gaussian_On'
ML_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_OFF = 'ML/Dataset/Wrong_ID/Random_CS_Off/Gaussian_Off'
ML_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_ON = 'ML/Dataset/Wrong_ID/Random_CS_Off/Gaussian_On'
ML_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_OFF = 'ML/Dataset/Wrong_ID/Random_CS_On/Gaussian_Off'
ML_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_ON = 'ML/Dataset/Wrong_ID/Random_CS_On/Gaussian_On'
ML_DATASET_PATH_DICT = {CID_RCOFF_GOFF: ML_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_OFF_PATH,
                        CID_RCOFF_GON: ML_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_ON_PATH,
                        CID_RCON_GOFF: ML_CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_OFF_PATH,
                        CID_RCON_GON: ML_CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_ON_PATH,
                        WCT_RCOFF_GOFF: ML_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_OFF,
                        WCT_RCOFF_GON: ML_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_ON,
                        WCT_RCON_GOFF: ML_WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_OFF,
                        WCT_RCON_GON: ML_WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_ON,
                        WET_RCOFF_GOFF: ML_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_OFF,
                        WET_RCOFF_GON: ML_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_ON,
                        WET_RCON_GOFF: ML_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_OFF,
                        WET_RCON_GON: ML_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_ON,
                        WID_RCOFF_GOFF: ML_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_OFF,
                        WID_RCOFF_GON: ML_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_ON,
                        WID_RCON_GOFF: ML_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_OFF,
                        WID_RCON_GON: ML_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_ON}
TIME_DIFF_ML = 'time_diff_ml'
TIME_DIFF_F1_SCORE = 'time_diff_f1_score'
STAT_ML = 'stat_ml'
STAT_FEATURE_COMBINATION = 'stat_feature_combination'
STAT_F1_SCORE = 'stat_f1_score'
TOP_TYPE = 'top_type'
TOP_COMBINATION = 'top_combination'
TOP_FEATURE_COMB_LIST = 'top_feature_comb_list'
TOP_ML_TYPE = 'top_ml_type'
TOP_F1_SCORE = 'top_f1_score'
