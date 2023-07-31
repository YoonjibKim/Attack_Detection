import Constant_Parameters
from Data_Extraction import Data_Extraction
from Dataset_Initialization import Dataset_Initialization
from Final_Model import Final_Model
from STAT_Analyser import STAT_Analyser
from STAT_Detection_Model import STAT_Detection_Model
from STAT_Parser import STAT_Parser
from TOP_Detection_Model import TOP_Detection_Model
from TOP_Parser import TOP_Parser
from TOP_Analyser import TOP_Analyser

CID_RCOFF_GOFF: Dataset_Initialization
CID_RCOFF_GON: Dataset_Initialization
CID_RCON_GOFF: Dataset_Initialization
CID_RCON_GON: Dataset_Initialization
WCT_RCOFF_GOFF: Dataset_Initialization
WCT_RCOFF_GON: Dataset_Initialization
WCT_RCON_GOFF: Dataset_Initialization
WCT_RCON_GON: Dataset_Initialization
WET_RCOFF_GOFF: Dataset_Initialization
WET_RCOFF_GON: Dataset_Initialization
WET_RCON_GOFF: Dataset_Initialization
WET_RCON_GON: Dataset_Initialization
WID_RCOFF_GOFF: Dataset_Initialization
WID_RCOFF_GON: Dataset_Initialization
WID_RCON_GOFF: Dataset_Initialization
WID_RCON_GON: Dataset_Initialization


def Initialize_File_Path():
    global CID_RCOFF_GOFF, CID_RCOFF_GON, CID_RCON_GOFF, CID_RCON_GON
    global WCT_RCOFF_GOFF, WCT_RCOFF_GON, WCT_RCON_GOFF, WCT_RCON_GON
    global WET_RCOFF_GOFF, WET_RCOFF_GON, WET_RCON_GOFF, WET_RCON_GON
    global WID_RCOFF_GOFF, WID_RCOFF_GON, WID_RCON_GOFF, WID_RCON_GON

    CID_RCOFF_GOFF = Dataset_Initialization(Constant_Parameters.RAW_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_OFF_ATTACK_PATH,
                                            Constant_Parameters.RAW_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_OFF_NORMAL_PATH)
    CID_RCOFF_GON = Dataset_Initialization(Constant_Parameters.RAW_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_ON_ATTACK_PATH,
                                           Constant_Parameters.RAW_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_ON_NORMAL_PATH)
    CID_RCON_GOFF = Dataset_Initialization(Constant_Parameters.RAW_CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_OFF_ATTACK_PATH,
                                           Constant_Parameters.RAW_CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_OFF_NORMAL_PATH)
    CID_RCON_GON = Dataset_Initialization(Constant_Parameters.RAW_CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_ON_ATTACK_PATH,
                                          Constant_Parameters.RAW_CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_ON_NORMAL_PATH)
    WCT_RCOFF_GOFF = Dataset_Initialization(Constant_Parameters.RAW_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_OFF_ATTACK_PATH,
                                            Constant_Parameters.RAW_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_OFF_NORMAL_PATH)
    WCT_RCOFF_GON = Dataset_Initialization(Constant_Parameters.RAW_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_ON_ATTACK_PATH,
                                           Constant_Parameters.RAW_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_ON_NORMAL_PATH)
    WCT_RCON_GOFF = Dataset_Initialization(Constant_Parameters.RAW_WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_OFF_ATTACK_PATH,
                                           Constant_Parameters.RAW_WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_OFF_NORMAL_PATH)
    WCT_RCON_GON = Dataset_Initialization(Constant_Parameters.RAW_WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_ON_ATTACK_PATH,
                                          Constant_Parameters.RAW_WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_ON_NORMAL_PATH)
    WET_RCOFF_GOFF = Dataset_Initialization(Constant_Parameters.RAW_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_OFF_ATTACK_PATH,
                                            Constant_Parameters.RAW_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_OFF_NORMAL_PATH)
    WET_RCOFF_GON = Dataset_Initialization(Constant_Parameters.RAW_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_ON_ATTACK_PATH,
                                           Constant_Parameters.RAW_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_ON_NORMAL_PATH)
    WET_RCON_GOFF = Dataset_Initialization(Constant_Parameters.RAW_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_OFF_ATTACK_PATH,
                                           Constant_Parameters.RAW_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_OFF_NORMAL_PATH)
    WET_RCON_GON = Dataset_Initialization(Constant_Parameters.RAW_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_ON_ATTACK_PATH,
                                          Constant_Parameters.RAW_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_ON_NORMAL_PATH)
    WID_RCOFF_GOFF = Dataset_Initialization(Constant_Parameters.RAW_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_OFF_ATTACK_PATH,
                                            Constant_Parameters.RAW_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_OFF_NORMAL_PATH)
    WID_RCOFF_GON = Dataset_Initialization(Constant_Parameters.RAW_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_ON_ATTACK_PATH,
                                           Constant_Parameters.RAW_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_ON_NORMAL_PATH)
    WID_RCON_GOFF = Dataset_Initialization(Constant_Parameters.RAW_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_OFF_ATTACK_PATH,
                                           Constant_Parameters.RAW_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_OFF_NORMAL_PATH)
    WID_RCON_GON = Dataset_Initialization(Constant_Parameters.RAW_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_ON_ATTACK_PATH,
                                          Constant_Parameters.RAW_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_ON_NORMAL_PATH)


def Load_Dataset(experiment_flag):
    global CID_RCOFF_GOFF, CID_RCOFF_GON, CID_RCON_GOFF, CID_RCON_GON
    global WCT_RCOFF_GOFF, WCT_RCOFF_GON, WCT_RCON_GOFF, WCT_RCON_GON
    global WET_RCOFF_GOFF, WET_RCOFF_GON, WET_RCON_GOFF, WET_RCON_GON
    global WID_RCOFF_GOFF, WID_RCOFF_GON, WID_RCON_GOFF, WID_RCON_GON

    dataset_init_list = [CID_RCOFF_GOFF, CID_RCOFF_GON, CID_RCON_GOFF, CID_RCON_GON,
                         WCT_RCOFF_GOFF, WCT_RCOFF_GON, WCT_RCON_GOFF, WCT_RCON_GON,
                         WET_RCOFF_GOFF, WET_RCOFF_GON, WET_RCON_GOFF, WET_RCON_GON,
                         WID_RCOFF_GOFF, WID_RCOFF_GON, WID_RCON_GOFF, WID_RCON_GON]

    if experiment_flag == Constant_Parameters.STAT:
        for dataset_init in dataset_init_list:
            stat_parser = STAT_Parser(dataset_init)
            stat_parser.run()
    elif experiment_flag == Constant_Parameters.TOP:
        for dataset_init in dataset_init_list:
            top_parser = TOP_Parser(dataset_init)
            top_parser.run()
        TOP_Parser.extract_overhead_ratio_and_index()
        TOP_Parser.calculate_total_feature_size()
        TOP_Parser.calculate_total_elbow_points()
        TOP_Parser.extract_record_size()
    else:
        print('Wrong Experiment Choice')


def Run_ML(experiment_flag):
    if experiment_flag == Constant_Parameters.STAT:
        for ml_scenario_path in Constant_Parameters.ML_DATASET_PATH_DICT.values():
            stat_detection_model = STAT_Detection_Model(ml_scenario_path)
            stat_detection_model.run()
    elif experiment_flag == Constant_Parameters.TOP:
        for ml_scenario_path in Constant_Parameters.ML_DATASET_PATH_DICT.values():
            top_detection_model = TOP_Detection_Model(ml_scenario_path)
            top_detection_model.run()
    else:
        print('Wrong Experiment Choice')


def Result_Analysis(experiment_flag):
    if experiment_flag == Constant_Parameters.STAT:
        cs_attack_loss_rate_dict, cs_normal_loss_rate_dict = \
            STAT_Analyser.building_combination_loss_rate_heatmap(Constant_Parameters.CS)
        gs_attack_loss_rate_dict, gs_normal_loss_rate_dict = \
            STAT_Analyser.building_combination_loss_rate_heatmap(Constant_Parameters.GS)

        cs_merged_loss_rate_dict = \
            STAT_Analyser.merging_attack_and_normal_loss_rate_dict(cs_attack_loss_rate_dict, cs_normal_loss_rate_dict)
        gs_merged_loss_rate_dict = \
            STAT_Analyser.merging_attack_and_normal_loss_rate_dict(gs_attack_loss_rate_dict, gs_normal_loss_rate_dict)

        STAT_Analyser.drawing_loss_rate(cs_merged_loss_rate_dict, Constant_Parameters.CS)
        STAT_Analyser.drawing_loss_rate(gs_merged_loss_rate_dict, Constant_Parameters.GS)

        for abbreviation, ml_scenario_path in Constant_Parameters.ML_DATASET_PATH_DICT.items():
            stat_analyser = STAT_Analyser(ml_scenario_path)
            stat_analyser.run(cs_merged_loss_rate_dict[abbreviation], gs_merged_loss_rate_dict[abbreviation])
    elif experiment_flag == Constant_Parameters.TOP:
        for abbreviation, ml_scenario_path in Constant_Parameters.ML_DATASET_PATH_DICT.items():
            top_analyser = TOP_Analyser(ml_scenario_path)
            top_analyser.run(abbreviation)
    else:
        print('Wrong Experiment Choice')


def Ensemble_Learning():
    final_model = Final_Model()
    final_model.run()


def Experiment_Data_Extraction():
    data_extraction = Data_Extraction()
    data_extraction.saving_station_id()
    data_extraction.saving_charging_session()
    data_extraction.saving_sim_time()


if __name__ == '__main__':
    print('Simulation Start')

    Initialize_File_Path()
    # Load_Dataset(Constant_Parameters.STAT)
    Load_Dataset(Constant_Parameters.TOP)
    # Run_ML(Constant_Parameters.STAT)
    # Run_ML(Constant_Parameters.TOP)
    # Result_Analysis(Constant_Parameters.STAT)
    # Result_Analysis(Constant_Parameters.TOP)
    # Ensemble_Learning()
    # Experiment_Data_Extraction()

    print('Simulation End')
