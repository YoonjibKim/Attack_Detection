import csv
import json
import Constant_Parameters


class Final_Model:
    __cs_best_time_diff_stat_dict: dict
    __gs_best_time_diff_stat_dict: dict
    __cs_best_top_dict: dict
    __gs_best_top_dict: dict

    def __init__(self):
        self.__cs_best_time_diff_stat_dict = self.__get_time_diff_and_stat_best_score_dict(Constant_Parameters.CS)
        self.__gs_best_time_diff_stat_dict = self.__get_time_diff_and_stat_best_score_dict(Constant_Parameters.GS)
        self.__cs_best_top_dict = self.__get_top_best_score_dict(Constant_Parameters.CS)
        self.__gs_best_top_dict = self.__get_top_best_score_dict(Constant_Parameters.GS)

    def run(self):
        best_feature_comb_dict = {}
        for scenario_type in Constant_Parameters.INITIAL_SCENARIO_NAME_LIST:
            cs_best_time_diff_stat_dict = self.__cs_best_time_diff_stat_dict[self.__get_full_name(scenario_type)]
            cs_best_top_dict = self.__cs_best_top_dict[scenario_type]
            gs_best_time_diff_stat_dict = self.__gs_best_time_diff_stat_dict[self.__get_full_name(scenario_type)]
            gs_best_top_dict = self.__gs_best_top_dict[scenario_type]

            param_cs_dict = {Constant_Parameters.STAT_TIME_DIFF: cs_best_time_diff_stat_dict,
                             Constant_Parameters.TOP: cs_best_top_dict}
            param_gs_dict = {Constant_Parameters.STAT_TIME_DIFF: gs_best_time_diff_stat_dict,
                             Constant_Parameters.TOP: gs_best_top_dict}

            best_feature_comb_dict[scenario_type] = {Constant_Parameters.CS: param_cs_dict,
                                                     Constant_Parameters.GS: param_gs_dict}

        path = Constant_Parameters.RESULT_FINAL_SUGGESTION_PATH + '/' + Constant_Parameters.FINAL_SUGGESTION + '.json'
        with open(path, 'w') as f:
            json.dump(best_feature_comb_dict, f)

    @classmethod
    def __get_top_best_score_dict(cls, station_type):
        root_path = Constant_Parameters.RESULT_TOP_DIR_PATH + '/' + station_type
        best_score_dict = {}
        for sub_initial_name in Constant_Parameters.INITIAL_SCENARIO_NAME_LIST:
            print('top', station_type, sub_initial_name)
            file_name = sub_initial_name + '_' + Constant_Parameters.BEST_SCORE_FILE_NAME + '.json'
            file_path = root_path + '/' + file_name
            with open(file_path, 'r') as f_1:
                temp_score_dict = json.load(f_1)
                feature_combination_index_str: str = temp_score_dict[Constant_Parameters.FEATURE_COMBINATION]
                feature_combination_index_list = feature_combination_index_str.split(' + ')
                combination_type = temp_score_dict[Constant_Parameters.COMBINATION_TYPE]
                feature_type = temp_score_dict[Constant_Parameters.TYPE]
                comb_index_dir_path = root_path + '/' + combination_type + '/' + Constant_Parameters.F1_SCORE
                comb_index_file_path = comb_index_dir_path + '/' + sub_initial_name + '_' + feature_type + '_' + \
                                       Constant_Parameters.SYMBOL_INDEX + '.json'

                symbol_full_name_list = []
                with open(comb_index_file_path, 'r') as f_2:
                    index_dict = json.load(f_2)
                    for index in feature_combination_index_list:
                        symbol_name = index_dict[index]
                        symbol_full_name_list.append(symbol_name)

            feature_combination = symbol_full_name_list
            ml_type = temp_score_dict[Constant_Parameters.ML_TYPE]
            comb_loss_rate = temp_score_dict[Constant_Parameters.COMBINATION_LOSS_RATE]
            f1_score = temp_score_dict[Constant_Parameters.F1_SCORE]
            support = temp_score_dict[Constant_Parameters.SUPPORT]

            param_dict = {Constant_Parameters.TYPE: feature_type,
                          Constant_Parameters.COMBINATION_TYPE: combination_type,
                          Constant_Parameters.FEATURE_COMBINATION: feature_combination,
                          Constant_Parameters.ML_TYPE: ml_type,
                          Constant_Parameters.COMBINATION_LOSS_RATE: comb_loss_rate,
                          Constant_Parameters.F1_SCORE: f1_score,
                          Constant_Parameters.SUPPORT: support}

            best_score_dict[sub_initial_name] = param_dict

        return best_score_dict

    @classmethod
    def __get_time_diff_and_stat_best_score_dict(cls, station_type):
        root_path = Constant_Parameters.RESULT_STAT_DIR_PATH + '/' + station_type + '/' \
                    + Constant_Parameters.BEST_FEATURE

        best_score_dict = {}
        for basic_scenario_name in Constant_Parameters.FULL_SCENARIO_NAME_LIST:
            print('stat', station_type, basic_scenario_name)
            file_path = root_path + '/' + basic_scenario_name + '_' + Constant_Parameters.STATISTICS + '.csv'

            with open(file_path, 'r') as f:
                rdr = csv.reader(f)
                for index, line in enumerate(rdr):
                    if index > 4:
                        best_score_list = line

            param_dict = {Constant_Parameters.COMBINATION_TYPE: best_score_list[0],
                          Constant_Parameters.ML_TYPE: best_score_list[1],
                          Constant_Parameters.COMBINATION_LOSS_RATE: best_score_list[2],
                          Constant_Parameters.F1_SCORE: best_score_list[3],
                          Constant_Parameters.SUPPORT: best_score_list[4],
                          Constant_Parameters.F1_SCORE_AVERAGE: best_score_list[5],
                          Constant_Parameters.F1_SCORE_MEDIAN: best_score_list[6]}
            best_score_dict[basic_scenario_name] = param_dict

        return best_score_dict

    @classmethod
    def __get_full_name(cls, scenario_name):
        full_name = ''
        if scenario_name == Constant_Parameters.CID_RCOFF_GOFF:
            full_name = Constant_Parameters.CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_OFF
        elif scenario_name == Constant_Parameters.CID_RCOFF_GON:
            full_name = Constant_Parameters.CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_ON
        elif scenario_name == Constant_Parameters.CID_RCON_GOFF:
            full_name = Constant_Parameters.CORRECT_ID_RANDOM_CS_ON_GAUSSAIN_OFF
        elif scenario_name == Constant_Parameters.CID_RCON_GON:
            full_name = Constant_Parameters.CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_ON
        elif scenario_name == Constant_Parameters.WCT_RCOFF_GOFF:
            full_name = Constant_Parameters.WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_OFF
        elif scenario_name == Constant_Parameters.WCT_RCOFF_GON:
            full_name = Constant_Parameters.WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_ON
        elif scenario_name == Constant_Parameters.WCT_RCON_GOFF:
            full_name = Constant_Parameters.WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_OFF
        elif scenario_name == Constant_Parameters.WCT_RCON_GON:
            full_name = Constant_Parameters.WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_ON
        elif scenario_name == Constant_Parameters.WET_RCOFF_GOFF:
            full_name = Constant_Parameters.WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_OFF
        elif scenario_name == Constant_Parameters.WET_RCOFF_GON:
            full_name = Constant_Parameters.WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_ON
        elif scenario_name == Constant_Parameters.WET_RCON_GOFF:
            full_name = Constant_Parameters.WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_OFF
        elif scenario_name == Constant_Parameters.WET_RCON_GON:
            full_name = Constant_Parameters.WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_ON
        elif scenario_name == Constant_Parameters.WID_RCOFF_GOFF:
            full_name = Constant_Parameters.WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_OFF
        elif scenario_name == Constant_Parameters.WID_RCOFF_GON:
            full_name = Constant_Parameters.WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_ON
        elif scenario_name == Constant_Parameters.WID_RCON_GOFF:
            full_name = Constant_Parameters.WRONG_ID_RANDOM_CS_ON_GAUSSIAN_OFF
        elif scenario_name == Constant_Parameters.WID_RCON_GON:
            full_name = Constant_Parameters.WRONG_ID_RANDOM_CS_ON_GAUSSIAN_ON

        return full_name
