import csv
import json
import os
import Constant_Parameters


class Final_Model:
    __cs_best_score_dict: dict
    __gs_best_score_dict: dict

    def __init__(self):
        cs_time_diff_dict, gs_time_diff_dict = self.__get_time_diff_f1_score_dict()
        cs_stat_dict, gs_stat_dict = self.__get_stat_f1_score_dict()
        cs_top_dict, gs_top_dict = self.__get_top_score_dict()

        self.__cs_best_score_dict = \
            self.__get_best_score_dict(cs_time_diff_dict, cs_stat_dict, cs_top_dict, Constant_Parameters.CS)
        self.__gs_best_score_dict = \
            self.__get_best_score_dict(gs_time_diff_dict, gs_stat_dict, gs_top_dict, Constant_Parameters.GS)

    def get_best_score(self):
        print('Best Score')
        print('CS: ', self.__cs_best_score_dict)
        print('GS: ', self.__gs_best_score_dict)

    @classmethod
    def __get_best_score_dict(cls, time_diff_dict, stat_dict, top_dict, station_type):
        default_top_path = Constant_Parameters.RESULT_TOP_DIR_PATH + '/' + station_type

        for scenario_name, time_diff_list in time_diff_dict.items():
            stat_list = stat_dict[scenario_name]

            top_scenario_initial = cls.__get_abbreviation(scenario_name)
            param_top_dict = top_dict[top_scenario_initial]

            feature_combination_list = param_top_dict[Constant_Parameters.FEATURE_COMBINATION]
            feature_index_list = feature_combination_list.split(' + ')

            combination_type = param_top_dict[Constant_Parameters.COMBINATION_TYPE]
            temp_path = default_top_path + '/' + combination_type + '/' + Constant_Parameters.F1_SCORE
            symbol_index_file_path = temp_path + '/' + top_scenario_initial + Constant_Parameters.SYMBOL_INDEX
            with open(symbol_index_file_path, 'r') as f:
                symbol_index_dict = json.load(f)
            symbol_name_list = []
            for feature_index in feature_index_list:
                symbol_name = symbol_index_dict[feature_index]
                symbol_name_list.append(symbol_name)

            time_diff_ml = time_diff_list[0]
            time_diff_f1_score = time_diff_list[1]

            stat_ml = stat_list[0]
            stat_feature_combination = stat_list[1]
            stat_f1_score = stat_list[3]

            top_type = param_top_dict[Constant_Parameters.TYPE]
            top_combination = feature_index_list
            top_feature_comb_list = feature_index_list
            top_ml_type = param_top_dict[Constant_Parameters.ML_TYPE]
            top_f1_score = param_top_dict[Constant_Parameters.F1_SCORE]

            param_dict = {Constant_Parameters.TIME_DIFF_ML: time_diff_ml,
                          Constant_Parameters.TIME_DIFF_F1_SCORE: time_diff_f1_score,
                          Constant_Parameters.STAT_ML: stat_ml,
                          Constant_Parameters.STAT_FEATURE_COMBINATION: stat_feature_combination,
                          Constant_Parameters.STAT_F1_SCORE: stat_f1_score,
                          Constant_Parameters.TOP_TYPE: top_type,
                          Constant_Parameters.TOP_COMBINATION: top_combination,
                          Constant_Parameters.TOP_FEATURE_COMB_LIST: top_feature_comb_list,
                          Constant_Parameters.TOP_ML_TYPE: top_ml_type,
                          Constant_Parameters.TOP_F1_SCORE: top_f1_score}

            return param_dict

    @classmethod
    def __get_abbreviation(cls, scenario_name):
        initial_name = ''
        if scenario_name == Constant_Parameters.CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_OFF:
            initial_name = Constant_Parameters.CID_RCOFF_GOFF
        elif scenario_name == Constant_Parameters.CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_ON:
            initial_name = Constant_Parameters.CID_RCOFF_GON
        elif scenario_name == Constant_Parameters.CORRECT_ID_RANDOM_CS_ON_GAUSSAIN_OFF:
            initial_name = Constant_Parameters.CID_RCON_GOFF
        elif scenario_name == Constant_Parameters.CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_ON:
            initial_name = Constant_Parameters.CID_RCON_GON
        elif scenario_name == Constant_Parameters.WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_OFF:
            initial_name = Constant_Parameters.WCT_RCOFF_GOFF
        elif scenario_name == Constant_Parameters.WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_ON:
            initial_name = Constant_Parameters.WCT_RCOFF_GON
        elif scenario_name == Constant_Parameters.WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_OFF:
            initial_name = Constant_Parameters.WCT_RCON_GOFF
        elif scenario_name == Constant_Parameters.WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_ON:
            initial_name = Constant_Parameters.WCT_RCON_GON
        elif scenario_name == Constant_Parameters.WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_OFF:
            initial_name = Constant_Parameters.WET_RCOFF_GOFF
        elif scenario_name == Constant_Parameters.WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_ON:
            initial_name = Constant_Parameters.WET_RCOFF_GON
        elif scenario_name == Constant_Parameters.WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_OFF:
            initial_name = Constant_Parameters.WET_RCON_GOFF
        elif scenario_name == Constant_Parameters.WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_ON:
            initial_name = Constant_Parameters.WET_RCON_GON
        elif scenario_name == Constant_Parameters.WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_OFF:
            initial_name = Constant_Parameters.WID_RCOFF_GOFF
        elif scenario_name == Constant_Parameters.WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_ON:
            initial_name = Constant_Parameters.WID_RCOFF_GON
        elif scenario_name == Constant_Parameters.WRONG_ID_RANDOM_CS_ON_GAUSSIAN_OFF:
            initial_name = Constant_Parameters.WID_RCON_GOFF
        elif scenario_name == Constant_Parameters.WRONG_ID_RANDOM_CS_ON_GAUSSIAN_ON:
            initial_name = Constant_Parameters.WID_RCON_GON

        return initial_name

    @classmethod
    def __get_top_score_dict(cls):
        default_path = Constant_Parameters.RESULT_TOP_DIR_PATH
        cs_path = default_path + '/' + Constant_Parameters.CS
        gs_path = default_path + '/' + Constant_Parameters.GS

        cs_all_file_list = os.listdir(cs_path)
        cs_best_score_file_list = [file for file in cs_all_file_list if file.endswith(".json")]
        gs_all_file_list = os.listdir(gs_path)
        gs_best_score_file_list = [file for file in gs_all_file_list if file.endswith(".json")]

        cs_score_dict = {}
        for file_name in cs_best_score_file_list:
            file_path = cs_path + '/' + file_name

            with open(file_path, 'r') as f:
                temp_dict = json.load(f)

            trimmed_file_name = file_name.replace(Constant_Parameters.BEST_SCORE_FILE_NAME + '.json', '')
            cs_score_dict[trimmed_file_name] = temp_dict

        gs_score_dict = {}
        for file_name in gs_best_score_file_list:
            file_path = gs_path + '/' + file_name

            with open(file_path, 'r') as f:
                temp_dict = json.load(f)

            trimmed_file_name = file_name.replace(Constant_Parameters.BEST_SCORE_FILE_NAME + '.json', '')
            gs_score_dict[trimmed_file_name] = temp_dict

        return cs_score_dict, gs_score_dict

    @classmethod
    def __get_stat_f1_score_dict(cls):
        default_path = Constant_Parameters.RESULT_STAT_DIR_PATH
        cs_path = default_path + '/' + Constant_Parameters.CS + '/' + Constant_Parameters.BEST_FEATURE_PATH
        gs_path = default_path + '/' + Constant_Parameters.GS + '/' + Constant_Parameters.BEST_FEATURE_PATH

        cs_all_file_list = os.listdir(cs_path)
        cs_best_score_file_list = [file for file in cs_all_file_list if file.endswith('.csv')]
        gs_all_file_list = os.listdir(gs_path)
        gs_best_score_file_list = [file for file in gs_all_file_list if file.endswith('.csv')]

        cs_score_dict = {}
        for file_name in cs_best_score_file_list:
            file_path = cs_path + '/' + file_name
            score_list = []
            with open(file_path, 'r') as f:
                rdr = csv.reader(f)
                for line in rdr:
                    score_list.append(line)

            sorted_score_list = sorted(score_list[1:], key=lambda x: (float(x[2]), -float(x[3])))
            modified_file_name = file_name.replace(Constant_Parameters.BEST_FEATURE + '.csv', '')
            cs_score_dict[modified_file_name] = sorted_score_list[0]

        gs_score_dict = {}
        for file_name in gs_best_score_file_list:
            file_path = gs_path + '/' + file_name
            score_list = []
            with open(file_path, 'r') as f:
                rdr = csv.reader(f)
                for line in rdr:
                    score_list.append(line)

            sorted_score_list = sorted(score_list[1:], key=lambda x: (float(x[2]), -float(x[3])))
            modified_file_name = file_name.replace(Constant_Parameters.BEST_FEATURE + '.csv', '')
            gs_score_dict[modified_file_name] = sorted_score_list[0]

        return cs_score_dict, gs_score_dict

    @classmethod
    def __get_time_diff_f1_score_dict(cls):
        ml_score_stat_path = Constant_Parameters.ML_SCORE_DIR_PATH + '/' + Constant_Parameters.STAT_PATH
        ml_score_stat_cs_path = ml_score_stat_path + '/' + Constant_Parameters.CS
        ml_score_stat_gs_path = ml_score_stat_path + '/' + Constant_Parameters.GS

        cs_all_file_list = os.listdir(ml_score_stat_cs_path)
        cs_best_score_file_list = [file for file in cs_all_file_list if file.endswith(".json")]
        gs_all_file_list = os.listdir(ml_score_stat_gs_path)
        gs_best_score_file_list = [file for file in gs_all_file_list if file.endswith(".json")]

        cs_time_diff_dict = {}
        for cs_file_name in cs_best_score_file_list:
            cs_path = ml_score_stat_cs_path + '/' + cs_file_name
            with open(cs_path, 'r') as f:
                temp_dict = json.load(f)
                temp_list = []
                for ml_name, score_dict in temp_dict[Constant_Parameters.TIME_DIFF].items():
                    f1_score = score_dict[Constant_Parameters.F1_SCORE]
                    temp_list.append([ml_name, f1_score])
                sorted_f1_score_list = sorted(temp_list, key=lambda x: x[1], reverse=True)
                modified_file_name = cs_file_name.replace('.json', '')
                cs_time_diff_dict[modified_file_name] = sorted_f1_score_list[0]

        gs_time_diff_dict = {}
        for gs_file_name in gs_best_score_file_list:
            gs_path = ml_score_stat_cs_path + '/' + gs_file_name
            with open(gs_path, 'r') as f:
                temp_dict = json.load(f)
                temp_list = []
                for ml_name, score_dict in temp_dict[Constant_Parameters.TIME_DIFF].items():
                    f1_score = score_dict[Constant_Parameters.F1_SCORE]
                    temp_list.append([ml_name, f1_score])
                sorted_f1_score_list = sorted(temp_list, key=lambda x: x[1], reverse=True)
                modified_file_name = gs_file_name.replace('.json', '')
                gs_time_diff_dict[modified_file_name] = sorted_f1_score_list[0]

        return cs_time_diff_dict, gs_time_diff_dict
