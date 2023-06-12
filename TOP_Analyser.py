import json
import os
import re
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import Constant_Parameters


class TOP_Analyser:
    __cs_top_score_dict: dict
    __gs_top_score_dict: dict

    __scenario_type: str

    def __init__(self, ml_scenario_path: str):
        temp_list = ml_scenario_path.split('/')
        self.__scenario_type = temp_list[2] + '_' + temp_list[3] + '_' + temp_list[4]

        root_dir_path = Constant_Parameters.ML_SCORE_DIR_PATH + '/' + Constant_Parameters.TOP_PATH
        cs_stat_file_path = root_dir_path + '/' + Constant_Parameters.CS + '/' + self.__scenario_type + '.json'
        gs_stat_file_path = root_dir_path + '/' + Constant_Parameters.GS + '/' + self.__scenario_type + '.json'

        with open(cs_stat_file_path, 'r') as f:
            self.__cs_top_score_dict = json.load(f)
        with open(gs_stat_file_path, 'r') as f:
            self.__gs_top_score_dict = json.load(f)

    @classmethod
    def __saving_f1_score_to_heatmap(cls, f1_score_dict, default_save_path):
        df = pd.DataFrame(f1_score_dict)
        sns.heatmap(df, cmap='YlGnBu', annot=True, fmt='1.3f', linewidths=.3)
        sns.set(font_scale=0.7)
        plt.xticks(rotation=-45)
        plt.gcf().set_size_inches(16, 9)
        plt.title('F1 Score')
        save_path = default_save_path + '_f1_score.png'
        plt.savefig(save_path)
        plt.clf()

    @classmethod
    def __get_heatmap_symbol_name_list(cls, param_symbol_name: str, param_symbol_dict):
        symbol_name_list = list(symbol_name for symbol_name in param_symbol_dict.keys())
        last_symbol_name_combination: str = symbol_name_list[len(symbol_name_list) - 1]
        temp_list = \
            last_symbol_name_combination.replace('[', '').replace(']', '').replace(' ', '').split(',')
        symbol_list = []
        for symbol_name in temp_list:
            temp = symbol_name[1:-1]
            symbol_list.append(temp)

        replaced_symbol = param_symbol_name
        symbol_dict = {}
        for index, symbol_name in enumerate(symbol_list):
            symbol_dict[str(index)] = symbol_name
            replaced_symbol = replaced_symbol.replace(symbol_name, str(index))

        indexed_symbol_list = re.findall(r'\d', replaced_symbol)
        symbol_string = ''
        for symbol_name in indexed_symbol_list:
            symbol_string += symbol_name
            symbol_string += ' + '
        symbol_string = symbol_string[:-3]

        return symbol_string, symbol_dict

    @classmethod
    def __saving_symbol_index(cls, symbol_comb_index_dict, default_save_path):
        save_path = default_save_path + '_symbol_index.json'
        with open(save_path, 'w') as f:
            json.dump(symbol_comb_index_dict, f)

    @classmethod
    def __saving_combined_loss_rate_to_heatmap(cls, symbol_CLR_dict, default_save_path):
        df = pd.DataFrame(symbol_CLR_dict)
        sns.heatmap(df, cmap='YlGnBu', annot=True, fmt='1.3f', linewidths=.3)
        sns.set(font_scale=0.7)
        plt.xticks(rotation=-45)
        plt.gcf().set_size_inches(16, 9)
        plt.title('Combined Loss Rate')
        save_path = default_save_path + '_CLR.png'
        plt.savefig(save_path)
        plt.clf()

    @classmethod
    def __get_combined_best_score_dict(cls, symbol_f1_score_dict, symbol_CLR_dict):
        temp_list = []
        for symbol_index, score_dict in symbol_f1_score_dict.items():
            clr_dict = symbol_CLR_dict[symbol_index]
            for ml_type, f1_score in score_dict.items():
                clr = clr_dict[ml_type]
                if clr > 0 and f1_score > 0:
                    temp_list.append([symbol_index, ml_type, clr, f1_score])

        if len(temp_list) > 0:
            sorted_temp_list = sorted(temp_list, key=lambda x: (x[2], -x[3]))
            best_list = sorted_temp_list[0]

            symbol_index = best_list[0]
            ml_type = best_list[1]
            clr = best_list[2]
            f1_score = best_list[3]
        else:
            symbol_index = None
            ml_type = None
            clr = Constant_Parameters.DUMMY_DATA
            f1_score = Constant_Parameters.DUMMY_DATA

        return symbol_index, ml_type, clr, f1_score

    def __get_best_score_dict(self, scenario_type, station_type, top_score_dict):
        score_list = []
        for type_name, comb_dict in top_score_dict.items():
            for comb_name, symbol_dict in comb_dict.items():
                param_symbol_f1_score_dict = {}
                param_symbol_CLR_dict = {}

                default_save_path = \
                    Constant_Parameters.RESULT_TOP_DIR_PATH + '/' + station_type + '/' + comb_name + '/' + \
                    Constant_Parameters.F1_SCORE_PATH + '/' + scenario_type

                symbol_index_dict = None
                for symbol_name, ml_dict in symbol_dict.items():
                    f1_score_dict = {}
                    CLR_dict = {}
                    for ml_name, score_dict in ml_dict.items():
                        combined_loss_rate = score_dict[Constant_Parameters.COMBINATION_LOSS_RATE]
                        f1_score = score_dict[Constant_Parameters.F1_SCORE]
                        f1_score_dict[ml_name] = f1_score
                        CLR_dict[ml_name] = combined_loss_rate

                    symbol_string, param_symbol_dict = self.__get_heatmap_symbol_name_list(symbol_name, symbol_dict)
                    symbol_index_dict = param_symbol_dict
                    param_symbol_f1_score_dict[symbol_string] = f1_score_dict
                    param_symbol_CLR_dict[symbol_string] = CLR_dict

                if len(symbol_dict) > 1:
                    self.__saving_f1_score_to_heatmap(param_symbol_f1_score_dict, default_save_path)
                    self.__saving_combined_loss_rate_to_heatmap(param_symbol_CLR_dict, default_save_path)
                    self.__saving_symbol_index(symbol_index_dict, default_save_path)

                    symbol_index, ml_type, clr, f1_score = \
                        self.__get_combined_best_score_dict(param_symbol_f1_score_dict, param_symbol_CLR_dict)
                    if clr > 0 and f1_score > 0:
                        score_list.append([type_name, comb_name, symbol_index, ml_type, clr, f1_score])

        if len(score_list) > 0:
            sorted_score_list = sorted(score_list, key=lambda x: (x[4], -x[5]))
            best_list = sorted_score_list[0]

            param_dict = {Constant_Parameters.TYPE: best_list[0], Constant_Parameters.COMBINATION_TYPE: best_list[1],
                          Constant_Parameters.FEATURE_COMBINATION: best_list[2], Constant_Parameters.ML_TYPE: best_list[3],
                          Constant_Parameters.COMBINATION_LOSS_RATE: best_list[4],
                          Constant_Parameters.F1_SCORE: best_list[5]}
        else:
            param_dict = {Constant_Parameters.TYPE: None, Constant_Parameters.COMBINATION_TYPE: None,
                          Constant_Parameters.FEATURE_COMBINATION: None, Constant_Parameters.ML_TYPE: None,
                          Constant_Parameters.COMBINATION_LOSS_RATE: Constant_Parameters.DUMMY_DATA,
                          Constant_Parameters.F1_SCORE: Constant_Parameters.DUMMY_DATA}

        return param_dict

    def run(self, scenario_type):
        cs_best_score_dict = self.__get_best_score_dict(scenario_type, Constant_Parameters.CS, self.__cs_top_score_dict)
        gs_best_score_dict = self.__get_best_score_dict(scenario_type, Constant_Parameters.GS, self.__gs_top_score_dict)

        cs_save_path = Constant_Parameters.RESULT_TOP_DIR_PATH + '/' + Constant_Parameters.CS + '/' + \
                       scenario_type + '_' + Constant_Parameters.BEST_SCORE_FILE_NAME
        gs_save_path = Constant_Parameters.RESULT_TOP_DIR_PATH + '/' + Constant_Parameters.GS + '/' + \
                       scenario_type + '_' + Constant_Parameters.BEST_SCORE_FILE_NAME

        with open(cs_save_path, 'w') as f:
            json.dump(cs_best_score_dict, f)
        with open(gs_save_path, 'w') as f:
            json.dump(gs_best_score_dict, f)

    @classmethod
    def get_final_best_score(cls):
        cs_root_path = Constant_Parameters.RESULT_TOP_DIR_PATH + '/' + Constant_Parameters.CS
        gs_root_path = Constant_Parameters.RESULT_TOP_DIR_PATH + '/' + Constant_Parameters.GS

        cs_all_file_list = os.listdir(cs_root_path)
        cs_best_score_file_list = [file for file in cs_all_file_list if file.endswith(".json")]

        gs_all_file_list = os.listdir(gs_root_path)
        gs_best_score_file_list = [file for file in gs_all_file_list if file.endswith(".json")]

        cs_score_list = []
        for cs_best_score_file in cs_best_score_file_list:
            cs_file_path = cs_root_path + '/' + cs_best_score_file
            with open(cs_file_path, 'r') as f:
                cs_score_dict = json.load(f)

            cs_type = cs_score_dict[Constant_Parameters.TYPE]
            cs_comb_type = cs_score_dict[Constant_Parameters.COMBINATION_TYPE]
            cs_feature_comb = cs_score_dict[Constant_Parameters.FEATURE_COMBINATION]
            cs_ml_type = cs_score_dict[Constant_Parameters.ML_TYPE]
            cs_clr = cs_score_dict[Constant_Parameters.COMBINATION_LOSS_RATE]
            cs_f1_score = cs_score_dict[Constant_Parameters.F1_SCORE]

            if cs_clr > 0 and cs_f1_score > 0:
                param_list = [cs_type, cs_comb_type, cs_feature_comb, cs_ml_type, cs_clr, cs_f1_score]
                cs_score_list.append(param_list)

        print('-------------------- CS Best Score --------------------')
        if len(cs_score_list) > 0:
            sorted_cs_score_list = sorted(cs_score_list, key=lambda x: (x[4], -x[5]))
            best_cs_score_list = sorted_cs_score_list[0]
            best_cs_score_dict = {Constant_Parameters.TYPE: best_cs_score_list[0],
                                  Constant_Parameters.COMBINATION_TYPE: best_cs_score_list[1],
                                  Constant_Parameters.FEATURE_COMBINATION: best_cs_score_list[2],
                                  Constant_Parameters.ML_TYPE: best_cs_score_list[3],
                                  Constant_Parameters.COMBINATION_LOSS_RATE: best_cs_score_list[4],
                                  Constant_Parameters.F1_SCORE: best_cs_score_list[5]}

            print(best_cs_score_dict)
        else:
            print('No CS Best Score')

        gs_score_list = []
        for gs_best_score_file in gs_best_score_file_list:
            gs_file_path = gs_root_path + '/' + gs_best_score_file
            with open(gs_file_path, 'r') as f:
                gs_score_dict = json.load(f)

            gs_type = gs_score_dict[Constant_Parameters.TYPE]
            gs_comb_type = gs_score_dict[Constant_Parameters.COMBINATION_TYPE]
            gs_feature_comb = gs_score_dict[Constant_Parameters.FEATURE_COMBINATION]
            gs_ml_type = gs_score_dict[Constant_Parameters.ML_TYPE]
            gs_clr = gs_score_dict[Constant_Parameters.COMBINATION_LOSS_RATE]
            gs_f1_score = gs_score_dict[Constant_Parameters.F1_SCORE]

            if gs_clr > 0 and gs_f1_score > 0:
                param_list = [gs_type, gs_comb_type, gs_feature_comb, gs_ml_type, gs_clr, gs_f1_score]
                gs_score_list.append(param_list)

        print('-------------------- GS Best Score --------------------')
        if len(gs_score_list) > 0:
            sorted_gs_score_list = sorted(gs_score_list, key=lambda x: (x[4], -x[5]))
            best_gs_score_list = sorted_gs_score_list[0]
            best_gs_score_dict = {Constant_Parameters.TYPE: best_gs_score_list[0],
                                  Constant_Parameters.COMBINATION_TYPE: best_gs_score_list[1],
                                  Constant_Parameters.FEATURE_COMBINATION: best_gs_score_list[2],
                                  Constant_Parameters.ML_TYPE: best_gs_score_list[3],
                                  Constant_Parameters.COMBINATION_LOSS_RATE: best_gs_score_list[4],
                                  Constant_Parameters.F1_SCORE: best_gs_score_list[5]}

            print(best_gs_score_dict)
        else:
            print('No Best GS Score')