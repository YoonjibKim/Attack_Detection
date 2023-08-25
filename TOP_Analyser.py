import json
import re
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import stats

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
        save_path = default_save_path + '_' + Constant_Parameters.SYMBOL_INDEX + '.json'
        with open(save_path, 'w') as f:
            json.dump(symbol_comb_index_dict, f)

    @classmethod
    def __saving_support(cls, support_dict, default_save_path):
        save_path = default_save_path + '_' + Constant_Parameters.SUPPORT + '.json'
        with open(save_path, 'w') as f:
            json.dump(support_dict, f)

    @classmethod
    def __saving_combined_sampling_resolution_to_heatmap(cls, symbol_CSR_dict, default_save_path):
        df = pd.DataFrame(symbol_CSR_dict)
        sns.heatmap(df, cmap='YlGnBu', annot=True, fmt='1.3f', linewidths=.3)
        sns.set(font_scale=0.7)
        plt.xticks(rotation=-45)
        plt.gcf().set_size_inches(16, 9)
        plt.title('Combined Sampling Resolution')
        save_path = default_save_path + '_' + Constant_Parameters.CSR + '.png'
        plt.savefig(save_path)
        plt.clf()

    @classmethod
    def __get_combined_best_score_dict(cls, symbol_f1_score_dict, symbol_CSR_dict, symbol_support_dict):
        temp_list = []
        for symbol_index, score_dict in symbol_f1_score_dict.items():
            csr_dict = symbol_CSR_dict[symbol_index]
            support_dict = symbol_support_dict[symbol_index]
            for ml_type, f1_score in score_dict.items():
                csr = csr_dict[ml_type]
                support = support_dict[ml_type]
                if csr > 0 and f1_score > 0:
                    temp_list.append([symbol_index, ml_type, csr, f1_score, support])

        if len(temp_list) > 0:
            sorted_temp_list = sorted(temp_list, key=lambda x: (-x[4], -x[2], -x[3]))  # support, csr, f1
            best_list = sorted_temp_list[0]

            symbol_index = best_list[0]
            ml_type = best_list[1]
            csr = best_list[2]
            f1_score = best_list[3]
            support = best_list[4]
        else:
            symbol_index = None
            ml_type = None
            csr = Constant_Parameters.DUMMY_DATA
            f1_score = Constant_Parameters.DUMMY_DATA
            support = Constant_Parameters.DUMMY_DATA

        return symbol_index, ml_type, csr, f1_score, support

    @classmethod
    def __saving_f1_score_result_to_json(cls, f1_score_dict, default_save_path):
        save_path = default_save_path + '_' + Constant_Parameters.F1_SCORE_PATH + '.json'
        with open(save_path, 'w') as f:
            json.dump(f1_score_dict, f)

    @classmethod
    def __saving__combined_sampling_resolution_to_json(cls, csr_dict, default_save_path):
        save_path = default_save_path + '_' + Constant_Parameters.CSR + '.json'
        with open(save_path, 'w') as f:
            json.dump(csr_dict, f)

    def __get_best_score_dict(self, scenario_type, station_type, top_score_dict):
        score_list = []
        for type_name, comb_dict in top_score_dict.items():
            for comb_name, symbol_dict in comb_dict.items():
                param_symbol_f1_score_dict = {}
                param_symbol_CSR_dict = {}
                param_symbol_support_dict = {}

                default_save_path = \
                    Constant_Parameters.RESULT_TOP_DIR_PATH + '/' + station_type + '/' + comb_name + '/' + \
                    Constant_Parameters.F1_SCORE_PATH + '/' + scenario_type + '_' + type_name

                symbol_index_dict = None
                for symbol_name, ml_dict in symbol_dict.items():
                    f1_score_dict = {}
                    support_dict = {}
                    CSR_dict = {}
                    for ml_name, score_dict in ml_dict.items():
                        csr = score_dict[Constant_Parameters.COMBINED_SAMPLING_RESOLUTION]
                        f1_score = score_dict[Constant_Parameters.F1_SCORE]
                        f1_score_dict[ml_name] = f1_score
                        support = score_dict[Constant_Parameters.SUPPORT]
                        support_dict[ml_name] = support
                        CSR_dict[ml_name] = csr

                    symbol_string, param_symbol_dict = self.__get_heatmap_symbol_name_list(symbol_name, symbol_dict)
                    symbol_index_dict = param_symbol_dict
                    param_symbol_f1_score_dict[symbol_string] = f1_score_dict
                    param_symbol_CSR_dict[symbol_string] = CSR_dict
                    param_symbol_support_dict[symbol_string] = support_dict

                if len(symbol_dict) > 1:
                    self.__saving_f1_score_to_heatmap(param_symbol_f1_score_dict, default_save_path)
                    self.__saving_f1_score_result_to_json(param_symbol_f1_score_dict, default_save_path)
                    self.__saving_combined_sampling_resolution_to_heatmap(param_symbol_CSR_dict, default_save_path)
                    self.__saving__combined_sampling_resolution_to_json(param_symbol_CSR_dict, default_save_path)
                    self.__saving_symbol_index(symbol_index_dict, default_save_path)
                    self.__saving_support(param_symbol_support_dict, default_save_path)

                    symbol_index, ml_type, clr, f1_score, support = \
                        self.__get_combined_best_score_dict(param_symbol_f1_score_dict, param_symbol_CSR_dict,
                                                            param_symbol_support_dict)
                    if clr > 0 and f1_score > 0:
                        score_list.append([type_name, comb_name, symbol_index, ml_type, clr, f1_score, support])

        if len(score_list) > 0:
            sorted_score_list = sorted(score_list, key=lambda x: (-x[6], -x[4], -x[5]))  # support, csr, f1
            best_list = sorted_score_list[0]

            param_dict = {Constant_Parameters.TYPE: best_list[0], Constant_Parameters.COMBINATION_TYPE: best_list[1],
                          Constant_Parameters.FEATURE_COMBINATION: best_list[2],
                          Constant_Parameters.ML_TYPE: best_list[3],
                          Constant_Parameters.COMBINED_SAMPLING_RESOLUTION: best_list[4],
                          Constant_Parameters.F1_SCORE: best_list[5],
                          Constant_Parameters.SUPPORT: best_list[6]}
        else:
            param_dict = {Constant_Parameters.TYPE: None, Constant_Parameters.COMBINATION_TYPE: None,
                          Constant_Parameters.FEATURE_COMBINATION: None, Constant_Parameters.ML_TYPE: None,
                          Constant_Parameters.COMBINED_SAMPLING_RESOLUTION: Constant_Parameters.DUMMY_DATA,
                          Constant_Parameters.F1_SCORE: Constant_Parameters.DUMMY_DATA,
                          Constant_Parameters.SUPPORT: Constant_Parameters.DUMMY_DATA}

        return param_dict

    def run(self, scenario_type):
        print(scenario_type)

        cs_best_score_dict = self.__get_best_score_dict(scenario_type, Constant_Parameters.CS, self.__cs_top_score_dict)
        gs_best_score_dict = self.__get_best_score_dict(scenario_type, Constant_Parameters.GS, self.__gs_top_score_dict)

        cs_save_path = Constant_Parameters.RESULT_TOP_DIR_PATH + '/' + Constant_Parameters.CS + '/' + \
                       scenario_type + '_' + Constant_Parameters.BEST_SCORE_FILE_NAME + '.json'
        gs_save_path = Constant_Parameters.RESULT_TOP_DIR_PATH + '/' + Constant_Parameters.GS + '/' + \
                       scenario_type + '_' + Constant_Parameters.BEST_SCORE_FILE_NAME + '.json'

        with open(cs_save_path, 'w') as f:
            json.dump(cs_best_score_dict, f)
        with open(gs_save_path, 'w') as f:
            json.dump(gs_best_score_dict, f)

    @classmethod
    def __get_cs_common_dataset_dict(cls):
        total_cs_common_dataset_dict = {}
        total_cs_common_csr_dict = {}
        total_cs_original_dataset_dict = {}

        for scenario, path in Constant_Parameters.PROCESSED_DATASET_PATH_DICT.items():
            cs_dir_path = path + '/' + Constant_Parameters.TOP + '/' + Constant_Parameters.CS
            cs_processed_path = cs_dir_path + '/' + Constant_Parameters.FINAL_DATASET + '.json'
            cs_original_path = cs_dir_path + '/' + Constant_Parameters.TOP + '.json'

            with open(cs_processed_path, 'r') as f:
                cs_processed_dict = json.load(f)
            with open(cs_original_path, 'r') as f:
                cs_original_dict = json.load(f)

            cs_common_dataset_dict = {}
            cs_common_csr_dict = {}
            cs_original_dataset_dict = {}

            zero_flag = False
            for category_name, type_dict in cs_processed_dict.items():
                symbol_list_dict = type_dict[Constant_Parameters.COMMON]
                if len(symbol_list_dict) < 1:
                    zero_flag = True
                    break

                common_1_dict = {}
                common_2_dict = {}
                original_1_dict = {}
                for symbol_comb, temp_dict in symbol_list_dict.items():
                    if ',' not in symbol_comb:
                        attack_dict = temp_dict[Constant_Parameters.ATTACK]
                        normal_dict = temp_dict[Constant_Parameters.NORMAL]
                        attack_csr = attack_dict[Constant_Parameters.COMBINED_SAMPLING_RESOLUTION]
                        normal_csr = normal_dict[Constant_Parameters.COMBINED_SAMPLING_RESOLUTION]
                        symbol_name = list(attack_dict[Constant_Parameters.DATA_POINT].keys())[0]
                        attack_processed_cs_dict = attack_dict[Constant_Parameters.DATA_POINT][symbol_name]
                        normal_processed_cs_dict = attack_dict[Constant_Parameters.DATA_POINT][symbol_name]

                        attack_original_cs_dict = {}
                        for attack_cs in attack_processed_cs_dict.keys():
                            attack_category_dict = \
                                cs_original_dict[attack_cs][category_name][Constant_Parameters.ATTACK][symbol_name]
                            attack_original_cs_dict[attack_cs] = attack_category_dict[Constant_Parameters.DATA_POINT]

                        normal_original_cs_dict = {}
                        for normal_cs in attack_processed_cs_dict.keys():
                            normal_category_dict = \
                                cs_original_dict[normal_cs][category_name][Constant_Parameters.NORMAL][symbol_name]
                            normal_original_cs_dict[normal_cs] = normal_category_dict[Constant_Parameters.DATA_POINT]

                        common_1_dict[symbol_name] = {Constant_Parameters.ATTACK: attack_processed_cs_dict,
                                                      Constant_Parameters.NORMAL: normal_processed_cs_dict}
                        common_2_dict[symbol_name] = {Constant_Parameters.ATTACK: attack_csr,
                                                      Constant_Parameters.NORMAL: normal_csr}
                        original_1_dict[symbol_name] = {Constant_Parameters.ATTACK: attack_original_cs_dict,
                                                        Constant_Parameters.NORMAL: normal_original_cs_dict}
                if zero_flag:
                    break

                temp_dict = list(symbol_list_dict.values())[len(symbol_list_dict) - 1]
                attack_dict = temp_dict[Constant_Parameters.ATTACK]
                normal_dict = temp_dict[Constant_Parameters.NORMAL]
                attack_csr = attack_dict[Constant_Parameters.COMBINED_SAMPLING_RESOLUTION]
                normal_csr = normal_dict[Constant_Parameters.COMBINED_SAMPLING_RESOLUTION]
                attack_processed_cs_dict = attack_dict[Constant_Parameters.DATA_POINT]
                normal_processed_cs_dict = normal_dict[Constant_Parameters.DATA_POINT]
                param_3_dict = {Constant_Parameters.ATTACK: attack_processed_cs_dict,
                                Constant_Parameters.NORMAL: normal_processed_cs_dict}
                param_4_dict = {Constant_Parameters.ATTACK: attack_csr, Constant_Parameters.NORMAL: normal_csr}
                cs_common_dataset_dict[category_name] = {Constant_Parameters.BASIS_SYMBOL: common_1_dict,
                                                         Constant_Parameters.COMBINED_SYMBOL: param_3_dict}
                cs_common_csr_dict[category_name] = {Constant_Parameters.BASIS_SYMBOL: common_2_dict,
                                                     Constant_Parameters.COMBINED_SYMBOL: param_4_dict}
                cs_original_dataset_dict[category_name] = {Constant_Parameters.BASIS_SYMBOL: original_1_dict}

            if zero_flag is False:
                total_cs_common_dataset_dict[scenario] = cs_common_dataset_dict
                total_cs_common_csr_dict[scenario] = cs_common_csr_dict
                total_cs_original_dataset_dict[scenario] = cs_original_dataset_dict

        return total_cs_common_dataset_dict, total_cs_common_csr_dict, total_cs_original_dataset_dict

    @classmethod
    def __calculate_histogram_dict(cls, processed_dict, original_dict):
        n_size = 100
        diff_rate_list = []
        for cs_id, processed_data_point_list in processed_dict.items():
            original_data_point_list = original_dict[cs_id]
            bin_list = list(_ for _ in range(0, n_size))
            original_bin_dict = {}
            processed_bin_dict = {}
            for index in bin_list:
                original_bin_dict[str(index)] = []
                processed_bin_dict[str(index)] = []

            for overhead in original_data_point_list:
                for bin_begin_val in bin_list:
                    bin_end_val = bin_begin_val + 1
                    if bin_begin_val <= overhead < bin_end_val:
                        original_bin_dict[str(bin_begin_val)].append(overhead)

            for overhead in processed_data_point_list:
                for bin_begin_val in bin_list:
                    bin_end_val = bin_begin_val + 1
                    if bin_begin_val <= overhead < bin_end_val:
                        processed_bin_dict[str(bin_begin_val)].append(overhead)

            difference_list = []
            for overhead in original_bin_dict.keys():
                original_overhead_list = original_bin_dict[overhead]
                processed_overhead_list = processed_bin_dict[overhead]
                original_size = len(original_overhead_list)
                processed_size = len(processed_overhead_list)
                difference = 1 - np.exp(original_size - processed_size) / np.exp(original_size)
                difference_list.append(difference)
            diff_rate = sum(difference_list) / n_size
            diff_rate_list.append(diff_rate)

        diff_mean = np.mean(diff_rate_list)

        return diff_mean

    @classmethod
    def __get_ordered_list(cls, csr_list):
        csr_copy_list = deepcopy(csr_list)
        sorted_csr_list = sorted(csr_copy_list)

        order_list = []
        for sorted_csr in sorted_csr_list:
            for index, csr in enumerate(csr_list):
                if csr == sorted_csr:
                    order_list.append(index + 1)
                    break

        return order_list

    @classmethod
    def __calculate_spearman_correlation_analysis(cls, total_result_dict):
        original_attack_cycle_list = []
        original_attack_branch_list = []
        original_attack_instruction_list = []
        processed_attack_cycle_list = []
        processed_attack_branch_list = []
        processed_attack_instruction_list = []

        original_normal_cycle_list = []
        original_normal_branch_list = []
        original_normal_instruction_list = []
        processed_normal_cycle_list = []
        processed_normal_branch_list = []
        processed_normal_instruction_list = []

        for scenario, category_dict in total_result_dict.items():
            for category, temp_dict in category_dict.items():
                attack_dict = temp_dict[Constant_Parameters.COMBINED_SYMBOL][Constant_Parameters.ATTACK]
                normal_dict = temp_dict[Constant_Parameters.COMBINED_SYMBOL][Constant_Parameters.NORMAL]

                original_attack_csr = attack_dict[Constant_Parameters.CSR_PROOF]
                original_normal_csr = normal_dict[Constant_Parameters.CSR_PROOF]
                processed_attack_csr = attack_dict[Constant_Parameters.COMBINED_SAMPLING_RESOLUTION]
                processed_normal_csr = normal_dict[Constant_Parameters.COMBINED_SAMPLING_RESOLUTION]

                if category == Constant_Parameters.INSTRUCTIONS:
                    original_attack_instruction_list.append(original_attack_csr)
                    original_normal_instruction_list.append(original_normal_csr)
                    processed_attack_instruction_list.append(processed_attack_csr)
                    processed_normal_instruction_list.append(processed_normal_csr)
                elif category == Constant_Parameters.BRANCH:
                    original_attack_branch_list.append(original_attack_csr)
                    original_normal_branch_list.append(original_normal_csr)
                    processed_attack_branch_list.append(processed_attack_csr)
                    processed_normal_branch_list.append(processed_normal_csr)
                elif category == Constant_Parameters.CYCLES:
                    original_attack_cycle_list.append(original_attack_csr)
                    original_normal_cycle_list.append(original_normal_csr)
                    processed_attack_cycle_list.append(processed_attack_csr)
                    processed_normal_cycle_list.append(processed_normal_csr)

        attack_cycle_stat_val, attack_cycle_p_val = \
            stats.spearmanr(original_attack_cycle_list, processed_attack_cycle_list)
        attack_branch_stat_val, attack_branch_p_val = \
            stats.spearmanr(original_attack_branch_list, processed_attack_branch_list)
        attack_instruction_stat_val, attack_instruction_p_val = \
            stats.spearmanr(original_attack_instruction_list, processed_attack_instruction_list)
        normal_cycle_stat_val, normal_cycle_p_val = \
            stats.spearmanr(original_normal_cycle_list, processed_normal_cycle_list)
        normal_branch_stat_val, normal_branch_p_val = \
            stats.spearmanr(original_normal_branch_list, processed_normal_branch_list)
        normal_instruction_stat_val, normal_instruction_p_val = \
            stats.spearmanr(original_normal_instruction_list, processed_normal_instruction_list)

        attack_cycle_dict = {Constant_Parameters.STATISTICS: attack_cycle_stat_val,
                             Constant_Parameters.P_VALUE: attack_cycle_p_val}
        attack_branch_dict = {Constant_Parameters.STATISTICS: attack_branch_stat_val,
                              Constant_Parameters.P_VALUE: attack_branch_p_val}
        attack_instruction_dict = {Constant_Parameters.STATISTICS: attack_instruction_stat_val,
                                   Constant_Parameters.P_VALUE: attack_instruction_p_val}
        normal_cycle_dict = {Constant_Parameters.STATISTICS: normal_cycle_stat_val,
                             Constant_Parameters.P_VALUE: normal_cycle_p_val}
        normal_branch_dict = {Constant_Parameters.STATISTICS: normal_branch_stat_val,
                              Constant_Parameters.P_VALUE: normal_branch_p_val}
        normal_instruction_dict = {Constant_Parameters.STATISTICS: normal_instruction_stat_val,
                                   Constant_Parameters.P_VALUE: normal_instruction_p_val}

        cycle_dict = {Constant_Parameters.ATTACK: attack_cycle_dict, Constant_Parameters.NORMAL: normal_cycle_dict}
        branch_dict = {Constant_Parameters.ATTACK: attack_branch_dict, Constant_Parameters.NORMAL: normal_branch_dict}
        instruction_dict = \
            {Constant_Parameters.ATTACK: attack_instruction_dict, Constant_Parameters.NORMAL: normal_instruction_dict}

        correlation_analysis_dict = {Constant_Parameters.CYCLES: cycle_dict, Constant_Parameters.BRANCH: branch_dict,
                                     Constant_Parameters.INSTRUCTIONS: instruction_dict}

        return correlation_analysis_dict

    @classmethod
    def proof_csr(cls):
        total_cs_common_dataset_dict, total_cs_common_csr_dict, total_cs_original_dataset_dict = \
            cls.__get_cs_common_dataset_dict()

        total_result_dict = {}
        for scenario, category_dict in total_cs_common_dataset_dict.items():
            scenario_dict = {}
            for category, temp_dict in category_dict.items():
                basis_category_dict = {}

                basis_symbol_data_dict = temp_dict[Constant_Parameters.BASIS_SYMBOL]
                for basis_symbol_name, type_dict in basis_symbol_data_dict.items():
                    attack_processed_basis_symbol_dict = type_dict[Constant_Parameters.ATTACK]
                    normal_processed_basis_symbol_dict = type_dict[Constant_Parameters.NORMAL]
                    original_basis_symbol = \
                        total_cs_original_dataset_dict[scenario][category][Constant_Parameters.BASIS_SYMBOL]
                    attack_original_basis_symbol_dict = \
                        original_basis_symbol[basis_symbol_name][Constant_Parameters.ATTACK]
                    normal_original_basis_symbol_dict = \
                        original_basis_symbol[basis_symbol_name][Constant_Parameters.NORMAL]
                    temp_csr_dict = total_cs_common_csr_dict[scenario][category][Constant_Parameters.BASIS_SYMBOL]

                    attack_basis_symbol_csr = temp_csr_dict[basis_symbol_name][Constant_Parameters.ATTACK]
                    normal_basis_symbol_csr = temp_csr_dict[basis_symbol_name][Constant_Parameters.NORMAL]
                    attack_diff_mean = cls.__calculate_histogram_dict(attack_processed_basis_symbol_dict,
                                                                      attack_original_basis_symbol_dict)
                    normal_diff_mean = cls.__calculate_histogram_dict(normal_processed_basis_symbol_dict,
                                                                      normal_original_basis_symbol_dict)
                    basis_attack_dict = {Constant_Parameters.COMBINED_SAMPLING_RESOLUTION: attack_basis_symbol_csr,
                                         Constant_Parameters.CSR_PROOF: attack_diff_mean}
                    basis_normal_dict = {Constant_Parameters.COMBINED_SAMPLING_RESOLUTION: normal_basis_symbol_csr,
                                         Constant_Parameters.CSR_PROOF: normal_diff_mean}
                    basis_category_dict[basis_symbol_name] = {Constant_Parameters.ATTACK: basis_attack_dict,
                                                              Constant_Parameters.NORMAL: basis_normal_dict}

                processed_combined_symbol_dict = temp_dict[Constant_Parameters.COMBINED_SYMBOL]
                temp_attack_processed_combined_symbol_dict = processed_combined_symbol_dict[Constant_Parameters.ATTACK]
                temp_normal_processed_combined_symbol_dict = processed_combined_symbol_dict[Constant_Parameters.NORMAL]

                attack_result_list = []
                for attack_combined_symbol_name, attack_processed_combined_symbol_dict \
                        in temp_attack_processed_combined_symbol_dict.items():
                    original_basis_symbol = \
                        total_cs_original_dataset_dict[scenario][category][Constant_Parameters.BASIS_SYMBOL]
                    attack_original_combined_symbol_dict = \
                        original_basis_symbol[attack_combined_symbol_name][Constant_Parameters.ATTACK]
                    attack_diff_mean = cls.__calculate_histogram_dict(attack_processed_combined_symbol_dict,
                                                                      attack_original_combined_symbol_dict)
                    attack_result_list.append(attack_diff_mean)

                normal_result_list = []
                for normal_combined_symbol_name, normal_processed_combined_symbol_dict \
                        in temp_normal_processed_combined_symbol_dict.items():
                    original_basis_symbol = \
                        total_cs_original_dataset_dict[scenario][category][Constant_Parameters.BASIS_SYMBOL]
                    normal_original_combined_symbol_dict = \
                        original_basis_symbol[normal_combined_symbol_name][Constant_Parameters.ATTACK]
                    normal_diff_mean = cls.__calculate_histogram_dict(normal_processed_combined_symbol_dict,
                                                                      normal_original_combined_symbol_dict)
                    normal_result_list.append(normal_diff_mean)

                temp = total_cs_common_csr_dict[scenario][category][Constant_Parameters.COMBINED_SYMBOL]
                attack_combined_csr = temp[Constant_Parameters.ATTACK]
                normal_combined_csr = temp[Constant_Parameters.NORMAL]

                common_attack_dict = {Constant_Parameters.COMBINED_SAMPLING_RESOLUTION: attack_combined_csr,
                                      Constant_Parameters.CSR_PROOF: np.mean(attack_result_list)}
                common_normal_dict = {Constant_Parameters.COMBINED_SAMPLING_RESOLUTION: normal_combined_csr,
                                      Constant_Parameters.CSR_PROOF: np.mean(normal_result_list)}
                common_category_dict = {Constant_Parameters.ATTACK: common_attack_dict,
                                        Constant_Parameters.NORMAL: common_normal_dict}

                scenario_dict[category] = {Constant_Parameters.BASIS_SYMBOL: basis_category_dict,
                                           Constant_Parameters.COMBINED_SYMBOL: common_category_dict}

            total_result_dict[scenario] = scenario_dict

        file_path = Constant_Parameters.RESULT + '/' + Constant_Parameters.CSR_ANALYSIS + '/' + \
                    Constant_Parameters.CSR_ANALYSIS + '.json'
        with open(file_path, 'w') as f:
            json.dump(total_result_dict, f)

        correlation_analysis_dict = cls.__calculate_spearman_correlation_analysis(total_result_dict)

        file_path = Constant_Parameters.RESULT + '/' + Constant_Parameters.CSR_ANALYSIS + '/' + \
                    Constant_Parameters.CORRELATION_ANALYSIS_FILENAME
        with open(file_path, 'w') as f:
            json.dump(correlation_analysis_dict, f)
