import json
import math
import random
from itertools import combinations
import numpy as np
import Constant_Parameters
from Dataset_Initialization import Dataset_Initialization
from TOP_Record_Analysis import TOP_Record_Analysis
from Time_Diff_Parser import Time_Diff_Parser


class TOP_Parser(Time_Diff_Parser, TOP_Record_Analysis):
    __cs_id_list: list
    __gs_id: list

    __attack_sim_time: float
    __normal_sim_time: float

    __scenario_list: list
    __type_list: list

    __cs_top_dict: dict
    __gs_top_dict: dict
    __time_diff_dict: dict

    __gs_unique_intersection_symbol_dict: dict
    __cs_unique_intersection_symbol_dict: dict

    __feature_count = 5

    def __init__(self, scenario_object: Dataset_Initialization):
        cs_top_path_dict = scenario_object.get_cs_top_file_path_dict()
        gs_top_path_dict = scenario_object.get_gs_top_file_path_dict()
        gs_time_diff_dict = scenario_object.get_gs_time_diff_file_path_dict()
        self.__attack_sim_time = scenario_object.get_attack_sim_time()
        self.__normal_sim_time = scenario_object.get_normal_sim_time()

        self.__scenario_list = scenario_object.get_scenario_list()
        self.__type_list = scenario_object.get_type_list()

        cs_id_set = set(cs_top_path_dict.keys())
        self.__cs_id_list = list(cs_id_set)

        gs_id_set = set(gs_top_path_dict.keys()) & set(gs_time_diff_dict.keys())
        self.__gs_id = gs_id_set.pop()

        temp_time_diff_dataset_dict = self.__loading_time_diff_dataset(gs_time_diff_dict, self.__gs_id)
        super().__init__(temp_time_diff_dataset_dict, self.__cs_id_list, self.__attack_sim_time, self.__normal_sim_time)
        self.__time_diff_dict = self.analyzing_time_diff_sampling_analysis_dict()

        self.__cs_top_dict = self.__loading_top_dataset(cs_top_path_dict, self.__cs_id_list, self.__type_list,
                                                        self.__attack_sim_time, self.__normal_sim_time)
        self.__gs_top_dict = self.__loading_top_dataset(gs_top_path_dict, [self.__gs_id], self.__type_list,
                                                        self.__attack_sim_time, self.__normal_sim_time)

    def run(self):
        self.run_top_record_analysis(self.__cs_id_list, self.__scenario_list)

        # all symbol information
        cs_sam_res_diff_and_mean_dict = self.__get_sam_res_diff_and_mean_dict(self.__cs_top_dict,
                                                                              self.__cs_unique_intersection_symbol_dict)
        gs_sam_res_diff_and_mean_dict = self.__get_sam_res_diff_and_mean_dict(self.__gs_top_dict,
                                                                              self.__gs_unique_intersection_symbol_dict)

        # combination reference symbols <- here
        cs_basis_symbol_dict = self.__get_reference_symbol_dict(self.__cs_unique_intersection_symbol_dict)
        gs_basis_symbol_dict = self.__get_reference_symbol_dict(self.__gs_unique_intersection_symbol_dict)

        # full individual feature
        cs_full_feature_dict = \
            self.__get_full_individual_feature_dict(cs_basis_symbol_dict, cs_sam_res_diff_and_mean_dict,
                                                    self.__cs_top_dict)
        gs_full_feature_dict = \
            self.__get_full_individual_feature_dict(gs_basis_symbol_dict, gs_sam_res_diff_and_mean_dict,
                                                    self.__gs_top_dict)

        # feature combination
        cs_comb_dict = self.__get_chosen_combination_dict(cs_full_feature_dict, self.__feature_count)
        gs_comb_dict = self.__get_chosen_combination_dict(gs_full_feature_dict, self.__feature_count)

        # final dataset
        cs_feature_dict = self.__get_combined_feature_dict(cs_comb_dict, cs_full_feature_dict)
        gs_feature_dict = self.__get_combined_feature_dict(gs_comb_dict, gs_full_feature_dict)

        # ml dataset
        cs_ml_feature_dict = self.__converting_to_ml_feature_list(cs_feature_dict)
        gs_ml_feature_dict = self.__converting_to_ml_feature_list(gs_feature_dict)

        self.__saving_data(self.__cs_unique_intersection_symbol_dict, Constant_Parameters.UNIQUE_INTERSECTION,
                           Constant_Parameters.CS)
        self.__saving_data(self.__gs_unique_intersection_symbol_dict, Constant_Parameters.UNIQUE_INTERSECTION,
                           Constant_Parameters.GS)
        self.__saving_data(self.__cs_top_dict, Constant_Parameters.TOP, Constant_Parameters.CS)
        self.__saving_data(self.__gs_top_dict, Constant_Parameters.TOP, Constant_Parameters.GS)
        self.__saving_data(cs_sam_res_diff_and_mean_dict, Constant_Parameters.SAM_RES_DIFF_AND_MEAN,
                           Constant_Parameters.CS)
        self.__saving_data(gs_sam_res_diff_and_mean_dict, Constant_Parameters.SAM_RES_DIFF_AND_MEAN,
                           Constant_Parameters.GS)
        self.__saving_data(cs_basis_symbol_dict, Constant_Parameters.BASIS_SYMBOL, Constant_Parameters.CS)
        self.__saving_data(gs_basis_symbol_dict, Constant_Parameters.BASIS_SYMBOL, Constant_Parameters.GS)
        self.__saving_data(cs_full_feature_dict, Constant_Parameters.FULL_FEATURE_INFORMATION, Constant_Parameters.CS)
        self.__saving_data(gs_full_feature_dict, Constant_Parameters.FULL_FEATURE_INFORMATION, Constant_Parameters.GS)
        self.__saving_data(cs_comb_dict, Constant_Parameters.FEATURE_COMBINATION, Constant_Parameters.CS)
        self.__saving_data(gs_comb_dict, Constant_Parameters.FEATURE_COMBINATION, Constant_Parameters.GS)
        self.__saving_data(cs_feature_dict, Constant_Parameters.FINAL_DATASET, Constant_Parameters.CS)
        self.__saving_data(gs_feature_dict, Constant_Parameters.FINAL_DATASET, Constant_Parameters.GS)
        self.__saving_ml_feature(cs_ml_feature_dict, self.__scenario_list, Constant_Parameters.CS)
        self.__saving_ml_feature(gs_ml_feature_dict, self.__scenario_list, Constant_Parameters.GS)

    @classmethod
    def __get_target_symbol_dict(cls, final_dataset_dict):
        param_1_dict = {}
        for category_name, type_dict in final_dataset_dict.items():
            param_2_dict = {}
            for type_name, symbol_dict in type_dict.items():
                if len(symbol_dict) > 0:
                    temp_list = list(symbol_dict.keys())
                    temp_str = temp_list[len(temp_list) - 1]

                    temp_list = temp_str.split(',')
                    chunk_list = []
                    for temp_str in temp_list:
                        chunk_str = temp_str.replace("\'", '')
                        chunk_str = chunk_str.replace('[', '')
                        chunk_str = chunk_str.replace(']', '')
                        chunk_str = chunk_str.strip()
                        chunk_list.append(chunk_str)

                    param_2_dict[type_name] = chunk_list
                else:
                    param_2_dict[type_name] = None
            param_1_dict[category_name] = param_2_dict

        return param_1_dict

    @classmethod
    def __get_target_symbol_size_dict(cls, target_symbol_dict, final_dataset_dict):
        param_1_dict = {}
        for category_name, type_dict in target_symbol_dict.items():
            exclusive_symbol_list = type_dict[Constant_Parameters.EXCLUSIVE]
            common_symbol_list = type_dict[Constant_Parameters.COMMON]

            exclusive_dict = {}
            if exclusive_symbol_list is not None:
                for symbol_name in exclusive_symbol_list:
                    exclusive_symbol_dict = final_dataset_dict[category_name][Constant_Parameters.EXCLUSIVE]
                    temp_dict = exclusive_symbol_dict["['" + symbol_name + "']"]
                    attack_data_point_dict = temp_dict[Constant_Parameters.ATTACK][Constant_Parameters.DATA_POINT]
                    normal_data_point_dict = temp_dict[Constant_Parameters.NORMAL][Constant_Parameters.DATA_POINT]
                    attack_symbol_dict = attack_data_point_dict[symbol_name]
                    normal_symbol_dict = normal_data_point_dict[symbol_name]

                    attack_symbol_size = len(list(attack_symbol_dict.values())[0])
                    normal_symbol_size = len(list(normal_symbol_dict.values())[0])
                    exclusive_dict[symbol_name] = {Constant_Parameters.ATTACK: attack_symbol_size,
                                                   Constant_Parameters.NORMAL: normal_symbol_size}
            else:
                exclusive_dict = None

            common_dict = {}
            if common_symbol_list is not None:
                for symbol_name in common_symbol_list:
                    common_symbol_dict = final_dataset_dict[category_name][Constant_Parameters.COMMON]
                    temp_dict = common_symbol_dict["['" + symbol_name + "']"]
                    attack_data_point_dict = temp_dict[Constant_Parameters.ATTACK][Constant_Parameters.DATA_POINT]
                    normal_data_point_dict = temp_dict[Constant_Parameters.NORMAL][Constant_Parameters.DATA_POINT]
                    attack_symbol_dict = attack_data_point_dict[symbol_name]
                    normal_symbol_dict = normal_data_point_dict[symbol_name]

                    attack_symbol_size = len(list(attack_symbol_dict.values())[0])
                    normal_symbol_size = len(list(normal_symbol_dict.values())[0])
                    common_dict[symbol_name] = {Constant_Parameters.ATTACK: attack_symbol_size,
                                                Constant_Parameters.NORMAL: normal_symbol_size}
            else:
                common_dict = None

            param_1_dict[category_name] = {Constant_Parameters.EXCLUSIVE: exclusive_dict,
                                           Constant_Parameters.COMMON: common_dict}

        return param_1_dict

    @classmethod
    def __converting_to_ml_feature_list(cls, feature_dict):
        param_type_dict = {}
        for type_name, comb_dict in feature_dict.items():
            param_comb_dict = {}
            for comb_name, symbol_dict in comb_dict.items():
                param_symbol_dict = {}
                for symbol_comb_list, category_dict in symbol_dict.items():
                    attack_data_dict = category_dict[Constant_Parameters.ATTACK][Constant_Parameters.DATA_POINT]
                    normal_data_dict = category_dict[Constant_Parameters.NORMAL][Constant_Parameters.DATA_POINT]

                    attack_data_point_list = \
                        cls.__get_combined_data_point_list(attack_data_dict, Constant_Parameters.ATTACK_LABEL)
                    normal_data_point_list = \
                        cls.__get_combined_data_point_list(normal_data_dict, Constant_Parameters.NORMAL_LABEL)

                    attack_size = len(attack_data_point_list)
                    normal_size = len(normal_data_point_list)
                    training_attack_size = round(attack_size * Constant_Parameters.TRAINING_SET_RATIO)
                    training_normal_size = round(normal_size * Constant_Parameters.TRAINING_SET_RATIO)

                    random.shuffle(attack_data_point_list)
                    random.shuffle(normal_data_point_list)

                    training_attack_list = attack_data_point_list[:training_attack_size]
                    testing_attack_list = attack_data_point_list[training_attack_size:]
                    training_normal_list = normal_data_point_list[:training_normal_size]
                    testing_normal_list = normal_data_point_list[training_normal_size:]

                    temp_training_list = []
                    temp_training_list.extend(training_attack_list)
                    temp_training_list.extend(training_normal_list)
                    temp_testing_list = []
                    temp_testing_list.extend(testing_attack_list)
                    temp_testing_list.extend(testing_normal_list)

                    random.shuffle(temp_training_list)
                    random.shuffle(temp_testing_list)

                    training_feature_list = []
                    training_label_list = []
                    for record_list in temp_training_list:
                        training_feature_list.append(record_list[:-1])
                        training_label_list.append(record_list[len(record_list) - 1])

                    testing_feature_list = []
                    testing_label_list = []
                    for record_list in temp_testing_list:
                        testing_feature_list.append(record_list[:-1])
                        testing_label_list.append(record_list[len(record_list) - 1])

                    attack_sampling_resolution = \
                        category_dict[Constant_Parameters.ATTACK][Constant_Parameters.COMBINED_SAMPLING_RESOLUTION]
                    normal_sampling_resolution = \
                        category_dict[Constant_Parameters.NORMAL][Constant_Parameters.COMBINED_SAMPLING_RESOLUTION]
                    average_CSR = (attack_sampling_resolution + normal_sampling_resolution) / 2

                    param_dict = {Constant_Parameters.COMBINED_SAMPLING_RESOLUTION: average_CSR,
                                  Constant_Parameters.TRAINING_FEATURE: training_feature_list,
                                  Constant_Parameters.TRAINING_LABEL: training_label_list,
                                  Constant_Parameters.TESTING_FEATURE: testing_feature_list,
                                  Constant_Parameters.TESTING_LABEL: testing_label_list}

                    param_symbol_dict[str(symbol_comb_list)] = param_dict
                param_comb_dict[comb_name] = param_symbol_dict
            param_type_dict[type_name] = param_comb_dict

        return param_type_dict

    @classmethod
    def __get_combined_data_point_list(cls, category_dict, label):
        data_point_size = 0
        total_data_point_list = []
        for data_point_dict in category_dict.values():
            for station_id, data_point_list in data_point_dict.items():
                data_point_size = len(data_point_list)
                break

            sub_data_point_list = []
            for index in range(0, data_point_size):
                record_list = []
                for data_point_list in data_point_dict.values():
                    record_list.append(data_point_list[index])
                sub_data_point_list.append(record_list)

            total_data_point_list.append(sub_data_point_list)

        combined_total_data_point_list = []
        for index in range(0, data_point_size):
            record_list = []
            for sub_data_point_list in total_data_point_list:
                temp_list = sub_data_point_list[index]
                record_list.extend(temp_list)
            record_list.append(label)
            combined_total_data_point_list.append(record_list)

        return combined_total_data_point_list

    @classmethod
    def __saving_ml_feature(cls, ml_feature_dict, scenario_list, station_type):
        fie_path = Constant_Parameters.ML_DATASET_PATH + '/' + Constant_Parameters.TOP + '/' + station_type
        file_name = ''
        for scenario in scenario_list:
            file_name += scenario
            file_name += '_'

        file_name = file_name[:-1]
        file_name += '.json'

        with open(fie_path + '/' + file_name, 'w') as f:
            json.dump(ml_feature_dict, f)

    @classmethod
    def __get_combined_feature_dict(cls, all_comb_dict, full_feature_dict):
        primary_symbol_name_dict = cls.__get_primary_symbol_name_dict(all_comb_dict)
        primary_symbol_dict = cls.__get_primary_symbol_dict(primary_symbol_name_dict, full_feature_dict)

        temp_type_dict = {}
        for type_name, comb_dict in all_comb_dict.items():
            temp_comb_dict = {}
            for comb_name, symbol_record_list in comb_dict.items():
                temp_sub_set_dict = {}
                for sub_set_list in symbol_record_list:
                    attack_min_size, normal_min_size, attack_min_CSR, normal_min_CSR = \
                        cls.__get_min_size_and_sampling_resolution_dict(sub_set_list, type_name, comb_name,
                                                                        primary_symbol_dict)
                    temp_attack_CSR_diff = 0
                    temp_normal_CSR_diff = 0
                    temp_attack_data_point_dict = {}
                    temp_normal_data_point_dict = {}

                    for symbol_name in sub_set_list:
                        param_attack_dict = \
                            primary_symbol_dict[type_name][comb_name][symbol_name][Constant_Parameters.ATTACK]
                        param_normal_dict = \
                            primary_symbol_dict[type_name][comb_name][symbol_name][Constant_Parameters.NORMAL]

                        temp_attack_CSR = param_attack_dict[Constant_Parameters.COMBINED_SAMPLING_RESOLUTION]
                        temp_normal_CSR = param_normal_dict[Constant_Parameters.COMBINED_SAMPLING_RESOLUTION]

                        if temp_attack_CSR > 0:
                            temp_attack_CSR_diff += (temp_attack_CSR / np.exp(temp_attack_CSR - attack_min_CSR))
                            param_attack_dict = \
                                cls.__get_reduced_data_point_dict(attack_min_size,
                                                                  param_attack_dict[Constant_Parameters.DATA_POINT])
                        else:
                            param_attack_dict = \
                                cls.__get_dummy_data_point_dict(attack_min_size,
                                                                param_attack_dict[Constant_Parameters.DATA_POINT])

                        if temp_normal_CSR > 0:
                            temp_normal_CSR_diff += (temp_normal_CSR / np.exp(temp_normal_CSR - normal_min_CSR))
                            param_normal_dict = \
                                cls.__get_reduced_data_point_dict(normal_min_size,
                                                                  param_normal_dict[Constant_Parameters.DATA_POINT])
                        else:
                            param_normal_dict = \
                                cls.__get_dummy_data_point_dict(normal_min_size,
                                                                param_normal_dict[Constant_Parameters.DATA_POINT])

                        temp_attack_data_point_dict[symbol_name] = param_attack_dict
                        temp_normal_data_point_dict[symbol_name] = param_normal_dict

                    feature_size = len(sub_set_list)
                    param_attack_CSR_avg = temp_attack_CSR_diff / (feature_size + 1)
                    param_normal_CSR_avg = temp_normal_CSR_diff / (feature_size + 1)

                    temp_attack_dict = {Constant_Parameters.COMBINED_SAMPLING_RESOLUTION: param_attack_CSR_avg,
                                        Constant_Parameters.DATA_POINT: temp_attack_data_point_dict}
                    temp_normal_dict = {Constant_Parameters.COMBINED_SAMPLING_RESOLUTION: param_normal_CSR_avg,
                                        Constant_Parameters.DATA_POINT: temp_normal_data_point_dict}

                    temp_sub_set_dict[str(sub_set_list)] = {Constant_Parameters.ATTACK: temp_attack_dict,
                                                            Constant_Parameters.NORMAL: temp_normal_dict}

                temp_comb_dict[comb_name] = temp_sub_set_dict

            temp_type_dict[type_name] = temp_comb_dict

        return temp_type_dict

    @classmethod
    def __get_dummy_data_point_dict(cls, reduce_size, station_dict):
        dummy_data_point_dict = {}

        for station_id in station_dict.keys():
            dummy_data_point_list = list(Constant_Parameters.DUMMY_DATA for _ in range(0, reduce_size))
            dummy_data_point_dict[station_id] = dummy_data_point_list

        return dummy_data_point_dict

    @classmethod
    def __get_reduced_data_point_dict(cls, reduce_size, station_dict):
        reduced_data_point_dict = {}

        for station_id, data_point_list in station_dict.items():
            dummy_list = list(Constant_Parameters.DUMMY_DATA for _ in range(0, reduce_size))
            reduced_data_point_list = cls.__make_same_feature_length(dummy_list, data_point_list)
            reduced_data_point_dict[station_id] = reduced_data_point_list

        return reduced_data_point_dict

    @classmethod
    def __get_min_size_and_sampling_resolution_dict(cls, sub_set_list, type_name, comb_name, primary_symbol_dict):
        temp_attack_list = []
        temp_normal_list = []

        for symbol_name in sub_set_list:
            attack_dict = primary_symbol_dict[type_name][comb_name][symbol_name][Constant_Parameters.ATTACK]
            normal_dict = primary_symbol_dict[type_name][comb_name][symbol_name][Constant_Parameters.NORMAL]

            attack_min_count = attack_dict[Constant_Parameters.MIN_SAMPLE_COUNT]
            normal_min_count = normal_dict[Constant_Parameters.MIN_SAMPLE_COUNT]
            attack_CSR = attack_dict[Constant_Parameters.COMBINED_SAMPLING_RESOLUTION]
            normal_CSR = normal_dict[Constant_Parameters.COMBINED_SAMPLING_RESOLUTION]

            attack_combined_sampling_resolution = attack_dict[Constant_Parameters.COMBINED_SAMPLING_RESOLUTION]
            normal_combined_sampling_resolution = normal_dict[Constant_Parameters.COMBINED_SAMPLING_RESOLUTION]

            if attack_combined_sampling_resolution > 0:
                temp_attack_list.append([attack_min_count, attack_CSR])
            else:
                temp_attack_list.append([normal_min_count, 0])

            if normal_combined_sampling_resolution > 0:
                temp_normal_list.append([normal_min_count, normal_CSR])
            else:
                temp_normal_list.append([attack_min_count, 0])

        sorted_attack_list = sorted(temp_attack_list, key=lambda x: x[0])
        sorted_normal_list = sorted(temp_normal_list, key=lambda x: x[0])

        smallest_attack_list = sorted_attack_list[0]
        smallest_normal_list = sorted_normal_list[0]

        return smallest_attack_list[0], smallest_normal_list[0], smallest_attack_list[1], smallest_normal_list[1]

    @classmethod
    def __get_primary_symbol_dict(cls, primary_symbol_name_dict, full_feature_dict):
        param_type_dict = {}
        for type_name, comb_dict in primary_symbol_name_dict.items():
            param_comb_dict = {}
            for comb_name, symbol_list in comb_dict.items():
                if symbol_list is not None:
                    param_symbol_dict = {}
                    for symbol_name in symbol_list:
                        attack_dict = full_feature_dict[type_name][comb_name][symbol_name][Constant_Parameters.ATTACK]
                        normal_dict = full_feature_dict[type_name][comb_name][symbol_name][Constant_Parameters.NORMAL]

                        attack_min_sam_cnt = attack_dict[Constant_Parameters.MIN_SAMPLE_COUNT]
                        normal_min_sam_cnt = normal_dict[Constant_Parameters.MIN_SAMPLE_COUNT]

                        if attack_min_sam_cnt > 0:
                            average_sampling_res_diff = attack_dict[Constant_Parameters.AVERAGE_SAMPLING_RES_DIFF]
                            average_sampling_resolution = attack_dict[Constant_Parameters.AVERAGE_SAMPLING_RESOLUTION]
                            attack_combined_SR = average_sampling_resolution / np.exp(average_sampling_res_diff)
                            attack_data_point_dict = attack_dict[Constant_Parameters.DATA_POINT]
                        else:
                            dummy_list = list(Constant_Parameters.DUMMY_DATA for _ in range(0, normal_min_sam_cnt))
                            attack_min_sam_cnt = normal_min_sam_cnt
                            attack_combined_SR = 0
                            attack_data_point_dict = {}
                            for station_id in normal_dict[Constant_Parameters.DATA_POINT].keys():
                                attack_data_point_dict[station_id] = dummy_list

                        if normal_min_sam_cnt > 0:
                            average_sampling_res_diff = normal_dict[Constant_Parameters.AVERAGE_SAMPLING_RES_DIFF]
                            average_sampling_resolution = normal_dict[Constant_Parameters.AVERAGE_SAMPLING_RESOLUTION]
                            normal_combined_SR = average_sampling_resolution / np.exp(average_sampling_res_diff)
                            normal_data_point_dict = normal_dict[Constant_Parameters.DATA_POINT]
                        else:
                            dummy_list = list(Constant_Parameters.DUMMY_DATA for _ in range(0, attack_min_sam_cnt))
                            normal_min_sam_cnt = attack_min_sam_cnt
                            normal_combined_SR = 0
                            normal_data_point_dict = {}
                            for station_id in attack_dict[Constant_Parameters.DATA_POINT].keys():
                                normal_data_point_dict[station_id] = dummy_list

                        param_attack_dict = {Constant_Parameters.MIN_SAMPLE_COUNT: attack_min_sam_cnt,
                                             Constant_Parameters.COMBINED_SAMPLING_RESOLUTION: attack_combined_SR,
                                             Constant_Parameters.DATA_POINT: attack_data_point_dict}
                        param_normal_dict = {Constant_Parameters.MIN_SAMPLE_COUNT: normal_min_sam_cnt,
                                             Constant_Parameters.COMBINED_SAMPLING_RESOLUTION: normal_combined_SR,
                                             Constant_Parameters.DATA_POINT: normal_data_point_dict}

                        param_symbol_dict[symbol_name] = {Constant_Parameters.ATTACK: param_attack_dict,
                                                          Constant_Parameters.NORMAL: param_normal_dict}

                    param_comb_dict[comb_name] = param_symbol_dict
            param_type_dict[type_name] = param_comb_dict

        return param_type_dict

    @classmethod
    def __get_primary_symbol_name_dict(cls, all_comb_dict):
        param_type_dict = {}
        for type_name, comb_dict in all_comb_dict.items():
            param_comb_dict = {}
            for comb_name, symbol_list in comb_dict.items():
                if len(symbol_list) > 0:
                    param_comb_dict[comb_name] = symbol_list[len(symbol_list) - 1]
                else:
                    param_comb_dict[comb_name] = None
            param_type_dict[type_name] = param_comb_dict

        return param_type_dict

    @classmethod
    def __get_chosen_combination_dict(cls, full_feature_dict, feature_count):
        param_type_dict = {}
        for type_name, comb_dict in full_feature_dict.items():
            param_comb_dict = {}
            for comb_name, symbol_dict in comb_dict.items():
                symbol_info_list = []
                for symbol_name, category_dict in symbol_dict.items():
                    attack_dict = category_dict[Constant_Parameters.ATTACK]
                    normal_dict = category_dict[Constant_Parameters.NORMAL]

                    attack_ASRD = attack_dict[Constant_Parameters.AVERAGE_SAMPLING_RES_DIFF]
                    attack_ASR = attack_dict[Constant_Parameters.AVERAGE_SAMPLING_RESOLUTION]
                    normal_ASRD = normal_dict[Constant_Parameters.AVERAGE_SAMPLING_RES_DIFF]
                    normal_ASR = normal_dict[Constant_Parameters.AVERAGE_SAMPLING_RESOLUTION]

                    if attack_ASR is not None and normal_ASR is not None:
                        combined_ASR = (attack_ASR + normal_ASR) / 2
                        combined_ASRD = (attack_ASRD + normal_ASRD) / 2
                    elif attack_ASR is None and normal_ASR is not None:
                        combined_ASR = normal_ASR
                        combined_ASRD = normal_ASRD
                    elif attack_ASR is not None and normal_ASR is None:
                        combined_ASR = attack_ASR
                        combined_ASRD = attack_ASRD
                    else:
                        combined_ASR = 0
                        combined_ASRD = 0

                    symbol_info_list.append([symbol_name, combined_ASR, combined_ASRD])

                # x[1]: max ASR, x[2]: min ASRD
                sorted_symbol_info_list = sorted(symbol_info_list, key=lambda x: (-x[1], x[2]))
                chosen_symbol_info_list = sorted_symbol_info_list[0:feature_count]

                chosen_symbol_list = list(symbol_name[0] for symbol_name in chosen_symbol_info_list)
                symbol_combination_list = cls.__get_combination_symbol_list(chosen_symbol_list)
                param_comb_dict[comb_name] = symbol_combination_list

            param_type_dict[type_name] = param_comb_dict

        return param_type_dict

    @classmethod
    def __get_combination_symbol_list(cls, chosen_symbol_list):
        comb_size = len(chosen_symbol_list)
        all_record_list = []
        for index in range(1, comb_size + 1):
            all_record_list.append(list(combinations(chosen_symbol_list, index)))

        param_all_record_list = []
        for record_list in all_record_list:
            for sub_record_list in record_list:
                sub_set_list = []
                for symbol_name in sub_record_list:
                    sub_set_list.append(symbol_name)
                param_all_record_list.append(sub_set_list)

        return param_all_record_list

    @classmethod
    def __get_full_individual_feature_dict(cls, basis_symbol_dict, sam_res_diff_and_mean_dict, top_dict):
        param_type_dict = {}
        for type_name, comb_dict in basis_symbol_dict.items():
            param_comb_dict = {}
            for comb_name, symbol_list in comb_dict.items():
                param_symbol_dict = {}
                for symbol_name in symbol_list:
                    attack_min_count, attack_ASRD, attack_ASR, normal_min_count, normal_ASRD, normal_ASR = \
                        cls.__get_symbol_information(symbol_name, type_name, sam_res_diff_and_mean_dict)
                    attack_data_point_dict, normal_data_point_dict = \
                        cls.__get_data_point_dict(symbol_name, attack_min_count, normal_min_count, type_name, top_dict)

                    param_symbol_dict[symbol_name] = {Constant_Parameters.ATTACK:
                                                          {Constant_Parameters.MIN_SAMPLE_COUNT: attack_min_count,
                                                           Constant_Parameters.AVERAGE_SAMPLING_RES_DIFF: attack_ASRD,
                                                           Constant_Parameters.AVERAGE_SAMPLING_RESOLUTION: attack_ASR,
                                                           Constant_Parameters.DATA_POINT: attack_data_point_dict},
                                                      Constant_Parameters.NORMAL:
                                                          {Constant_Parameters.MIN_SAMPLE_COUNT: normal_min_count,
                                                           Constant_Parameters.AVERAGE_SAMPLING_RES_DIFF: normal_ASRD,
                                                           Constant_Parameters.AVERAGE_SAMPLING_RESOLUTION: normal_ASR,
                                                           Constant_Parameters.DATA_POINT: normal_data_point_dict}}

                param_comb_dict[comb_name] = param_symbol_dict

            param_type_dict[type_name] = param_comb_dict

        return param_type_dict

    @classmethod
    def __get_symbol_information(cls, symbol_name, type_name, sam_res_diff_and_mean_dict):
        temp_attack_dict = sam_res_diff_and_mean_dict[type_name][Constant_Parameters.ATTACK]
        if symbol_name not in temp_attack_dict:
            attack_min_sample_count = 0
            attack_average_sam_res_diff = None
            attack_average_sampling_resolution = None
        else:
            attack_dict = temp_attack_dict[symbol_name]
            attack_min_sample_count = attack_dict[Constant_Parameters.MIN_SAMPLE_COUNT]
            attack_average_sam_res_diff = attack_dict[Constant_Parameters.AVERAGE_SAMPLING_RES_DIFF]
            attack_average_sampling_resolution = attack_dict[Constant_Parameters.AVERAGE_SAMPLING_RESOLUTION]

        temp_normal_dict = sam_res_diff_and_mean_dict[type_name][Constant_Parameters.NORMAL]
        if symbol_name not in temp_normal_dict:
            normal_min_sample_count = 0
            normal_average_sam_res_diff = None
            normal_average_sampling_resolution = None
        else:
            normal_dict = temp_normal_dict[symbol_name]
            normal_min_sample_count = normal_dict[Constant_Parameters.MIN_SAMPLE_COUNT]
            normal_average_sam_res_diff = normal_dict[Constant_Parameters.AVERAGE_SAMPLING_RES_DIFF]
            normal_average_sampling_resolution = normal_dict[Constant_Parameters.AVERAGE_SAMPLING_RESOLUTION]

        return attack_min_sample_count, attack_average_sam_res_diff, attack_average_sampling_resolution, \
               normal_min_sample_count, normal_average_sam_res_diff, normal_average_sampling_resolution

    @classmethod
    def __get_data_point_dict(cls, symbol_name, attack_min_count, normal_min_count, type_name, top_dict):
        station_attack_dict = {}
        station_normal_dict = {}

        for station_id, type_dict in top_dict.items():
            category_dict = type_dict[type_name]
            attack_dict = category_dict[Constant_Parameters.ATTACK]
            normal_dict = category_dict[Constant_Parameters.NORMAL]

            if symbol_name in attack_dict:
                if attack_min_count == 0:
                    attack_data_point_list = [0]
                else:
                    min_list = list(Constant_Parameters.DUMMY_DATA for _ in range(0, attack_min_count))
                    attack_data_point_list = attack_dict[symbol_name][Constant_Parameters.DATA_POINT]
                    attack_data_point_list = cls.__make_same_feature_length(min_list, attack_data_point_list)

                station_attack_dict[station_id] = attack_data_point_list
            else:
                if attack_min_count is not None:
                    attack_data_point_list = list(Constant_Parameters.DUMMY_DATA for _ in range(0, attack_min_count))
                    station_attack_dict[station_id] = attack_data_point_list
                elif normal_min_count is not None:
                    attack_data_point_list = list(Constant_Parameters.DUMMY_DATA for _ in range(0, normal_min_count))
                    station_attack_dict[station_id] = attack_data_point_list

            if symbol_name in normal_dict:
                if normal_min_count == 0:
                    normal_data_point_list = [0]
                else:
                    min_list = list(Constant_Parameters.DUMMY_DATA for _ in range(0, normal_min_count))
                    normal_data_point_list = normal_dict[symbol_name][Constant_Parameters.DATA_POINT]
                    normal_data_point_list = cls.__make_same_feature_length(min_list, normal_data_point_list)

                station_normal_dict[station_id] = normal_data_point_list
            else:
                if normal_min_count is not None:
                    normal_data_point_list = list(Constant_Parameters.DUMMY_DATA for _ in range(0, normal_min_count))
                    station_normal_dict[station_id] = normal_data_point_list
                elif attack_min_count is not None:
                    normal_data_point_list = list(Constant_Parameters.DUMMY_DATA for _ in range(0, attack_min_count))
                    station_normal_dict[station_id] = normal_data_point_list

        return station_attack_dict, station_normal_dict

    @classmethod
    def __make_same_feature_length(cls, first_feature_list, second_feature_list):
        first_len = len(first_feature_list)
        second_len = len(second_feature_list)
        temp_first_feature_list = None
        temp_second_feature_list = None
        same_size_flag = True
        largest_size = 0
        smallest_size = 0
        largest_list = None
        smallest_list = None
        first_flag = False

        if first_len > second_len:
            smallest_size = second_len
            largest_size = first_len
            largest_list = first_feature_list
            smallest_list = second_feature_list
            first_flag = True
        elif first_len < second_len:
            smallest_size = first_len
            largest_size = second_len
            largest_list = second_feature_list
            smallest_list = first_feature_list
        else:
            temp_first_feature_list = first_feature_list
            temp_second_feature_list = second_feature_list
            same_size_flag = False

        if same_size_flag:
            try:
                result = largest_size / smallest_size
            except ZeroDivisionError:
                result = 0

            quotient = math.floor(result)
            remainder = result - quotient
            total_remainder = 0
            index = 0
            chosen_data_list = []

            while index < largest_size:
                if quotient == 0:
                    temp = 0
                else:
                    temp = index % quotient

                if temp == 0:
                    if total_remainder > quotient:
                        chosen_data_list.append(largest_list[index])
                        index += quotient
                        total_remainder = total_remainder - quotient
                    else:
                        total_remainder += remainder
                        chosen_data_list.append(largest_list[index])

                index += 1

            shrunk_index_size = len(chosen_data_list)
            index_list = list(index for index in range(0, shrunk_index_size))

            random_index_list = random.sample(index_list, smallest_size)
            random_index_list = sorted(random_index_list)

            final_shrunk_list = []
            for index in random_index_list:
                final_shrunk_list.append(chosen_data_list[index])

            if first_flag:
                temp_first_feature_list = final_shrunk_list
                temp_second_feature_list = smallest_list
            else:
                temp_first_feature_list = smallest_list
                temp_second_feature_list = final_shrunk_list

        temp_first_size = len(temp_first_feature_list)
        temp_second_size = len(temp_second_feature_list)

        if temp_first_size < temp_second_size:
            temp_second_feature_list = temp_second_feature_list[0:temp_first_size]

        return temp_second_feature_list

    def __saving_data(self, file_data, file_type, station_type):
        scenario_type = self.__scenario_list

        root_path = Constant_Parameters.DATASET_PROCESSED_DATA_PATH + '/' + scenario_type[0] + '/' + scenario_type[1] \
                    + '/' + scenario_type[2] + '/' + Constant_Parameters.TOP + '/' + station_type + '/' + file_type + \
                    '.json'
        with open(root_path, 'w') as f:
            json.dump(file_data, f)

    @classmethod
    def __get_reference_symbol_dict(cls, unique_intersection_symbol_dict):
        basis_symbol_dict = {}
        for type_name, category_dict in unique_intersection_symbol_dict.items():
            attack_list = category_dict[Constant_Parameters.ATTACK]
            normal_list = category_dict[Constant_Parameters.NORMAL]

            only_attack_list = list(set(attack_list) - set(normal_list))
            only_normal_list = list(set(normal_list) - set(attack_list))

            common_list = list(set(attack_list) & set(normal_list))

            exclusive_list = []
            exclusive_list.extend(only_attack_list)
            exclusive_list.extend(only_normal_list)

            all_together_list = []
            all_together_list.extend(common_list)
            all_together_list.extend(exclusive_list)

            basis_symbol_dict[type_name] = {Constant_Parameters.COMMON: common_list,  # only both exist
                                            Constant_Parameters.EXCLUSIVE: exclusive_list,  # only attack + only normal
                                            Constant_Parameters.ALL: all_together_list}
            # only both + only attack + only normal

        return basis_symbol_dict

    @classmethod
    def __get_sam_res_diff_and_mean_dict(cls, top_dict, unique_intersection_symbol_dict):
        sam_res_diff_and_mean_dict = {}
        for type_name, type_dict in unique_intersection_symbol_dict.items():
            attack_symbol_list = type_dict[Constant_Parameters.ATTACK]
            normal_symbol_list = type_dict[Constant_Parameters.NORMAL]

            attack_sam_res_diff_and_mean_dict = \
                cls.__get_sorted_sam_res_diff_and_mean_dict(top_dict, type_name, attack_symbol_list,
                                                            Constant_Parameters.ATTACK)
            normal_sam_res_diff_and_mean_dict = \
                cls.__get_sorted_sam_res_diff_and_mean_dict(top_dict, type_name, normal_symbol_list,
                                                            Constant_Parameters.NORMAL)

            sam_res_diff_and_mean_dict[type_name] = \
                {Constant_Parameters.ATTACK: attack_sam_res_diff_and_mean_dict,
                 Constant_Parameters.NORMAL: normal_sam_res_diff_and_mean_dict}

        return sam_res_diff_and_mean_dict

    @classmethod
    def __get_sorted_sam_res_diff_and_mean_dict(cls, top_dict, type_name, symbol_list, category):
        min_list = []
        for symbol_name in symbol_list:
            temp_list = []
            min_count_index = 0
            for param_station_id, param_type_dict in top_dict.items():
                param_category_dict = param_type_dict[type_name]
                param_symbol_dict = param_category_dict[category]
                temp_dict = param_symbol_dict[symbol_name]
                sampling_count = temp_dict[Constant_Parameters.SAMPLING_COUNT]
                sampling_resolution = temp_dict[Constant_Parameters.SAMPLING_RESOLUTION]
                param_record = [param_station_id, symbol_name, sampling_resolution, sampling_count]
                temp_list.append([param_station_id, symbol_name, sampling_resolution, sampling_count])
                min_count_index = len(param_record) - 1

            sorted_temp_list = sorted(temp_list, key=lambda x: x[min_count_index])
            min_sampling_count_record = sorted_temp_list[0]

            sam_res_index = 2
            sampling_resolution_difference_list = []
            for record in sorted_temp_list:
                sampling_resolution = record[sam_res_index]
                min_sampling_resolution = min_sampling_count_record[sam_res_index]
                sampling_resolution_difference = sampling_resolution - min_sampling_resolution
                sampling_resolution_difference_list.append(sampling_resolution_difference)

            sampling_resolution_list = []
            for record in sorted_temp_list:
                sampling_resolution = record[sam_res_index]
                sampling_resolution_list.append(sampling_resolution)

            if len(sampling_resolution_difference_list) > 0:
                res_diff_mean = np.mean(sampling_resolution_difference_list)
            else:
                res_diff_mean = 0

            sam_res_mean = np.mean(sampling_resolution_list)
            param_list = [symbol_name, min_sampling_count_record[len(min_sampling_count_record) - 1],
                          res_diff_mean, sam_res_mean]
            min_list.append(param_list)

        # symbol_name, min sample count, sample_resolution_difference_mean, sample_resolution_mean
        dual_sorted_list = sorted(min_list, key=lambda x: (-x[3], x[2]))
        # min: sample_resolution_difference_mean & max: sample_resolution_mean

        param_dict = {}
        for record in dual_sorted_list:
            param_dict[record[0]] = {Constant_Parameters.MIN_SAMPLE_COUNT: record[1],
                                     Constant_Parameters.AVERAGE_SAMPLING_RES_DIFF: record[2],
                                     Constant_Parameters.AVERAGE_SAMPLING_RESOLUTION: record[3]}

        return param_dict

    def __loading_top_dataset(self, top_path_dict, id_list, type_list, attack_sim_time, normal_sim_time):
        path_dict = self.__get_top_path(top_path_dict, id_list, type_list)
        raw_data_dict = self.__get_raw_data_dict(path_dict, type_list)
        split_record_dict = self.__get_split_record(raw_data_dict)
        unique_intersection_symbol_dict = \
            self.__get_station_unique_intersection_symbol_dict(split_record_dict, type_list, id_list)
        basic_top_dict = self.__get_basic_top_symbol_dict(split_record_dict, unique_intersection_symbol_dict)
        sampling_resolution_dict = self.__get_sampling_resolution_dict(basic_top_dict, attack_sim_time, normal_sim_time)

        return sampling_resolution_dict

    @classmethod
    def __get_sampling_resolution_dict(cls, basic_top_dict, attack_sim_time, normal_sim_time):
        station_dict = {}
        for station_id, type_dict in basic_top_dict.items():
            param_type_dict = {}
            for type_, top_dict in type_dict.items():
                attack_dict = top_dict[Constant_Parameters.ATTACK]
                normal_dict = top_dict[Constant_Parameters.NORMAL]

                param_attack_dict = {}
                for symbol_name, overhead_list in attack_dict.items():
                    sampling_count = len(overhead_list)
                    sampling_resolution = sampling_count / attack_sim_time
                    param_attack_dict[symbol_name] = {Constant_Parameters.DATA_POINT: overhead_list,
                                                      Constant_Parameters.SAMPLING_COUNT: sampling_count,
                                                      Constant_Parameters.SIMULATION_TIME: attack_sim_time,
                                                      Constant_Parameters.SAMPLING_RESOLUTION: sampling_resolution}
                param_normal_dict = {}
                for symbol_name, overhead_list in normal_dict.items():
                    sampling_count = len(overhead_list)
                    sampling_resolution = sampling_count / normal_sim_time
                    param_normal_dict[symbol_name] = {Constant_Parameters.DATA_POINT: overhead_list,
                                                      Constant_Parameters.SAMPLING_COUNT: sampling_count,
                                                      Constant_Parameters.SIMULATION_TIME: normal_sim_time,
                                                      Constant_Parameters.SAMPLING_RESOLUTION: sampling_resolution}

                param_type_dict[type_] = {Constant_Parameters.ATTACK: param_attack_dict,
                                          Constant_Parameters.NORMAL: param_normal_dict}

            station_dict[station_id] = param_type_dict

        return station_dict

    @classmethod
    def __get_top_path(cls, top_path_dict, id_list, type_list):
        path_dict = {}

        for id in id_list:
            temp_list = top_path_dict[id]
            temp_list_1 = temp_list[0]
            temp_list_2 = temp_list[1]

            attack_list = None
            normal_list = None

            attack_flag = temp_list_1[0]
            if attack_flag == Constant_Parameters.ATTACK:
                attack_list = temp_list_1[1]
            else:
                normal_list = temp_list_1[1]

            normal_flag = temp_list_2[0]
            if normal_flag == Constant_Parameters.NORMAL:
                normal_list = temp_list_2[1]
            else:
                attack_list = temp_list_2[1]

            attack_path_dict = {}
            for attack in attack_list:
                for type_ in type_list:
                    attack_flag = attack[0]
                    if attack_flag == type_:
                        attack_path_dict[type_] = attack[1]

            normal_path_dict = {}
            for normal in normal_list:
                for type_ in type_list:
                    normal_flag = normal[0]
                    if normal_flag == type_:
                        normal_path_dict[type_] = normal[1]

            path_dict[id] = {Constant_Parameters.ATTACK: attack_path_dict, Constant_Parameters.NORMAL: normal_path_dict}

        return path_dict

    @classmethod
    def __get_split_record(cls, raw_data_dict):
        param_station_dict = {}
        for station_id, type_dict in raw_data_dict.items():
            param_type_dict = {}
            for type_, category_dict in type_dict.items():
                attack_list = category_dict[Constant_Parameters.ATTACK]
                normal_list = category_dict[Constant_Parameters.NORMAL]

                attack_record_list = []
                for record in attack_list:
                    if record.find(Constant_Parameters.KERNEL) > -1:
                        temp_list = record.split(Constant_Parameters.K)
                        temp_list_1 = temp_list[0].split(Constant_Parameters.PERCENT)
                        temp_list_1 = temp_list_1[0].strip()
                        temp_list_1 = float(temp_list_1)
                        temp_list_2 = temp_list[1].strip()
                        attack_record_list.append([temp_list_2, temp_list_1])

                normal_record_list = []
                for record in normal_list:
                    if record.find(Constant_Parameters.KERNEL) > -1:
                        temp_list = record.split(Constant_Parameters.K)
                        temp_list_1 = temp_list[0].split(Constant_Parameters.PERCENT)
                        temp_list_1 = temp_list_1[0].strip()
                        temp_list_1 = float(temp_list_1)
                        temp_list_2 = temp_list[1].strip()
                        normal_record_list.append([temp_list_2, temp_list_1])

                param_type_dict[type_] = {Constant_Parameters.ATTACK: attack_record_list,
                                          Constant_Parameters.NORMAL: normal_record_list}

            param_station_dict[station_id] = param_type_dict

        return param_station_dict

    def __get_station_unique_intersection_symbol_dict(self, split_record_dict, type_list, id_list):
        all_type_dict = {}
        for type_ in type_list:
            category_dict = {}

            for station_id, type_dict in split_record_dict.items():
                temp_type_dict = type_dict[type_]
                attack_list = temp_type_dict[Constant_Parameters.ATTACK]
                normal_list = temp_type_dict[Constant_Parameters.NORMAL]

                attack_symbol_list = []
                normal_symbol_list = []

                for attack in attack_list:
                    symbol = attack[0]
                    attack_symbol_list.append(symbol)
                for normal in normal_list:
                    symbol = normal[0]
                    normal_symbol_list.append(symbol)

                category_dict[station_id] = {Constant_Parameters.ATTACK: attack_symbol_list,
                                             Constant_Parameters.NORMAL: normal_symbol_list}

            all_type_dict[type_] = category_dict

        unique_intersection_symbol_dict = {}
        for type_, station_dict in all_type_dict.items():
            attack_intersection_set = set()
            normal_intersection_set = set()
            for station_id, category_dict in station_dict.items():
                attack_list = category_dict[Constant_Parameters.ATTACK]
                normal_list = category_dict[Constant_Parameters.NORMAL]

                if len(attack_intersection_set) > 0:
                    attack_intersection_set &= set(attack_list)
                else:
                    attack_intersection_set = set(attack_list)

                if len(normal_intersection_set) > 0:
                    normal_intersection_set &= set(normal_list)
                else:
                    normal_intersection_set = set(normal_list)

            unique_intersection_symbol_dict[type_] = {Constant_Parameters.ATTACK: list(attack_intersection_set),
                                                      Constant_Parameters.NORMAL: list(normal_intersection_set)}

        param_dict_1 = {}
        for id in id_list:
            param_dict_2 = {}
            for type_, category_dict in unique_intersection_symbol_dict.items():
                param_dict_2[type_] = category_dict
            param_dict_1[id] = param_dict_2

        if id_list[0] == Constant_Parameters.GS_ID:
            self.__gs_unique_intersection_symbol_dict = unique_intersection_symbol_dict
        else:
            self.__cs_unique_intersection_symbol_dict = unique_intersection_symbol_dict

        return param_dict_1

    @classmethod
    def __get_basic_top_symbol_dict(cls, split_record_dict, unique_intersection_symbol_dict):
        basic_top_dict = {}
        for station_id, type_dict in unique_intersection_symbol_dict.items():
            param_type_dict = {}
            for type_, category_dict in type_dict.items():
                attack_symbol_list = category_dict[Constant_Parameters.ATTACK]
                normal_symbol_list = category_dict[Constant_Parameters.NORMAL]

                temp_category_dict = split_record_dict[station_id][type_]
                temp_attack_list = temp_category_dict[Constant_Parameters.ATTACK]
                temp_normal_list = temp_category_dict[Constant_Parameters.NORMAL]

                param_attack_symbol_dict = {}
                for symbol_name in attack_symbol_list:
                    param_attack_list = []
                    for record in temp_attack_list:
                        temp_symbol_name = record[0]
                        temp_overhead_value = record[1]
                        if symbol_name == temp_symbol_name:
                            param_attack_list.append(temp_overhead_value)
                    param_attack_symbol_dict[symbol_name] = param_attack_list

                param_normal_symbol_dict = {}
                for symbol_name in normal_symbol_list:
                    param_normal_list = []
                    for record in temp_normal_list:
                        temp_symbol_name = record[0]
                        temp_overhead_value = record[1]
                        if symbol_name == temp_symbol_name:
                            param_normal_list.append(temp_overhead_value)
                    param_normal_symbol_dict[symbol_name] = param_normal_list

                param_type_dict[type_] = {Constant_Parameters.ATTACK: param_attack_symbol_dict,
                                          Constant_Parameters.NORMAL: param_normal_symbol_dict}

            basic_top_dict[station_id] = param_type_dict

        return basic_top_dict

    def __get_raw_data_dict(self, path_dict, type_list):
        raw_dataset_dict = {}
        for station_id, temp_dict in path_dict.items():
            attack_dict = temp_dict[Constant_Parameters.ATTACK]
            normal_dict = temp_dict[Constant_Parameters.NORMAL]

            type_dict = {}
            for type_ in type_list:
                attack_path = attack_dict[type_]
                normal_path = normal_dict[type_]

                attack_file_list = self.__loading_txt_data(attack_path)
                normal_file_list = self.__loading_txt_data(normal_path)

                type_dict[type_] = {Constant_Parameters.ATTACK: attack_file_list,
                                    Constant_Parameters.NORMAL: normal_file_list}

            raw_dataset_dict[station_id] = type_dict

        return raw_dataset_dict

    @classmethod
    def __loading_txt_data(cls, file_path):
        with open(file_path, 'r') as f:
            data_list = f.readlines()

        return data_list

    def __loading_time_diff_dataset(self, file_path_dict, gs_id):
        id_file_path_list = file_path_dict[gs_id]
        time_diff_dict = {}
        for id_file_path in id_file_path_list:
            combined_type = id_file_path[0]
            if combined_type == Constant_Parameters.ATTACK:
                attack_file_path = id_file_path[1]
                attack_temp_list = self.__loading_json_data(attack_file_path)
                time_diff_dict[Constant_Parameters.ATTACK] = attack_temp_list
            elif combined_type == Constant_Parameters.NORMAL:
                normal_file_path = id_file_path[1]
                normal_temp_list = self.__loading_json_data(normal_file_path)
                time_diff_dict[Constant_Parameters.NORMAL] = normal_temp_list

        return {gs_id: time_diff_dict}

    @classmethod
    def __loading_json_data(cls, file_path):
        with open(file_path, 'r') as f:
            dataset_dict = json.load(f)

        return dataset_dict

    @classmethod
    def __get_cs_total_feature_size_dict(cls, station_dict):
        param_3_dict = {}
        for category_name, type_dict in station_dict.items():
            param_2_dict = {}
            for type_name, symbol_list_dict in type_dict.items():
                param_1_dict = {}
                for symbol_list_name, temp_dict in symbol_list_dict.items():
                    attack_dict = temp_dict[Constant_Parameters.ATTACK]
                    normal_dict = temp_dict[Constant_Parameters.NORMAL]
                    attack_data_point_list = \
                        list(list(attack_dict[Constant_Parameters.DATA_POINT].values())[0].values())[0]
                    normal_data_point_list = \
                        list(list(normal_dict[Constant_Parameters.DATA_POINT].values())[0].values())[0]

                    attack_size = len(attack_data_point_list)
                    normal_size = len(normal_data_point_list)

                    param_1_dict[symbol_list_name] = {Constant_Parameters.ATTACK: attack_size,
                                                      Constant_Parameters.NORMAL: normal_size}

                param_2_dict[type_name] = param_1_dict
            param_3_dict[category_name] = param_2_dict

        return param_3_dict

    @classmethod
    def __get_gs_total_feature_size_dict(cls, station_dict):
        param_3_dict = {}
        for category_name, type_dict in station_dict.items():
            param_2_dict = {}
            for type_name, symbol_list_dict in type_dict.items():
                param_1_dict = {}
                for symbol_list_name, temp_dict in symbol_list_dict.items():
                    attack_dict = temp_dict[Constant_Parameters.ATTACK][Constant_Parameters.DATA_POINT]
                    normal_dict = temp_dict[Constant_Parameters.NORMAL][Constant_Parameters.DATA_POINT]

                    attack_list = list(attack_dict.values())[0][Constant_Parameters.GS_ID]
                    normal_list = list(normal_dict.values())[0][Constant_Parameters.GS_ID]

                    attack_size = len(attack_list)
                    normal_size = len(normal_list)

                    param_1_dict[symbol_list_name] = {Constant_Parameters.ATTACK: attack_size,
                                                      Constant_Parameters.NORMAL: normal_size}
                param_2_dict[type_name] = param_1_dict
            param_3_dict[category_name] = param_2_dict

        return param_3_dict

    @classmethod
    def extract_all_feature_size(cls):
        cs_size_dict = {}
        gs_size_dict = {}
        for scenario, path in Constant_Parameters.PROCESSED_DATASET_PATH_DICT.items():
            root_path = path + '/' + Constant_Parameters.TOP
            cs_dir_path = root_path + '/' + Constant_Parameters.CS
            gs_dir_path = root_path + '/' + Constant_Parameters.GS

            cs_file_path = cs_dir_path + '/' + Constant_Parameters.FINAL_DATASET + '.json'
            gs_file_path = gs_dir_path + '/' + Constant_Parameters.FINAL_DATASET + '.json'

            with open(cs_file_path, 'r') as f:
                cs_dict = json.load(f)
            with open(gs_file_path, 'r') as f:
                gs_dict = json.load(f)

            cs_temp_dict = cls.__get_cs_total_feature_size_dict(cs_dict)
            gs_temp_dict = cls.__get_gs_total_feature_size_dict(gs_dict)

            cs_size_dict[scenario] = cs_temp_dict
            gs_size_dict[scenario] = gs_temp_dict

        save_dir_path = Constant_Parameters.RESULT + '/' + Constant_Parameters.FEATURE_SIZE
        cs_save_file_path = save_dir_path + '/' + Constant_Parameters.CS_ALL_FEATURE_SIZE_FILENAME
        gs_save_file_path = save_dir_path + '/' + Constant_Parameters.GS_ALL_FEATURE_SIZE_FILENAME

        with open(cs_save_file_path, 'w') as f:
            json.dump(cs_size_dict, f)
        with open(gs_save_file_path, 'w') as f:
            json.dump(gs_size_dict, f)
