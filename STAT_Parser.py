import copy
import json
import math
import random
from itertools import combinations
import Constant_Parameters
from Dataset_Initialization import Dataset_Initialization
import numpy as np
from Time_Diff_Parser import Time_Diff_Parser


class STAT_Parser(Time_Diff_Parser):
    __gs_id: str
    __cs_id_list: list

    __cs_stat_dataset_dict: dict
    __gs_stat_dataset_dict: dict

    __attack_sim_time: float
    __normal_sim_time: float

    __scenario_list: list

    __time_diff_dataset_dict: dict

    def __init__(self, scenario_object: Dataset_Initialization):
        cs_stat_path_dict = scenario_object.get_cs_stat_file_path_dict()
        gs_stat_path_dict = scenario_object.get_gs_stat_file_path_dict()
        gs_time_diff_dict = scenario_object.get_gs_time_diff_file_path_dict()
        self.__attack_sim_time = scenario_object.get_attack_sim_time()
        self.__normal_sim_time = scenario_object.get_normal_sim_time()

        self.__scenario_list = scenario_object.get_scenario_list()

        cs_id_set = set(cs_stat_path_dict.keys())
        self.__cs_id_list = list(cs_id_set)
        gs_id_set = set(gs_stat_path_dict.keys()) & set(gs_time_diff_dict.keys())
        self.__gs_id = gs_id_set.pop()

        self.__cs_stat_dataset_dict = self.__loading_stat_dataset(cs_stat_path_dict, self.__cs_id_list)
        self.__gs_stat_dataset_dict = self.__loading_stat_dataset(gs_stat_path_dict, [self.__gs_id])
        temp_time_diff_dataset_dict = self.__loading_time_diff_dataset(gs_time_diff_dict, self.__gs_id)
        super().__init__(temp_time_diff_dataset_dict, self.__cs_id_list, self.__attack_sim_time, self.__normal_sim_time)

        self.__type_list = scenario_object.get_type_list()

    @classmethod
    def __loading_json_data(cls, file_path):
        with open(file_path, 'r') as f:
            dataset_dict = json.load(f)

        return dataset_dict

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
    def __loading_txt_data(cls, file_path):
        with open(file_path, 'r') as f:
            data_list = f.readlines()

        return data_list

    def __loading_stat_dataset(self, file_path_dict, id_list):
        stat_dict = {}

        for id in id_list:
            id_file_path_list = file_path_dict[id]
            id_combined_dict = {}
            for id_file_path in id_file_path_list:
                combined_type = id_file_path[0]
                if combined_type == Constant_Parameters.ATTACK:
                    attack_file_path = id_file_path[1]
                    attack_temp_list = self.__loading_txt_data(attack_file_path)
                    id_combined_dict[Constant_Parameters.ATTACK] = attack_temp_list
                elif combined_type == Constant_Parameters.NORMAL:
                    normal_file_path = id_file_path[1]
                    normal_temp_list = self.__loading_txt_data(normal_file_path)
                    id_combined_dict[Constant_Parameters.NORMAL] = normal_temp_list

            stat_dict[id] = id_combined_dict

        return stat_dict

    def __extracting_stat_item_list(self, combined_stat_list):
        count = 0
        cycle_record = None
        instruction_record = None
        branch_record = None

        consumed_cycle_ratio_list = []
        consumed_instruction_ratio_list = []
        consumed_branch_ratio_list = []

        for record in combined_stat_list:
            if record.find(Constant_Parameters.NOT_COUNTED) < 0:
                if record.find(Constant_Parameters.CYCLES) >= 0:
                    count += 1
                    cycle_record = record
                elif record.find(Constant_Parameters.INSTRUCTIONS) >= 0:
                    count += 1
                    instruction_record = record
                elif record.find(Constant_Parameters.BRANCH) >= 0 and \
                        record.find(Constant_Parameters.BRANCH_MISSES) < 0:
                    count += 1
                    branch_record = record
                if count > 2:
                    count = 0
                    consumed_cycle_ratio, consumed_cycles = self.__get_consumed_cycle_on_cpu_clock(cycle_record)
                    consumed_instruction_ratio, consumed_instructions = \
                        self.__get_consumed_instruction_ratio_per_cycle(instruction_record, consumed_cycles)
                    consumed_branch_ratio = \
                        self.__get_consumed_branch_ratio_per_instruction(branch_record, consumed_instructions)

                    consumed_cycle_ratio_list.append(consumed_cycle_ratio)
                    consumed_instruction_ratio_list.append(consumed_instruction_ratio)
                    consumed_branch_ratio_list.append(consumed_branch_ratio)

        param_dict = {Constant_Parameters.CYCLES: consumed_cycle_ratio_list,
                      Constant_Parameters.INSTRUCTIONS: consumed_instruction_ratio_list,
                      Constant_Parameters.BRANCH: consumed_branch_ratio_list}

        return param_dict

    @classmethod
    def __get_consumed_branch_ratio_per_instruction(cls, branch_record, consumed_instructions):
        temp_list_1 = branch_record.split()
        temp_list_2 = temp_list_1[1]
        branches = temp_list_2.replace(',', '')
        branches = float(branches)

        branch_ratio = branches / consumed_instructions

        return branch_ratio

    @classmethod
    def __get_consumed_instruction_ratio_per_cycle(cls, instruction_record, consumed_cycles):
        temp_list_1 = instruction_record.split()
        temp_list_2 = temp_list_1[1]
        instructions = temp_list_2.replace(',', '')
        instructions = float(instructions)

        instruction_ratio = instructions / consumed_cycles

        return instruction_ratio, instructions

    @classmethod
    def __get_consumed_cycle_on_cpu_clock(cls, cycle_record):
        temp_list_1 = cycle_record.split()
        temp_list_2 = temp_list_1[1]
        cpu_clock = float(temp_list_1[4])
        clock_type = temp_list_1[5]

        if clock_type == Constant_Parameters.GIGA_HERTZ:
            cpu_mhz_clock = cpu_clock * 1000 * 1000 * 1000
        elif clock_type == Constant_Parameters.MEGA_HERTZ:
            cpu_mhz_clock = cpu_clock * 1000 * 1000
        else:
            cpu_mhz_clock = cpu_clock

        cycles = temp_list_2.replace(',', '')
        cycles = float(cycles)
        cycle_ratio = cycles / cpu_mhz_clock

        return cycle_ratio, cycles

    def __parsing_stat_dataset(self, id_list, stat_dict):
        parsed_stat_dict = {}

        for id in id_list:
            combined_stat_dict = stat_dict[id]

            attack_stat_list = combined_stat_dict[Constant_Parameters.ATTACK]
            normal_stat_list = combined_stat_dict[Constant_Parameters.NORMAL]

            attack_stat_dict = self.__extracting_stat_item_list(attack_stat_list)
            normal_stat_dict = self.__extracting_stat_item_list(normal_stat_list)

            param_dict = {Constant_Parameters.ATTACK: attack_stat_dict, Constant_Parameters.NORMAL: normal_stat_dict}
            parsed_stat_dict[id] = param_dict

        return parsed_stat_dict

    def __analyzing_stat_sampling_analysis_dict(self, id_list, stat_dict):
        id_type_combined_sampling_second_dict = {}
        attack_sampling_time_dict = {}
        normal_sampling_time_dict = {}
        for type_ in self.__type_list:
            attack_sampling_time_dict[type_] = []
            normal_sampling_time_dict[type_] = []

        for id in id_list:
            id_stat_dict = stat_dict[id]
            attack_id_stat_dict = id_stat_dict[Constant_Parameters.ATTACK]
            normal_id_stat_dict = id_stat_dict[Constant_Parameters.NORMAL]

            type_combined_sampling_second_dict = {}
            for type_ in self.__type_list:
                type_attack_id_stat_list = attack_id_stat_dict[type_]
                type_normal_id_stat_list = normal_id_stat_dict[type_]

                attack_feature_size = len(type_attack_id_stat_list)
                normal_feature_size = len(type_normal_id_stat_list)

                attack_avg_freq = attack_feature_size / self.__attack_sim_time
                normal_avg_freq = normal_feature_size / self.__normal_sim_time
                attack_sampling_time_dict[type_].append(attack_avg_freq)
                normal_sampling_time_dict[type_].append(normal_avg_freq)

                attack_param_dict = {Constant_Parameters.DATA_POINT: type_attack_id_stat_list,
                                     Constant_Parameters.SAMPLING_COUNT: attack_feature_size,
                                     Constant_Parameters.SIMULATION_TIME: self.__attack_sim_time,
                                     Constant_Parameters.SAMPLING_RESOLUTION: attack_avg_freq}
                normal_param_dict = {Constant_Parameters.DATA_POINT: type_normal_id_stat_list,
                                     Constant_Parameters.SAMPLING_COUNT: normal_feature_size,
                                     Constant_Parameters.SIMULATION_TIME: self.__normal_sim_time,
                                     Constant_Parameters.SAMPLING_RESOLUTION: normal_avg_freq}

                combined_dict = {Constant_Parameters.ATTACK: attack_param_dict,
                                 Constant_Parameters.NORMAL: normal_param_dict}
                type_combined_sampling_second_dict[type_] = combined_dict

            id_type_combined_sampling_second_dict[id] = type_combined_sampling_second_dict

        return id_type_combined_sampling_second_dict

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
            result = largest_size / smallest_size
            quotient = math.floor(result)
            remainder = result - quotient
            total_remainder = 0
            index = 0
            chosen_data_list = []

            while index < largest_size:
                if index % quotient == 0:
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

    def __get_combination_list(self):
        type_list = copy.deepcopy(self.__type_list)
        type_list.append(Constant_Parameters.TIME_DIFF)

        comb_list = []
        for index in range(1, len(type_list) + 1):
            temp_list_1 = list(combinations(type_list, index))
            for sub_comb_list in temp_list_1:
                temp_list_3 = []
                for temp_list_2 in sub_comb_list:
                    temp_list_3.append(temp_list_2)
                comb_list.append(temp_list_3)

        return comb_list

    @classmethod
    def __get_feature_min_size_and_min_loss_rate(cls, comb_type, analysis_dict):
        attack_sampling_list = []
        normal_sampling_list = []

        for type_dict in analysis_dict.values():
            category_dict = type_dict[comb_type]
            attack_sampling_count = category_dict[Constant_Parameters.ATTACK][Constant_Parameters.SAMPLING_COUNT]
            attack_sampling_resolution = \
                category_dict[Constant_Parameters.ATTACK][Constant_Parameters.SAMPLING_RESOLUTION]
            normal_sampling_count = category_dict[Constant_Parameters.NORMAL][Constant_Parameters.SAMPLING_COUNT]
            normal_sampling_resolution = \
                category_dict[Constant_Parameters.NORMAL][Constant_Parameters.SAMPLING_RESOLUTION]

            attack_sampling_list.append([attack_sampling_count, attack_sampling_resolution])
            normal_sampling_list.append([normal_sampling_count, normal_sampling_resolution])

        sorted_attack_sampling_list = sorted(attack_sampling_list, key=lambda x: x[0])
        sorted_normal_sampling_list = sorted(normal_sampling_list, key=lambda x: x[0])

        attack_min_sampling = sorted_attack_sampling_list[0]
        normal_min_sampling = sorted_normal_sampling_list[0]

        attack_min_dict = {Constant_Parameters.SAMPLING_COUNT: attack_min_sampling[0],
                           Constant_Parameters.SAMPLING_RESOLUTION: attack_min_sampling[1]}
        normal_min_dict = {Constant_Parameters.SAMPLING_COUNT: normal_min_sampling[0],
                           Constant_Parameters.SAMPLING_RESOLUTION: normal_min_sampling[1]}

        return attack_min_dict, normal_min_dict

    def __get_min_feature_analysis_list(self, all_comb_list, stat_analysis_dict, time_diff_analysis_dict):
        all_min_comb_list = []
        for sub_comb_list in all_comb_list:
            attack_feature_min_list = []
            normal_feature_min_list = []

            sub_feature_dict = {}
            for sub_comb in sub_comb_list:
                if sub_comb == Constant_Parameters.TIME_DIFF:
                    attack_sub_min_dict, normal_sub_min_dict = \
                        self.__get_feature_min_size_and_min_loss_rate(sub_comb, time_diff_analysis_dict)
                else:
                    attack_sub_min_dict, normal_sub_min_dict = \
                        self.__get_feature_min_size_and_min_loss_rate(sub_comb, stat_analysis_dict)

                attack_feature_min_list.append([attack_sub_min_dict[Constant_Parameters.SAMPLING_COUNT],
                                                attack_sub_min_dict[Constant_Parameters.SAMPLING_RESOLUTION]])
                normal_feature_min_list.append([normal_sub_min_dict[Constant_Parameters.SAMPLING_COUNT],
                                                normal_sub_min_dict[Constant_Parameters.SAMPLING_RESOLUTION]])

                sub_feature_dict[sub_comb] = {Constant_Parameters.SAMPLING_COUNT:
                                                  {Constant_Parameters.ATTACK:
                                                       {attack_sub_min_dict[Constant_Parameters.SAMPLING_COUNT]},
                                                   Constant_Parameters.NORMAL:
                                                       {normal_sub_min_dict[Constant_Parameters.SAMPLING_COUNT]}},
                                              Constant_Parameters.SAMPLING_RESOLUTION:
                                                  {Constant_Parameters.ATTACK:
                                                       {attack_sub_min_dict[Constant_Parameters.SAMPLING_RESOLUTION]},
                                                   Constant_Parameters.NORMAL:
                                                       {normal_sub_min_dict[Constant_Parameters.SAMPLING_RESOLUTION]}}}

            sorted_attack_feature_list = sorted(attack_feature_min_list, key=lambda x: x[1])
            sorted_normal_feature_list = sorted(normal_feature_min_list, key=lambda x: x[1])

            attack_min_dict = sorted_attack_feature_list[0]
            normal_min_dict = sorted_normal_feature_list[0]

            sampling_count_dict = \
                {Constant_Parameters.ATTACK: attack_min_dict[0], Constant_Parameters.NORMAL: normal_min_dict[0]}
            sampling_resolution_dict = \
                {Constant_Parameters.ATTACK: attack_min_dict[1], Constant_Parameters.NORMAL: normal_min_dict[1]}

            param_dict = {Constant_Parameters.COMBINATION_LIST: sub_feature_dict,
                          Constant_Parameters.SAMPLING_COUNT: sampling_count_dict,
                          Constant_Parameters.SAMPLING_RESOLUTION: sampling_resolution_dict}

            all_min_comb_list.append(param_dict)

        return all_min_comb_list

    @classmethod
    def __get_min_feature_comb_list(cls, min_comb_list):
        all_comb_list = []

        for comb in min_comb_list:
            sub_comb_list = list(comb[Constant_Parameters.COMBINATION_LIST].keys())
            sampling_count_dict = comb[Constant_Parameters.SAMPLING_COUNT]
            all_comb_list.append({Constant_Parameters.COMBINATION_LIST: sub_comb_list,
                                  Constant_Parameters.SAMPLING_COUNT: sampling_count_dict})

        return all_comb_list

    @classmethod
    def __get_min_feature_sampling_resolution_list(cls, min_comb_list):
        all_comb_list = []

        for comb in min_comb_list:
            sub_comb_list = list(comb[Constant_Parameters.COMBINATION_LIST].keys())
            sampling_resolution_dict = comb[Constant_Parameters.SAMPLING_RESOLUTION]
            all_comb_list.append({Constant_Parameters.COMBINATION_LIST: sub_comb_list,
                                  Constant_Parameters.SAMPLING_RESOLUTION: sampling_resolution_dict})

        return all_comb_list

    def __get_meta_feature_combination_dict(self, stat_analysis_dict, time_diff_analysis_dict):
        comb_list = self.__get_combination_list()
        min_comb_list = self.__get_min_feature_analysis_list(comb_list, stat_analysis_dict, time_diff_analysis_dict)
        min_feature_size_list = self.__get_min_feature_comb_list(min_comb_list)
        min_feature_sampling_resolution_list = self.__get_min_feature_sampling_resolution_list(min_comb_list)

        return min_feature_size_list, min_feature_sampling_resolution_list

    @classmethod
    def __calculating_combination_loss_rate_list(cls, min_sampling_resolution_list, stat_dict, time_diff_dict):
        comb_loss_rate_list = []
        for comb_dict in min_sampling_resolution_list:
            attack_min_sampling_resolution = \
                comb_dict[Constant_Parameters.SAMPLING_RESOLUTION][Constant_Parameters.ATTACK]
            normal_min_sampling_resolution = \
                comb_dict[Constant_Parameters.SAMPLING_RESOLUTION][Constant_Parameters.NORMAL]

            sub_comb_list = comb_dict[Constant_Parameters.COMBINATION_LIST]

            sub_attack_diff_resolution_list = []
            sub_normal_diff_resolution_list = []
            for sub_comb in sub_comb_list:
                if sub_comb == Constant_Parameters.TIME_DIFF:
                    for temp_dict in time_diff_dict.values():
                        attack_time_diff = temp_dict[Constant_Parameters.TIME_DIFF][Constant_Parameters.ATTACK]
                        normal_time_diff = temp_dict[Constant_Parameters.TIME_DIFF][Constant_Parameters.NORMAL]
                        sub_attack_sampling_resolution = attack_time_diff[Constant_Parameters.SAMPLING_RESOLUTION]
                        sub_normal_sampling_resolution = normal_time_diff[Constant_Parameters.SAMPLING_RESOLUTION]

                        attack_diff_resolution = sub_attack_sampling_resolution - attack_min_sampling_resolution
                        normal_diff_resolution = sub_normal_sampling_resolution - normal_min_sampling_resolution

                        sub_attack_diff_resolution_list.append(attack_diff_resolution)
                        sub_normal_diff_resolution_list.append(normal_diff_resolution)
                else:
                    for temp_dict in stat_dict.values():
                        attack_time_diff = temp_dict[sub_comb][Constant_Parameters.ATTACK]
                        normal_time_diff = temp_dict[sub_comb][Constant_Parameters.NORMAL]
                        sub_attack_sampling_resolution = attack_time_diff[Constant_Parameters.SAMPLING_RESOLUTION]
                        sub_normal_sampling_resolution = normal_time_diff[Constant_Parameters.SAMPLING_RESOLUTION]

                        attack_diff_resolution = sub_attack_sampling_resolution - attack_min_sampling_resolution
                        normal_diff_resolution = sub_normal_sampling_resolution - normal_min_sampling_resolution

                        sub_attack_diff_resolution_list.append(attack_diff_resolution)
                        sub_normal_diff_resolution_list.append(normal_diff_resolution)

            sub_attack_diff_resolution_avg = np.mean(sub_attack_diff_resolution_list)
            sub_normal_diff_resolution_avg = np.mean(sub_normal_diff_resolution_list)

            param_dict = \
                {Constant_Parameters.COMBINATION_LIST: sub_comb_list,
                 Constant_Parameters.COMBINED_LOSS_RATE:
                     {Constant_Parameters.ATTACK: sub_attack_diff_resolution_avg,
                      Constant_Parameters.NORMAL: sub_normal_diff_resolution_avg}}

            comb_loss_rate_list.append(param_dict)

        return comb_loss_rate_list

    def __get_basic_feature_dict(self, feature_type, feature_dict, attack_min_list, normal_min_list):
        basic_feature_dict = {}

        for station_id, station_dict in feature_dict.items():
            attack_feature_list = station_dict[feature_type][Constant_Parameters.ATTACK][Constant_Parameters.DATA_POINT]
            normal_feature_list = station_dict[feature_type][Constant_Parameters.NORMAL][Constant_Parameters.DATA_POINT]

            attack_same_sized_list = self.__make_same_feature_length(attack_min_list, attack_feature_list)
            normal_same_sized_list = self.__make_same_feature_length(normal_min_list, normal_feature_list)

            basic_feature_dict[station_id] = {Constant_Parameters.ATTACK: attack_same_sized_list,
                                              Constant_Parameters.NORMAL: normal_same_sized_list}

        return basic_feature_dict

    def __get_same_sized_feature_list(self, min_size_list, stat_dict, time_diff_dict):
        feature_combination_list = []
        for sub_comb_dict in min_size_list:
            attack_min_size = sub_comb_dict[Constant_Parameters.SAMPLING_COUNT][Constant_Parameters.ATTACK]
            normal_min_size = sub_comb_dict[Constant_Parameters.SAMPLING_COUNT][Constant_Parameters.NORMAL]
            attack_min_list = list(Constant_Parameters.DUMMY_DATA for _ in range(0, attack_min_size))
            normal_min_list = list(Constant_Parameters.DUMMY_DATA for _ in range(0, normal_min_size))

            sub_comb_list = sub_comb_dict[Constant_Parameters.COMBINATION_LIST]
            sub_comb_dict = {}
            for sub_comb in sub_comb_list:
                if sub_comb == Constant_Parameters.TIME_DIFF:
                    basic_feature_dict = \
                        self.__get_basic_feature_dict(sub_comb, time_diff_dict, attack_min_list, normal_min_list)
                else:
                    basic_feature_dict = \
                        self.__get_basic_feature_dict(sub_comb, stat_dict, attack_min_list, normal_min_list)

                sub_comb_dict[sub_comb] = basic_feature_dict

            feature_combination_list.append(sub_comb_dict)

        return feature_combination_list

    @classmethod
    def __get_training_and_testing_feature_array(cls, attack_list, normal_list):
        attack_array = np.array(attack_list).T
        normal_array = np.array(normal_list).T
        np.random.shuffle(attack_array)
        np.random.shuffle(normal_array)

        attack_training_size = attack_array.shape[0] * Constant_Parameters.TRAINING_SET_RATIO
        attack_training_size = round(attack_training_size)
        attack_testing_size = attack_array.shape[0] - attack_training_size
        normal_training_size = normal_array.shape[0] * Constant_Parameters.TRAINING_SET_RATIO
        normal_training_size = round(normal_training_size)
        normal_testing_size = normal_array.shape[0] - normal_training_size

        attack_training_array = attack_array[0:attack_training_size]
        attack_testing_array = attack_array[attack_training_size:attack_training_size + attack_testing_size]

        normal_training_array = normal_array[0:normal_training_size]
        normal_testing_array = normal_array[normal_training_size:normal_training_size + normal_testing_size]

        attack_training_label = \
            np.array(list(Constant_Parameters.ATTACK_LABEL for _ in range(0, attack_training_array.shape[0])))
        attack_training_label = attack_training_label.reshape(-1, 1)

        attack_testing_label = \
            np.array(list(Constant_Parameters.ATTACK_LABEL for _ in range(0, attack_testing_array.shape[0])))
        attack_testing_label = attack_testing_label.reshape(-1, 1)

        normal_training_label = \
            np.array(list(Constant_Parameters.NORMAL_LABEL for _ in range(0, normal_training_array.shape[0])))
        normal_training_label = normal_training_label.reshape(-1, 1)

        normal_testing_label = \
            np.array(list(Constant_Parameters.NORMAL_LABEL for _ in range(0, normal_testing_array.shape[0])))
        normal_testing_label = normal_testing_label.reshape(-1, 1)

        training_attack_array = np.concatenate((attack_training_array, attack_training_label), axis=1)
        training_normal_array = np.concatenate((normal_training_array, normal_training_label), axis=1)
        testing_attack_array = np.concatenate((attack_testing_array, attack_testing_label), axis=1)
        testing_normal_array = np.concatenate((normal_testing_array, normal_testing_label), axis=1)

        training_array = np.concatenate((training_attack_array, training_normal_array), axis=0)
        testing_array = np.concatenate((testing_attack_array, testing_normal_array), axis=0)
        np.random.shuffle(training_array)
        np.random.shuffle(testing_array)

        training_feature_array = training_array[0:training_array.shape[0], 0:training_array.shape[1] - 1]
        training_label_array = \
            training_array[0:training_array.shape[0], training_array.shape[1] - 1:training_array.shape[1]]
        testing_feature_array = testing_array[0:testing_array.shape[0], 0:testing_array.shape[1] - 1]
        testing_label_array = \
            testing_array[0:testing_array.shape[0], testing_array.shape[1] - 1:testing_array.shape[1]]

        return training_feature_array, training_label_array, testing_feature_array, testing_label_array

    def __converting_to_ml_feature_list(self, feature_list):
        combination_feature_list = []
        for comb_dict in feature_list:
            attack_list = []
            normal_list = []
            feature_type_list = []

            for feature_type, station_dict in comb_dict.items():
                feature_type_list.append(feature_type)
                for temp_dict in station_dict.values():
                    sub_attack_list = temp_dict[Constant_Parameters.ATTACK]
                    sub_normal_list = temp_dict[Constant_Parameters.NORMAL]
                    attack_list.append(sub_attack_list)
                    normal_list.append(sub_normal_list)

            training_feature_array, training_label_array, testing_feature_array, testing_label_array = \
                self.__get_training_and_testing_feature_array(attack_list, normal_list)

            param_dict = {Constant_Parameters.COMBINATION_LIST: feature_type_list,
                          Constant_Parameters.TRAINING_FEATURE: training_feature_array.tolist(),
                          Constant_Parameters.TRAINING_LABEL: training_label_array.tolist(),
                          Constant_Parameters.TESTING_FEATURE: testing_feature_array.tolist(),
                          Constant_Parameters.TESTING_LABEL: testing_label_array.tolist()}

            combination_feature_list.append(param_dict)

        return combination_feature_list

    @classmethod
    def __get_key_from_comb(cls, combination_list):
        type_name = ''
        for comb in combination_list:
            type_name += comb + '_'

        return type_name[:-1]

    def __saving_analysis_information(self, min_loss_rate_list, loss_rate_list, station_type):
        combined_min_loss_rate_dict = {}
        for temp_dict in min_loss_rate_list:
            comb_list = temp_dict[Constant_Parameters.COMBINATION_LIST]
            type_name = self.__get_key_from_comb(comb_list)
            sampling_resolution_dict = temp_dict[Constant_Parameters.SAMPLING_RESOLUTION]
            combined_min_loss_rate_dict[type_name] = sampling_resolution_dict

        clr_dict = {}
        for temp_dict in loss_rate_list:
            comb_list = temp_dict[Constant_Parameters.COMBINATION_LIST]
            type_name = self.__get_key_from_comb(comb_list)
            temp_sr_dict = temp_dict[Constant_Parameters.COMBINED_LOSS_RATE]
            clr_dict[type_name] = temp_sr_dict

        root_dir_path = self.__get_processed_data_root_dir_path(station_type)
        min_clr_filename = Constant_Parameters.MIN_COMBINED_LOSS_RATE + '.json'
        clr_filename = Constant_Parameters.COMBINED_LOSS_RATE + '.json'

        min_clr_path = root_dir_path + '/' + min_clr_filename
        clr_path = root_dir_path + '/' + clr_filename

        with open(min_clr_path, 'w') as f:
            json.dump(combined_min_loss_rate_dict, f)
        with open(clr_path, 'w') as f:
            json.dump(clr_dict, f)

    def __get_processed_data_root_dir_path(self, station_type):
        scenario_list = self.__scenario_list
        sub_path = ''

        for root_path in Constant_Parameters.PROCESSED_DATASET_PATH_DICT.values():
            temp_path = root_path.split('/')
            temp_path_list = temp_path[2:]
            matched_chunk = ''
            match_count = 0
            for chunk in temp_path_list:
                if chunk in scenario_list:
                    matched_chunk += chunk + '/'
                    match_count += 1

            if match_count == 3:
                sub_path = Constant_Parameters.DATASET_PROCESSED_DATA_PATH + '/'
                sub_path += matched_chunk + Constant_Parameters.STAT_PATH + '/' + station_type
                break

        return sub_path

    def __saving_original_dataset(self, stat_dict, time_diff_dict, station_type):
        root_dir_path = self.__get_processed_data_root_dir_path(station_type)
        stat_file_path = root_dir_path + '/' + Constant_Parameters.STAT_PATH + '.json'
        time_diff_file_path = root_dir_path + '/' + Constant_Parameters.TIME_DIFF + '.json'

        with open(stat_file_path, 'w') as f:
            json.dump(stat_dict, f)
        with open(time_diff_file_path, 'w') as f:
            json.dump(time_diff_dict, f)

    def __get_ml_root_dir_path(self, station_type):
        scenario_list = self.__scenario_list
        matched_chunk = scenario_list[0] + '_' + scenario_list[1] + '_' + scenario_list[2]

        sub_path = Constant_Parameters.ML_DATASET_PATH + '/' + Constant_Parameters.STAT_PATH + '/'
        sub_path += station_type + '/' + matched_chunk

        return sub_path

    def __saving_ml_feature(self, feature_list, station_type):
        feature_dict = {}
        for temp_dict in feature_list:
            comb_list = temp_dict[Constant_Parameters.COMBINATION_LIST]
            type_name = self.__get_key_from_comb(comb_list)
            temp_dict.pop(Constant_Parameters.COMBINATION_LIST)
            feature_dict[type_name] = temp_dict

        file_path = self.__get_ml_root_dir_path(station_type) + '.json'
        with open(file_path, 'w') as f:
            json.dump(feature_dict, f)

    def run(self):
        cs_stat_dict = self.__parsing_stat_dataset(self.__cs_id_list, self.__cs_stat_dataset_dict)
        gs_stat_dict = self.__parsing_stat_dataset([self.__gs_id], self.__gs_stat_dataset_dict)

        cs_analysis_dict = self.__analyzing_stat_sampling_analysis_dict(self.__cs_id_list, cs_stat_dict)
        gs_analysis_dict = self.__analyzing_stat_sampling_analysis_dict([self.__gs_id], gs_stat_dict)
        time_diff_analysis_dict = self.analyzing_time_diff_sampling_analysis_dict()

        cs_min_size_list, cs_min_loss_rate_list = \
            self.__get_meta_feature_combination_dict(cs_analysis_dict, time_diff_analysis_dict)
        gs_min_size_list, gs_min_loss_rate_list = \
            self.__get_meta_feature_combination_dict(gs_analysis_dict, time_diff_analysis_dict)

        cs_loss_rate_list = self.__calculating_combination_loss_rate_list(cs_min_loss_rate_list,
                                                                          cs_analysis_dict, time_diff_analysis_dict)
        gs_loss_rate_list = self.__calculating_combination_loss_rate_list(gs_min_loss_rate_list,
                                                                          gs_analysis_dict, time_diff_analysis_dict)

        cs_combined_feature_list = \
            self.__get_same_sized_feature_list(cs_min_size_list, cs_analysis_dict, time_diff_analysis_dict)
        gs_combined_feature_list = \
            self.__get_same_sized_feature_list(gs_min_size_list, gs_analysis_dict, time_diff_analysis_dict)

        cs_combination_feature_list = self.__converting_to_ml_feature_list(cs_combined_feature_list)
        gs_combination_feature_list = self.__converting_to_ml_feature_list(gs_combined_feature_list)

        self.__saving_analysis_information(cs_min_loss_rate_list, cs_loss_rate_list, Constant_Parameters.CS)
        self.__saving_analysis_information(gs_min_loss_rate_list, gs_loss_rate_list, Constant_Parameters.GS)
        self.__saving_original_dataset(cs_analysis_dict, time_diff_analysis_dict, Constant_Parameters.CS)
        self.__saving_original_dataset(gs_analysis_dict, time_diff_analysis_dict, Constant_Parameters.GS)
        self.__saving_ml_feature(cs_combination_feature_list, Constant_Parameters.CS)
        self.__saving_ml_feature(gs_combination_feature_list, Constant_Parameters.GS)
