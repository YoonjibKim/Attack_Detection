import csv
import json
import os
import numpy as np
import sklearn
from scipy.sparse import csc_array
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import Constant_Parameters
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings


class TOP_Record_Analysis:
    @classmethod
    def __get_scenario_dir_path(cls, scenario_list):
        scenario = scenario_list[0] + '_' + scenario_list[1] + '_' + scenario_list[2]

        raw_data_path_dict = None
        processed_data_path_dict = None

        if Constant_Parameters.CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_OFF == scenario:
            raw_data_path_dict = Constant_Parameters.RAW_DATA_DICT[Constant_Parameters.CID_RCOFF_GOFF]
            processed_data_path_dict = \
                Constant_Parameters.PROCESSED_DATASET_PATH_DICT[Constant_Parameters.CID_RCOFF_GOFF]
        elif Constant_Parameters.CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_ON == scenario:
            raw_data_path_dict = Constant_Parameters.RAW_DATA_DICT[Constant_Parameters.CID_RCOFF_GON]
            processed_data_path_dict = \
                Constant_Parameters.PROCESSED_DATASET_PATH_DICT[Constant_Parameters.CID_RCOFF_GON]
        elif Constant_Parameters.CORRECT_ID_RANDOM_CS_ON_GAUSSAIN_OFF == scenario:
            raw_data_path_dict = Constant_Parameters.RAW_DATA_DICT[Constant_Parameters.CID_RCON_GOFF]
            processed_data_path_dict = \
                Constant_Parameters.PROCESSED_DATASET_PATH_DICT[Constant_Parameters.CID_RCON_GOFF]
        elif Constant_Parameters.CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_ON == scenario:
            raw_data_path_dict = Constant_Parameters.RAW_DATA_DICT[Constant_Parameters.CID_RCON_GON]
            processed_data_path_dict = \
                Constant_Parameters.PROCESSED_DATASET_PATH_DICT[Constant_Parameters.CID_RCON_GON]
        elif Constant_Parameters.WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_OFF == scenario:
            raw_data_path_dict = Constant_Parameters.RAW_DATA_DICT[Constant_Parameters.WCT_RCOFF_GOFF]
            processed_data_path_dict = \
                Constant_Parameters.PROCESSED_DATASET_PATH_DICT[Constant_Parameters.WCT_RCOFF_GOFF]
        elif Constant_Parameters.WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_ON == scenario:
            raw_data_path_dict = Constant_Parameters.RAW_DATA_DICT[Constant_Parameters.WCT_RCOFF_GON]
            processed_data_path_dict = \
                Constant_Parameters.PROCESSED_DATASET_PATH_DICT[Constant_Parameters.WCT_RCOFF_GON]
        elif Constant_Parameters.WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_OFF == scenario:
            raw_data_path_dict = Constant_Parameters.RAW_DATA_DICT[Constant_Parameters.WCT_RCON_GOFF]
            processed_data_path_dict = \
                Constant_Parameters.PROCESSED_DATASET_PATH_DICT[Constant_Parameters.WCT_RCON_GOFF]
        elif Constant_Parameters.WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_ON == scenario:
            raw_data_path_dict = Constant_Parameters.RAW_DATA_DICT[Constant_Parameters.WCT_RCON_GON]
            processed_data_path_dict = \
                Constant_Parameters.PROCESSED_DATASET_PATH_DICT[Constant_Parameters.WCT_RCON_GON]
        elif Constant_Parameters.WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_OFF == scenario:
            raw_data_path_dict = Constant_Parameters.RAW_DATA_DICT[Constant_Parameters.WET_RCOFF_GOFF]
            processed_data_path_dict = \
                Constant_Parameters.PROCESSED_DATASET_PATH_DICT[Constant_Parameters.WET_RCOFF_GOFF]
        elif Constant_Parameters.WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_ON == scenario:
            raw_data_path_dict = Constant_Parameters.RAW_DATA_DICT[Constant_Parameters.WET_RCOFF_GON]
            processed_data_path_dict = \
                Constant_Parameters.PROCESSED_DATASET_PATH_DICT[Constant_Parameters.WET_RCOFF_GON]
        elif Constant_Parameters.WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_OFF == scenario:
            raw_data_path_dict = Constant_Parameters.RAW_DATA_DICT[Constant_Parameters.WET_RCON_GOFF]
            processed_data_path_dict = \
                Constant_Parameters.PROCESSED_DATASET_PATH_DICT[Constant_Parameters.WET_RCON_GOFF]
        elif Constant_Parameters.WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_ON == scenario:
            raw_data_path_dict = Constant_Parameters.RAW_DATA_DICT[Constant_Parameters.WET_RCON_GON]
            processed_data_path_dict = \
                Constant_Parameters.PROCESSED_DATASET_PATH_DICT[Constant_Parameters.WET_RCON_GON]
        elif Constant_Parameters.WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_OFF == scenario:
            raw_data_path_dict = Constant_Parameters.RAW_DATA_DICT[Constant_Parameters.WID_RCOFF_GOFF]
            processed_data_path_dict = \
                Constant_Parameters.PROCESSED_DATASET_PATH_DICT[Constant_Parameters.WID_RCOFF_GOFF]
        elif Constant_Parameters.WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_ON == scenario:
            raw_data_path_dict = Constant_Parameters.RAW_DATA_DICT[Constant_Parameters.WID_RCOFF_GON]
            processed_data_path_dict = \
                Constant_Parameters.PROCESSED_DATASET_PATH_DICT[Constant_Parameters.WID_RCOFF_GON]
        elif Constant_Parameters.WRONG_ID_RANDOM_CS_ON_GAUSSIAN_OFF == scenario:
            raw_data_path_dict = Constant_Parameters.RAW_DATA_DICT[Constant_Parameters.WID_RCON_GOFF]
            processed_data_path_dict = \
                Constant_Parameters.PROCESSED_DATASET_PATH_DICT[Constant_Parameters.WID_RCON_GOFF]
        elif Constant_Parameters.WRONG_ID_RANDOM_CS_ON_GAUSSIAN_ON == scenario:
            raw_data_path_dict = Constant_Parameters.RAW_DATA_DICT[Constant_Parameters.WID_RCON_GON]
            processed_data_path_dict = \
                Constant_Parameters.PROCESSED_DATASET_PATH_DICT[Constant_Parameters.WID_RCON_GON]

        return raw_data_path_dict, processed_data_path_dict

    @classmethod
    def __get_gs_record_file_name_list(cls, gs_record_path):
        file_ls = []
        for path, dir, files in os.walk(gs_record_path):
            for file in files:
                current = os.path.join(path, file).replace('\\', '/')
                if current.rfind('.txt') > 0:
                    file_ls.append(current)

        return file_ls

    @classmethod
    def __get_record_list(cls, record_file_path):
        with open(record_file_path, 'r') as f:
            lines = f.readlines()

        return lines

    @classmethod
    def __get_category_record_list(cls, record_list):
        cycle = False
        instruction = False
        branch = False

        cycle_dict = {}
        branch_dict = {}
        instruction_dict = {}

        for record in record_list:
            if record.find('of event ') > 1 and record.find(Constant_Parameters.CYCLES) > 0:
                cycle = True
                instruction = False
                branch = False
            elif record.find('of event ') > 1 and record.find(Constant_Parameters.INSTRUCTIONS) > 0:
                instruction = True
                cycle = False
                branch = False
            elif record.find('of event ') > 1 and record.find(Constant_Parameters.BRANCH) > 0:
                branch = True
                cycle = False
                instruction = False
            else:
                if cycle is True:
                    if record.find('[k]') > 0:
                        temp = record.split()
                        cycle_dict[temp[len(temp) - 1]] = temp[0][:-1]
                elif instruction is True:
                    if record.find('[k]') > 0:
                        temp = record.split()
                        instruction_dict[temp[len(temp) - 1]] = temp[0][:-1]
                elif branch is True:
                    if record.find('[k]') > 0:
                        temp = record.split()
                        branch_dict[temp[len(temp) - 1]] = temp[0][:-1]

        sorted_cycle_tuple = sorted(cycle_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_instruction_tuple = sorted(instruction_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_branch_tuple = sorted(branch_dict.items(), key=lambda x: x[1], reverse=True)

        sorted_cycle_dict = dict(sorted_cycle_tuple)
        sorted_instruction_dict = dict(sorted_instruction_tuple)
        sorted_branch_dict = dict(sorted_branch_tuple)

        return sorted_cycle_dict, sorted_instruction_dict, sorted_branch_dict

    @classmethod
    def __extract_pid(cls, record_file_path: str):
        temp_list = record_file_path.split('_')
        temp_str = temp_list[len(temp_list) - 1]
        temp_list = temp_str.split('.')
        pid = temp_list[0]

        return pid

    @classmethod
    def __get_basic_symbol_list(cls, processed_data_dir_path_dict):
        root_processed_data_dir_path = processed_data_dir_path_dict + '/' + Constant_Parameters.TOP
        cs_processed_data_dir_path = root_processed_data_dir_path + '/' + Constant_Parameters.CS
        gs_processed_data_dir_path = root_processed_data_dir_path + '/' + Constant_Parameters.GS
        cs_feature_comb_path = cs_processed_data_dir_path + '/' + Constant_Parameters.FEATURE_COMBINATION + '.json'
        gs_feature_comb_path = gs_processed_data_dir_path + '/' + Constant_Parameters.FEATURE_COMBINATION + '.json'

        with open(cs_feature_comb_path, 'r') as f:
            cs_feature_comb_dict = json.load(f)
        with open(gs_feature_comb_path, 'r') as f:
            gs_feature_comb_dict = json.load(f)

        cs_basic_feature_comb_dict = {}
        for category_name, type_dict in cs_feature_comb_dict.items():
            param_dict = {}
            for type_name, symbol_list in type_dict.items():
                symbol_list_size = len(symbol_list)
                if symbol_list_size > 0:
                    basic_symbol_list = symbol_list[len(symbol_list) - 1]
                    param_dict[type_name] = basic_symbol_list
            cs_basic_feature_comb_dict[category_name] = param_dict

        gs_basic_feature_comb_dict = {}
        for category_name, type_dict in gs_feature_comb_dict.items():
            param_dict = {}
            for type_name, symbol_list in type_dict.items():
                symbol_list_size = len(symbol_list)
                if symbol_list_size > 0:
                    basic_symbol_list = symbol_list[len(symbol_list) - 1]
                    param_dict[type_name] = basic_symbol_list
            gs_basic_feature_comb_dict[category_name] = param_dict

        return cs_basic_feature_comb_dict, gs_basic_feature_comb_dict

    @classmethod
    def __get_symbol_index(cls, param_symbol_name, symbol_dict: dict):
        symbol_list = list(symbol_dict.keys())

        found_index = Constant_Parameters.DUMMY_DATA
        for index, symbol_name in enumerate(symbol_list):
            if symbol_name == param_symbol_name:
                found_index = index
                break

        return found_index

    def __get_symbol_ratio_and_index(self, record_dict: dict, basic_feature_comb_dict: dict):
        param_3_dict = {}
        for category_1_name, type_dict in basic_feature_comb_dict.items():
            param_2_dict = {}
            for type_name, symbol_1_list in type_dict.items():
                param_1_dict = {}
                for symbol_name in symbol_1_list:
                    param_0_dict = {}
                    overhead_list = []
                    index_list = []
                    for cs_id, category_dict in record_dict.items():
                        symbol_dict = category_dict[category_1_name]
                        if symbol_name in symbol_dict:
                            overhead = symbol_dict[symbol_name]
                            symbol_index = self.__get_symbol_index(symbol_name, symbol_dict)
                            overhead_list.append(float(overhead))
                            index_list.append(symbol_index)
                        else:
                            overhead = Constant_Parameters.DUMMY_DATA
                            symbol_index = Constant_Parameters.DUMMY_DATA

                        param_0_dict[cs_id] = {Constant_Parameters.OVERHEAD_RATIO: overhead,
                                               Constant_Parameters.SYMBOL_INDEX: symbol_index}

                    if len(overhead_list) > 0:
                        overhead_mean = np.mean(overhead_list)
                    else:
                        overhead_mean = Constant_Parameters.DUMMY_DATA

                    if len(index_list) > 0:
                        index_mean = np.mean(index_list)
                    else:
                        index_mean = Constant_Parameters.DUMMY_DATA

                    param_1_dict[symbol_name] = {Constant_Parameters.RECORD_INFORMATION: param_0_dict,
                                                 Constant_Parameters.OVERHEAD_RATIO_MEAN: overhead_mean,
                                                 Constant_Parameters.SYMBOL_INDEX_MEAN: index_mean}
                param_2_dict[type_name] = param_1_dict
            param_3_dict[category_1_name] = param_2_dict

        return param_3_dict

    @classmethod
    def __calculate_silhouette_score_tuple(cls, data_list) -> tuple:
        k_size = round(len(data_list) / 10)
        k_list = list(k for k in range(2, k_size))
        data_array = np.array(data_list).reshape(-1, 1)

        k_score_dict = {}
        for k in k_list:
            kmeans = KMeans(n_clusters=k, n_init=10)
            kmeans.fit(data_array)
            clusters = kmeans.predict(data_array)
            try:
                score = silhouette_score(data_array, clusters)
            except ValueError:
                score = Constant_Parameters.DUMMY_DATA
            k_score_dict[k] = score

        k_score_dict = sorted(k_score_dict.items(), key=lambda x: x[1], reverse=True)

        return k_score_dict[0], k_score_dict

    def __find_each_station_elbow_point_dict(self, record_dict: dict):
        param_3_dict = {}
        for station_id, category_dict in record_dict.items():
            param_1_dict = {}
            for category_name, symbol_dict in category_dict.items():
                overhead_list = list(float(data) for data in symbol_dict.values())
                best_score_tuple, all_score_tuple_list = self.__calculate_silhouette_score_tuple(overhead_list)
                best_k = best_score_tuple[0]
                best_score = best_score_tuple[1]
                other_scores = all_score_tuple_list

                param_2_dict = {Constant_Parameters.BEST_K: best_k,
                                Constant_Parameters.BEST_SILHOUETTE_SCORE: best_score,
                                Constant_Parameters.OTHER_SILHOUETTE_SCORES: other_scores}

                param_1_dict[category_name] = param_2_dict
            param_3_dict[station_id] = param_1_dict

        return param_3_dict

    @classmethod
    def __extract_cs_common_symbols(cls, record_dict):
        cycle_symbol_list = []
        branch_symbol_list = []
        instruction_symbol_list = []
        for category_dict in record_dict.values():
            for category_name, symbol_dict in category_dict.items():
                temp_symbol_list = []
                for symbol_name in symbol_dict.keys():
                    temp_symbol_list.append(symbol_name)

                if Constant_Parameters.CYCLES == category_name:
                    cycle_symbol_list.append(temp_symbol_list)
                elif Constant_Parameters.BRANCH == category_name:
                    branch_symbol_list.append(temp_symbol_list)
                elif Constant_Parameters.INSTRUCTIONS == category_name:
                    instruction_symbol_list.append(temp_symbol_list)

        prev_set = set(cycle_symbol_list[0])
        for index in range(1, len(cycle_symbol_list)):
            next_set = set(cycle_symbol_list[index])
            prev_set &= next_set
        common_cycle_list = list(prev_set)

        prev_set = set(branch_symbol_list[0])
        for index in range(1, len(branch_symbol_list)):
            next_set = set(branch_symbol_list[index])
            prev_set &= next_set
        common_branch_list = list(prev_set)

        prev_set = set(instruction_symbol_list[0])
        for index in range(1, len(instruction_symbol_list)):
            next_set = set(instruction_symbol_list[index])
            prev_set &= next_set
        common_instruction_list = list(prev_set)

        return common_cycle_list, common_branch_list, common_instruction_list

    @classmethod
    def __get_average_overhead_dict(cls, common_cycle_list, common_branch_list, common_instruction_list, record_dict):
        cycle_dict = {}
        for symbol_name in common_cycle_list:
            overhead_list = []
            for category_dict in record_dict.values():
                symbol_dict = category_dict[Constant_Parameters.CYCLES]
                overhead = symbol_dict[symbol_name]
                overhead_list.append(float(overhead))
            avg_overhead = np.mean(overhead_list)
            cycle_dict[symbol_name] = round(float(avg_overhead), 2)

        branch_dict = {}
        for symbol_name in common_branch_list:
            overhead_list = []
            for category_dict in record_dict.values():
                symbol_dict = category_dict[Constant_Parameters.BRANCH]
                overhead = symbol_dict[symbol_name]
                overhead_list.append(float(overhead))
            avg_overhead = np.mean(overhead_list)
            branch_dict[symbol_name] = round(float(avg_overhead), 2)

        instruction_dict = {}
        for symbol_name in common_instruction_list:
            overhead_list = []
            for category_dict in record_dict.values():
                symbol_dict = category_dict[Constant_Parameters.INSTRUCTIONS]
                overhead = symbol_dict[symbol_name]
                overhead_list.append(float(overhead))
            avg_overhead = np.mean(overhead_list)
            instruction_dict[symbol_name] = round(float(avg_overhead), 2)

        sorted_cycle_symbol_dict = sorted(cycle_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_branch_symbol_dict = sorted(branch_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_instruction_symbol_dict = sorted(instruction_dict.items(), key=lambda x: x[1], reverse=True)

        return dict(sorted_cycle_symbol_dict), dict(sorted_branch_symbol_dict), dict(sorted_instruction_symbol_dict)

    @sklearn.utils._testing.ignore_warnings(category=ConvergenceWarning)
    def __find_avg_cs_elbow_point_dict(self, cs_common_symbol_dict):
        attack_dict = cs_common_symbol_dict[Constant_Parameters.ATTACK]
        normal_dict = cs_common_symbol_dict[Constant_Parameters.NORMAL]

        attack_overhead_dict = {}
        for category_name, symbol_dict in attack_dict.items():
            overhead_list = list(symbol_dict.values())
            best_score_tuple, all_score_tuple_list = self.__calculate_silhouette_score_tuple(overhead_list)
            best_k = best_score_tuple[0]
            best_score = best_score_tuple[1]
            other_scores = all_score_tuple_list

            param_dict = {Constant_Parameters.BEST_K: best_k, Constant_Parameters.BEST_SILHOUETTE_SCORE: best_score,
                          Constant_Parameters.OTHER_SILHOUETTE_SCORES: other_scores}

            attack_overhead_dict[category_name] = param_dict

        normal_overhead_dict = {}
        for category_name, symbol_dict in normal_dict.items():
            overhead_list = list(symbol_dict.values())
            best_score_tuple, all_score_tuple_list = self.__calculate_silhouette_score_tuple(overhead_list)
            best_k = best_score_tuple[0]
            best_score = best_score_tuple[1]
            other_scores = all_score_tuple_list

            param_dict = {Constant_Parameters.BEST_K: best_k, Constant_Parameters.BEST_SILHOUETTE_SCORE: best_score,
                          Constant_Parameters.OTHER_SILHOUETTE_SCORES: other_scores}

            normal_overhead_dict[category_name] = param_dict

        avg_elbow_point_dict = {Constant_Parameters.ATTACK: attack_overhead_dict,
                                Constant_Parameters.NORMAL: normal_overhead_dict}

        return avg_elbow_point_dict

    @sklearn.utils._testing.ignore_warnings(category=ConvergenceWarning)
    def run_top_record_analysis(self, cs_id_list, scenario_list):
        print('The record analysis of ' + str(scenario_list) + ' has been started.')

        raw_data_dir_path_dict, processed_data_dir_path_dict = self.__get_scenario_dir_path(scenario_list)
        cs_basic_feature_comb_dict, gs_basic_feature_comb_dict = \
            self.__get_basic_symbol_list(processed_data_dir_path_dict)

        attack_raw_data_dir_path = raw_data_dir_path_dict[Constant_Parameters.ATTACK]
        normal_raw_data_dir_path = raw_data_dir_path_dict[Constant_Parameters.NORMAL]
        attack_cs_pid_list = []
        normal_cs_pid_list = []
        attack_cs_id_pid_path = attack_raw_data_dir_path + '/' + Constant_Parameters.CS_ID_PID_FILENAME
        normal_cs_id_pid_path = normal_raw_data_dir_path + '/' + Constant_Parameters.CS_ID_PID_FILENAME

        attack_cs_pid_id_dict = {}
        with open(attack_cs_id_pid_path, 'r') as f:
            rdr = csv.reader(f)
            for line in rdr:
                if line[0] in cs_id_list:
                    attack_cs_pid_list.append(line[1])
                    attack_cs_pid_id_dict[line[1]] = line[0]

        normal_cs_pid_id_dict = {}
        with open(normal_cs_id_pid_path, 'r') as f:
            rdr = csv.reader(f)
            for line in rdr:
                if line[0] in cs_id_list:
                    normal_cs_pid_list.append(line[1])
                    normal_cs_pid_id_dict[line[1]] = line[0]

        attack_cs_record_path = attack_raw_data_dir_path + '/' + Constant_Parameters.CS_RECORD_PATH
        normal_cs_record_path = normal_raw_data_dir_path + '/' + Constant_Parameters.CS_RECORD_PATH
        attack_gs_record_path = attack_raw_data_dir_path + '/' + Constant_Parameters.GS_RECORD_PATH
        normal_gs_record_path = normal_raw_data_dir_path + '/' + Constant_Parameters.GS_RECORD_PATH

        cs_attack_record_file_path_list = []
        cs_normal_record_file_path_list = []
        for attack_cs_pid, normal_cs_pid in zip(attack_cs_pid_list, normal_cs_pid_list):
            attack_file_name = 'perf_record_' + attack_cs_pid + '.txt'
            normal_file_name = 'perf_record_' + normal_cs_pid + '.txt'

            full_attack_file_path = attack_cs_record_path + '/' + attack_file_name
            full_normal_file_path = normal_cs_record_path + '/' + normal_file_name

            cs_attack_record_file_path_list.append(full_attack_file_path)
            cs_normal_record_file_path_list.append(full_normal_file_path)

        gs_attack_record_file_path_list = self.__get_gs_record_file_name_list(attack_gs_record_path)
        gs_normal_record_file_path_list = self.__get_gs_record_file_name_list(normal_gs_record_path)

        gs_attack_record_dict = {}
        gs_normal_record_dict = {}
        for gs_attack_record_file_path, gs_normal_record_file_path \
                in zip(gs_attack_record_file_path_list, gs_normal_record_file_path_list):
            with open(gs_attack_record_file_path, 'r') as f:
                attack_lines = self.__get_record_list(gs_attack_record_file_path)
                gs_attack_pid = self.__extract_pid(gs_attack_record_file_path)
            with open(gs_normal_record_file_path, 'r') as f:
                normal_lines = self.__get_record_list(gs_normal_record_file_path)
                gs_normal_pid = self.__extract_pid(gs_normal_record_file_path)

            attack_cycle_dict, attack_instruction_dict, attack_branch_dict = \
                self.__get_category_record_list(attack_lines)
            normal_cycle_dict, normal_instruction_dict, normal_branch_dict = \
                self.__get_category_record_list(normal_lines)

            attack_category_dict = {Constant_Parameters.CYCLES: attack_cycle_dict,
                                    Constant_Parameters.INSTRUCTIONS: attack_instruction_dict,
                                    Constant_Parameters.BRANCH: attack_branch_dict}
            normal_category_dict = {Constant_Parameters.CYCLES: normal_cycle_dict,
                                    Constant_Parameters.INSTRUCTIONS: normal_instruction_dict,
                                    Constant_Parameters.BRANCH: normal_branch_dict}

            gs_attack_record_dict[gs_attack_pid] = attack_category_dict
            gs_normal_record_dict[gs_normal_pid] = normal_category_dict

        cs_attack_record_dict = {}
        cs_normal_record_dict = {}
        for cs_attack_record_file_path, cs_normal_record_file_path \
                in zip(cs_attack_record_file_path_list, cs_normal_record_file_path_list):
            attack_lines = self.__get_record_list(cs_attack_record_file_path)
            attack_cycle_dict, attack_instruction_dict, attack_branch_dict = \
                self.__get_category_record_list(attack_lines)
            normal_lines = self.__get_record_list(cs_normal_record_file_path)
            normal_cycle_dict, normal_instruction_dict, normal_branch_dict = \
                self.__get_category_record_list(normal_lines)

            cs_attack_pid = self.__extract_pid(cs_attack_record_file_path)
            cs_normal_pid = self.__extract_pid(cs_normal_record_file_path)
            attack_cs_id = attack_cs_pid_id_dict[cs_attack_pid]
            normal_cs_id = normal_cs_pid_id_dict[cs_normal_pid]

            attack_category_dict = {Constant_Parameters.CYCLES: attack_cycle_dict,
                                    Constant_Parameters.INSTRUCTIONS: attack_instruction_dict,
                                    Constant_Parameters.BRANCH: attack_branch_dict}
            normal_category_dict = {Constant_Parameters.CYCLES: normal_cycle_dict,
                                    Constant_Parameters.INSTRUCTIONS: normal_instruction_dict,
                                    Constant_Parameters.BRANCH: normal_branch_dict}

            cs_attack_record_dict[attack_cs_id] = attack_category_dict
            cs_normal_record_dict[normal_cs_id] = normal_category_dict

        cs_attack_ratio_dict = self.__get_symbol_ratio_and_index(cs_attack_record_dict, cs_basic_feature_comb_dict)
        cs_normal_ratio_dict = self.__get_symbol_ratio_and_index(cs_normal_record_dict, cs_basic_feature_comb_dict)
        gs_attack_ratio_dict = self.__get_symbol_ratio_and_index(gs_attack_record_dict, gs_basic_feature_comb_dict)
        gs_normal_ratio_dict = self.__get_symbol_ratio_and_index(gs_normal_record_dict, gs_basic_feature_comb_dict)

        cs_attack_elbow_point_dict = self.__find_each_station_elbow_point_dict(cs_attack_record_dict)
        cs_normal_elbow_point_dict = self.__find_each_station_elbow_point_dict(cs_normal_record_dict)
        gs_attack_elbow_point_dict = self.__find_each_station_elbow_point_dict(gs_attack_record_dict)
        gs_normal_elbow_point_dict = self.__find_each_station_elbow_point_dict(gs_normal_record_dict)

        cs_path = processed_data_dir_path_dict + '/' + Constant_Parameters.TOP + '/' + Constant_Parameters.CS
        gs_path = processed_data_dir_path_dict + '/' + Constant_Parameters.TOP + '/' + Constant_Parameters.GS

        cs_ratio_dict = {Constant_Parameters.ATTACK: cs_attack_ratio_dict,
                         Constant_Parameters.NORMAL: cs_normal_ratio_dict}
        gs_ratio_dict = {Constant_Parameters.ATTACK: gs_attack_ratio_dict,
                         Constant_Parameters.NORMAL: gs_normal_ratio_dict}

        cs_elbow_point_dict = {Constant_Parameters.ATTACK: cs_attack_elbow_point_dict,
                               Constant_Parameters.NORMAL: cs_normal_elbow_point_dict}
        gs_elbow_point_dict = {Constant_Parameters.ATTACK: gs_attack_elbow_point_dict,
                               Constant_Parameters.NORMAL: gs_normal_elbow_point_dict}

        cs_ratio_file_path = cs_path + '/' + Constant_Parameters.RATIO_INFORMATION_FILE_NAME
        gs_ratio_file_path = gs_path + '/' + Constant_Parameters.RATIO_INFORMATION_FILE_NAME
        cs_elbow_point_file_path = cs_path + '/' + Constant_Parameters.ELBOW_POINT_INFORMATION_FILE_NAME
        gs_elbow_point_file_path = gs_path + '/' + Constant_Parameters.ELBOW_POINT_INFORMATION_FILE_NAME

        cs_record_dict = {Constant_Parameters.ATTACK: cs_attack_record_dict,
                          Constant_Parameters.NORMAL: cs_normal_record_dict}
        gs_record_dict = {Constant_Parameters.ATTACK: gs_attack_record_dict,
                          Constant_Parameters.NORMAL: gs_normal_record_dict}

        attack_common_cycle_list, attack_common_branch_list, attack_common_instruction_list = \
            self.__extract_cs_common_symbols(cs_record_dict[Constant_Parameters.ATTACK])
        normal_common_cycle_list, normal_common_branch_list, normal_common_instruction_list = \
            self.__extract_cs_common_symbols(cs_record_dict[Constant_Parameters.NORMAL])

        attack_cs_cycle_symbol_dict, attack_cs_branch_symbol_dict, attack_cs_instruction_symbol_dict = \
            self.__get_average_overhead_dict(attack_common_cycle_list, attack_common_branch_list,
                                             attack_common_instruction_list, cs_record_dict[Constant_Parameters.ATTACK])
        normal_cs_cycle_symbol_dict, normal_cs_branch_symbol_dict, normal_cs_instruction_symbol_dict = \
            self.__get_average_overhead_dict(normal_common_cycle_list, normal_common_branch_list,
                                             normal_common_instruction_list, cs_record_dict[Constant_Parameters.NORMAL])

        attack_cs_symbol_dict = {Constant_Parameters.CYCLES: attack_cs_cycle_symbol_dict,
                                 Constant_Parameters.BRANCH: attack_cs_branch_symbol_dict,
                                 Constant_Parameters.INSTRUCTIONS: attack_cs_instruction_symbol_dict}
        normal_cs_symbol_dict = {Constant_Parameters.CYCLES: normal_cs_cycle_symbol_dict,
                                 Constant_Parameters.BRANCH: normal_cs_branch_symbol_dict,
                                 Constant_Parameters.INSTRUCTIONS: normal_cs_instruction_symbol_dict}

        cs_common_symbol_dict = {Constant_Parameters.ATTACK: attack_cs_symbol_dict,
                                 Constant_Parameters.NORMAL: normal_cs_symbol_dict}

        cs_avg_elbow_point_dict = self.__find_avg_cs_elbow_point_dict(cs_common_symbol_dict)
        cs_avg_elbow_point_file_path = cs_path + '/' + Constant_Parameters.CS_AVG_ELBOW_POINT_FILENAME
        with open(cs_avg_elbow_point_file_path, 'w') as f:
            json.dump(cs_avg_elbow_point_dict, f)

        cs_common_symbol_file_path = cs_path + '/' + Constant_Parameters.CS_AVG_OVERHEAD_FILENAME
        with open(cs_common_symbol_file_path, 'w') as f:
            json.dump(cs_common_symbol_dict, f)

        cs_record_analysis_file_path = cs_path + '/' + Constant_Parameters.KERNEL_RECORD_FILENAME
        gs_record_analysis_file_path = gs_path + '/' + Constant_Parameters.KERNEL_RECORD_FILENAME

        with open(cs_record_analysis_file_path, 'w') as f:
            json.dump(cs_record_dict, f)
        with open(gs_record_analysis_file_path, 'w') as f:
            json.dump(gs_record_dict, f)
        with open(cs_ratio_file_path, 'w') as f:
            json.dump(cs_ratio_dict, f)
        with open(gs_ratio_file_path, 'w') as f:
            json.dump(gs_ratio_dict, f)
        with open(cs_elbow_point_file_path, 'w') as f:
            json.dump(cs_elbow_point_dict, f)
        with open(gs_elbow_point_file_path, 'w') as f:
            json.dump(gs_elbow_point_dict, f)

    @classmethod
    def calculate_total_elbow_points(cls):
        print('BEST K saved.')
        total_cs_elbow_point_dict = {}
        total_gs_elbow_point_dict = {}
        for scenario, root_dir_path in Constant_Parameters.PROCESSED_DATASET_PATH_DICT.items():
            cs_dir_path = root_dir_path + '/' + Constant_Parameters.TOP + '/' + Constant_Parameters.CS
            gs_dir_path = root_dir_path + '/' + Constant_Parameters.TOP + '/' + Constant_Parameters.GS

            cs_file_path = cs_dir_path + '/' + Constant_Parameters.CS_AVG_ELBOW_POINT_FILENAME
            gs_file_path = gs_dir_path + '/' + Constant_Parameters.ELBOW_POINT_INFORMATION_FILE_NAME

            with open(cs_file_path, 'r') as f:
                cs_elbow_point_dict = json.load(f)
            with open(gs_file_path, 'r') as f:
                gs_elbow_point_dict = json.load(f)

            attack_cs_elbow_point_dict = cs_elbow_point_dict[Constant_Parameters.ATTACK]
            normal_cs_elbow_point_dict = cs_elbow_point_dict[Constant_Parameters.NORMAL]
            attack_gs_elbow_point_dict = gs_elbow_point_dict[Constant_Parameters.ATTACK].popitem()[1]
            normal_gs_elbow_point_dict = gs_elbow_point_dict[Constant_Parameters.NORMAL].popitem()[1]

            param_cs_dict = {Constant_Parameters.ATTACK: attack_cs_elbow_point_dict,
                             Constant_Parameters.NORMAL: normal_cs_elbow_point_dict}
            param_gs_dict = {Constant_Parameters.ATTACK: attack_gs_elbow_point_dict,
                             Constant_Parameters.NORMAL: normal_gs_elbow_point_dict}

            total_cs_elbow_point_dict[scenario] = param_cs_dict
            total_gs_elbow_point_dict[scenario] = param_gs_dict

        cs_save_path = Constant_Parameters.RESULT + '/' + Constant_Parameters.BEST_K + '/' \
                       + Constant_Parameters.CS_BEST_FILENAME
        gs_save_path = Constant_Parameters.RESULT + '/' + Constant_Parameters.BEST_K + '/' \
                       + Constant_Parameters.GS_BEST_FILENAME

        with open(cs_save_path, 'w') as f:
            json.dump(total_cs_elbow_point_dict, f)
        with open(gs_save_path, 'w') as f:
            json.dump(total_gs_elbow_point_dict, f)

    @classmethod
    def __get_overhead_ratio_and_index(cls, data_dict: dict):
        param_3_dict = {}
        for category_name, type_dict in data_dict.items():
            param_2_dict = {}
            for type_name, symbol_dict in type_dict.items():
                param_1_dict = {}
                for symbol_name, temp_dict in symbol_dict.items():
                    overhead_ratio_mean = temp_dict[Constant_Parameters.OVERHEAD_RATIO_MEAN]
                    symbol_index_mean = temp_dict[Constant_Parameters.SYMBOL_INDEX_MEAN]
                    param_1_dict[symbol_name] = {Constant_Parameters.OVERHEAD_RATIO_MEAN: overhead_ratio_mean,
                                                 Constant_Parameters.SYMBOL_INDEX_MEAN: symbol_index_mean}
                param_2_dict[type_name] = param_1_dict
            param_3_dict[category_name] = param_2_dict

        return param_3_dict

    @classmethod
    def extract_overhead_ratio_and_index(cls):
        print('Overhead and Index saved.')

        cs_dict = {}
        gs_dict = {}
        for scenario, root_dir_path in Constant_Parameters.PROCESSED_DATASET_PATH_DICT.items():
            cs_dir_path = root_dir_path + '/' + Constant_Parameters.TOP + '/' + Constant_Parameters.CS
            gs_dir_path = root_dir_path + '/' + Constant_Parameters.TOP + '/' + Constant_Parameters.GS
            cs_file_path = cs_dir_path + '/' + Constant_Parameters.RATIO_INFORMATION_FILE_NAME
            gs_file_path = gs_dir_path + '/' + Constant_Parameters.RATIO_INFORMATION_FILE_NAME

            with open(cs_file_path, 'r') as f:
                cs_ratio_dict = json.load(f)
            with open(gs_file_path, 'r') as f:
                gs_ratio_dict = json.load(f)

            cs_attack_dict = cs_ratio_dict[Constant_Parameters.ATTACK]
            cs_normal_dict = cs_ratio_dict[Constant_Parameters.NORMAL]
            cs_attack_overhead_dict = cls.__get_overhead_ratio_and_index(cs_attack_dict)
            cs_normal_overhead_dict = cls.__get_overhead_ratio_and_index(cs_normal_dict)

            gs_attack_dict = gs_ratio_dict[Constant_Parameters.ATTACK]
            gs_normal_dict = gs_ratio_dict[Constant_Parameters.NORMAL]
            gs_attack_overhead_dict = cls.__get_overhead_ratio_and_index(gs_attack_dict)
            gs_normal_overhead_dict = cls.__get_overhead_ratio_and_index(gs_normal_dict)

            cs_dict[scenario] = {Constant_Parameters.ATTACK: cs_attack_overhead_dict,
                                 Constant_Parameters.NORMAL: cs_normal_overhead_dict}
            gs_dict[scenario] = {Constant_Parameters.ATTACK: gs_attack_overhead_dict,
                                 Constant_Parameters.NORMAL: gs_normal_overhead_dict}

        root_save_dir_path = Constant_Parameters.RESULT + '/' + Constant_Parameters.OVERHEAD_INDEX
        cs_save_path = root_save_dir_path + '/' + Constant_Parameters.CS_OVERHEAD_INDEX_FILENAME
        gs_save_path = root_save_dir_path + '/' + Constant_Parameters.GS_OVERHEAD_INDEX_FILENAME

        with open(cs_save_path, 'w') as f:
            json.dump(cs_dict, f)
        with open(gs_save_path, 'w') as f:
            json.dump(gs_dict, f)