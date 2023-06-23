import csv
import json
import numpy as np
import Constant_Parameters


class Data_Extraction:
    __cs_station_id_dict: dict
    __gs_station_id_dict: dict

    __cs_attack_mode_dict: dict
    __cs_normal_mode_dict: dict

    __simulation_time_dict: dict

    def __init__(self):
        self.__loading_station_id()
        self.__loading_charging_session(self.__cs_station_id_dict)
        self.__loading_sim_time(self.__gs_station_id_dict)

    def __loading_charging_session(self, cs_station_id_dict):
        self.__cs_attack_mode_dict = self.__get_charging_session_dict(cs_station_id_dict, Constant_Parameters.ATTACK)
        self.__cs_normal_mode_dict = self.__get_charging_session_dict(cs_station_id_dict, Constant_Parameters.NORMAL)

    def __loading_sim_time(self, gs_station_id_dict):
        scenario_abbreviation_list = list(gs_station_id_dict.keys())

        sim_time_dict = {}
        sim_time_diff_list = []
        for scenario in scenario_abbreviation_list:
            attack_dir_path = Constant_Parameters.RAW_DATA_DICT[scenario][Constant_Parameters.ATTACK]
            normal_dir_path = Constant_Parameters.RAW_DATA_DICT[scenario][Constant_Parameters.NORMAL]
            attack_sim_file_path = attack_dir_path + '/' + Constant_Parameters.SIM_SEC_ATTACK_FILENAME
            normal_sim_file_path = normal_dir_path + '/' + Constant_Parameters.SIM_SEC_NORMAL_FILENAME

            with open(attack_sim_file_path, 'r') as f:
                temp_attack_sim_time = f.read()
                temp_list = temp_attack_sim_time.split(' ')
                attack_sim_time = float(temp_list[0])

            with open(normal_sim_file_path, 'r') as f:
                temp_normal_sim_time = f.read()
                temp_list = temp_normal_sim_time.split(' ')
                normal_sim_time = float(temp_list[0])

            sim_time_diff = attack_sim_time - normal_sim_time
            sim_time_diff = abs(sim_time_diff)
            sim_time_diff_list.append(sim_time_diff)

            sim_time_dict[scenario] = {Constant_Parameters.ATTACK: attack_sim_time,
                                       Constant_Parameters.NORMAL: normal_sim_time,
                                       Constant_Parameters.TIME_DIFF: sim_time_diff}

        self.__simulation_time_dict = {Constant_Parameters.SIMULATION_TIME: sim_time_dict,
                                       Constant_Parameters.SIMULATION_TIME_DIFF_AVG: np.mean(sim_time_diff_list)}

    def saving_sim_time(self):
        save_dir_path = Constant_Parameters.EXPERIMENT_DATA + '/' + Constant_Parameters.SIMULATION_TIME
        save_file_path = save_dir_path + '/' + Constant_Parameters.SIMULATION_TIME + '.json'

        with open(save_file_path, 'w') as f:
            json.dump(self.__simulation_time_dict, f)

    def saving_charging_session(self):
        dir_path = Constant_Parameters.EXPERIMENT_DATA + '/' + Constant_Parameters.CHARGING_SESSION
        attack_mode_path = dir_path + '/' + Constant_Parameters.ATTACK + '.json'
        normal_mode_path = dir_path + '/' + Constant_Parameters.NORMAL + '.json'

        with open(attack_mode_path, 'w') as f:
            json.dump(self.__cs_attack_mode_dict, f)
        with open(normal_mode_path, 'w') as f:
            json.dump(self.__cs_normal_mode_dict, f)

    @classmethod
    def __get_charging_session_dict(cls, station_id_dict, category):
        cs_id_index = 1
        session_index = 0
        type_index = 5

        scenario_dict = {}
        for scenario, station_id_list in station_id_dict.items():
            root_path = Constant_Parameters.RAW_DATA_DICT[scenario][category]
            file_path = root_path + '/' + Constant_Parameters.AUTHENTICATION_RESULTS + '.csv'

            record_list = []
            with open(file_path, 'r') as f:
                rdr = csv.reader(f)
                for line in rdr:
                    record_list.append(line)

            station_session_dict = {}
            for station_id in station_id_list:
                attack_session_list = []
                normal_session_list = []
                for record in record_list:
                    cs_id = record[cs_id_index]
                    if station_id == cs_id:
                        type_category = record[type_index]
                        if type_category == Constant_Parameters.ATTACK:
                            attack_session_list.append(record[session_index])
                        elif type_category == Constant_Parameters.NORMAL:
                            normal_session_list.append(record[session_index])

                attack_unique_session_list = list(set(attack_session_list))
                normal_unique_session_list = list(set(normal_session_list))

                attack_ev_number = len(attack_unique_session_list)
                normal_ev_number = len(normal_unique_session_list)
                total_attack_count = len(attack_session_list)
                total_normal_count = len(normal_session_list)
                total_gs_auth_count = total_attack_count + total_normal_count

                station_session_dict[station_id] = {Constant_Parameters.ATTACK_EV_NUMBER: attack_ev_number,
                                                    Constant_Parameters.NORMAL_EV_NUMBER: normal_ev_number,
                                                    Constant_Parameters.TOTAL_ATTACK_COUNT: total_attack_count,
                                                    Constant_Parameters.TOTAL_NORMAL_COUNT: total_normal_count,
                                                    Constant_Parameters.TOTAL_GS_AUTH_COUNT: total_gs_auth_count}

            scenario_dict[scenario] = station_session_dict

        return scenario_dict

    def __loading_station_id(self):
        stat_cs_path = \
            Constant_Parameters.STAT + '/' + Constant_Parameters.CS + '/' + Constant_Parameters.STAT + '.json'
        stat_gs_path = \
            Constant_Parameters.STAT + '/' + Constant_Parameters.GS + '/' + Constant_Parameters.STAT + '.json'
        top_cs_path = \
            Constant_Parameters.TOP + '/' + Constant_Parameters.CS + '/' + Constant_Parameters.TOP + '.json'
        top_gs_path = \
            Constant_Parameters.TOP + '/' + Constant_Parameters.GS + '/' + Constant_Parameters.TOP + '.json'

        self.__cs_station_id_dict = self.__get_unique_and_common_station_id_list(stat_cs_path, top_cs_path)
        self.__gs_station_id_dict = self.__get_unique_and_common_station_id_list(stat_gs_path, top_gs_path)

    def saving_station_id(self):
        default_path = Constant_Parameters.EXPERIMENT_DATA + '/' + Constant_Parameters.STATION_ID + '/'
        cs_path = default_path + '/cs_station_id.json'
        gs_path = default_path + '/gs_station_id.json'

        with open(cs_path, 'w') as f:
            json.dump(self.__cs_station_id_dict, f)
        with open(gs_path, 'w') as f:
            json.dump(self.__gs_station_id_dict, f)

    @classmethod
    def __get_unique_and_common_station_id_list(cls, stat_path, top_path):
        station_dict = {}
        for scenario, default_path in Constant_Parameters.PROCESSED_DATASET_PATH_DICT.items():
            stat_cs_path = default_path + '/' + stat_path
            top_cs_path = default_path + '/' + top_path

            stat_station_id_list = cls.__get_station_id_list(stat_cs_path)
            top_station_id_list = cls.__get_station_id_list(top_cs_path)

            station_id_list = cls.__get_common_station_id_list(stat_station_id_list, top_station_id_list)

            station_dict[scenario] = station_id_list

        return station_dict

    @classmethod
    def __get_common_station_id_list(cls, stat_station_id_list, top_station_id_list):
        common_station_id_set = set(stat_station_id_list) & set(top_station_id_list)
        return list(common_station_id_set)

    @classmethod
    def __get_station_id_list(cls, file_path):
        with open(file_path, 'r') as f:
            temp_dict = json.load(f)

        station_id_list = list(station_id for station_id in temp_dict.keys())

        return station_id_list