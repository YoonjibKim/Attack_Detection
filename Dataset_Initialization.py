import csv
import os
import Constant_Parameters


class Dataset_Initialization:
    __combined_cs_stat_dict: dict
    __combined_cs_top_dict: dict
    __combined_gs_stat_dict: dict
    __combined_gs_top_dict: dict
    __combined_gs_time_diff_dict: dict

    __attack_sim_time: float
    __normal_sim_time: float

    __top_item_list: list

    def __init__(self, raw_attack_root_path, raw_normal_root_path):
        self.__raw_attack_root_path = raw_attack_root_path
        self.__raw_normal_root_path = raw_normal_root_path

        self.__top_item_list = Constant_Parameters.TYPE_LIST

        stat_cs_pid_list, top_cs_pid_list, stat_gs_pid_list, top_gs_pid_list = \
            self.__check_valid_cs_gs(self.__raw_attack_root_path, self.__raw_normal_root_path)
        self.__load_file_list(stat_cs_pid_list, top_cs_pid_list, stat_gs_pid_list, top_gs_pid_list)

        attack_stat_cs_dict = self.__get_stat_cs_id_full_file_path_dict(self.__attack_stat_cs_full_file_path_list,
                                                                        self.__attack_cs_id_pid_list)
        normal_stat_cs_dict = self.__get_stat_cs_id_full_file_path_dict(self.__normal_stat_cs_full_file_path_list,
                                                                        self.__normal_cs_id_pid_list)

        attack_top_cs_dict = self.__get_top_cs_id_full_file_path_dict(self.__attack_top_cs_full_file_path_list,
                                                                      self.__attack_cs_id_pid_list)
        normal_top_cs_dict = self.__get_top_cs_id_full_file_path_dict(self.__normal_top_cs_full_file_path_list,
                                                                      self.__normal_cs_id_pid_list)

        attack_stat_gs_dict = self.__get_stat_gs_id_full_file_path_dict(self.__attack_stat_gs_full_file_path_list)
        normal_stat_gs_dict = self.__get_stat_gs_id_full_file_path_dict(self.__normal_stat_gs_full_file_path_list)

        attack_top_gs_dict = self.__get_top_gs_id_full_file_path_dict(self.__attack_top_gs_full_file_path_list)
        normal_top_gs_dict = self.__get_top_gs_id_full_file_path_dict(self.__normal_top_gs_full_file_path_list)

        attack_time_diff_gs_dict = self.__get_time_diff_gs_id_full_file_path_dict(self.__attack_time_diff_path)
        normal_time_diff_gs_dict = self.__get_time_diff_gs_id_full_file_path_dict(self.__normal_time_diff_path)

        self.__combined_cs_stat_dict = self.__combine_file_path_dict(attack_stat_cs_dict, normal_stat_cs_dict)
        self.__combined_cs_top_dict = self.__combine_file_path_dict(attack_top_cs_dict, normal_top_cs_dict)
        self.__combined_gs_stat_dict = self.__combine_file_path_dict(attack_stat_gs_dict, normal_stat_gs_dict)
        self.__combined_gs_top_dict = self.__combine_file_path_dict(attack_top_gs_dict, normal_top_gs_dict)
        self.__combined_gs_time_diff_dict = self.__combine_file_path_dict(attack_time_diff_gs_dict,
                                                                          normal_time_diff_gs_dict)

        self.__attack_sim_time, self.__normal_sim_time = \
            self.__get_passed_simulation_time(raw_attack_root_path, raw_normal_root_path)

    def get_scenario_list(self):
        attack_path: str = self.__raw_attack_root_path
        normal_path: str = self.__raw_normal_root_path

        attack_path_list = attack_path.split('/')
        normal_path_list = normal_path.split('/')

        attack_list = attack_path_list[2:-1]
        normal_list = normal_path_list[2:-1]

        common_path_list = []
        for temp_attack_path, temp_normal_path in zip(attack_list, normal_list):
            if temp_attack_path == temp_normal_path:
                common_path_list.append(temp_attack_path)
            else:
                common_path_list.append(None)

        return common_path_list

    def get_type_list(self):
        return self.__top_item_list

    def get_attack_sim_time(self):
        return self.__attack_sim_time

    def get_normal_sim_time(self):
        return self.__normal_sim_time

    def get_cs_stat_file_path_dict(self):
        return self.__combined_cs_stat_dict

    def get_cs_top_file_path_dict(self):
        return self.__combined_cs_top_dict

    def get_gs_stat_file_path_dict(self):
        return self.__combined_gs_stat_dict

    def get_gs_top_file_path_dict(self):
        return self.__combined_gs_top_dict

    def get_gs_time_diff_file_path_dict(self):
        return self.__combined_gs_time_diff_dict

    @classmethod
    def __get_passed_simulation_time(cls, attack_root_path, normal_root_path):
        attack_sim_file_path = attack_root_path + '/' + Constant_Parameters.SIM_SEC_ATTACK_FILENAME
        normal_sim_file_path = normal_root_path + '/' + Constant_Parameters.SIM_SEC_NORMAL_FILENAME

        with open(attack_sim_file_path, 'r') as f:
            attack_temp_line = f.readline()

        attack_temp_list = attack_temp_line.split()
        attack_sim_time = float(attack_temp_list[0])

        with open(normal_sim_file_path, 'r') as f:
            normal_temp_line = f.readline()

        normal_temp_list = normal_temp_line.split()
        normal_sim_time = float(normal_temp_list[0])

        return attack_sim_time, normal_sim_time

    @classmethod
    def __combine_file_path_dict(cls, attack_stat_dict, normal_stat_dict):
        attack_id_set = set(attack_stat_dict.keys())
        normal_id_set = set(normal_stat_dict.keys())
        unique_id_list = attack_id_set & normal_id_set

        combined_stat_dict = {}
        for unique_id in unique_id_list:
            attack_list = [Constant_Parameters.ATTACK, attack_stat_dict[unique_id]]
            normal_list = [Constant_Parameters.NORMAL, normal_stat_dict[unique_id]]
            combined_stat_dict[unique_id] = [attack_list, normal_list]

        return combined_stat_dict

    @classmethod
    def __get_time_diff_gs_id_full_file_path_dict(cls, time_diff_path):
        gs_dict = {Constant_Parameters.GS_ID: time_diff_path}
        return gs_dict

    @classmethod
    def __get_top_gs_id_full_file_path_dict(cls, gs_full_file_path_list):
        top_list = []
        for gs_full_file_path in gs_full_file_path_list:
            temp_list = gs_full_file_path.split('_')
            top_type = temp_list[len(temp_list) - 2]
            top_list.append([top_type, gs_full_file_path])

        gs_dict = {Constant_Parameters.GS_ID: top_list}
        return gs_dict

    @classmethod
    def __get_stat_gs_id_full_file_path_dict(cls, gs_full_file_path_list):
        gs_dict = {Constant_Parameters.GS_ID: gs_full_file_path_list[0]}
        return gs_dict

    @classmethod
    def __get_top_cs_id_full_file_path_dict(cls, cs_full_file_path_list, cs_id_pid_list):
        cs_top_list = []
        cs_id_list = []
        for cs_full_file_path in cs_full_file_path_list:
            temp_list = cs_full_file_path.split('_')
            top_type = temp_list[len(temp_list) - 2]
            cs_pid = temp_list[len(temp_list) - 1].replace('.txt', '')
            for cs_id_pid in cs_id_pid_list:
                found_cs_pid = cs_id_pid[1]
                if cs_pid == found_cs_pid:
                    found_cs_id = cs_id_pid[0]
                    cs_top_list.append([found_cs_id, top_type, cs_full_file_path])
                    cs_id_list.append(found_cs_id)

        cs_top_dict = {}
        unique_cs_id_list = list(set(cs_id_list))
        for unique_cs_id in unique_cs_id_list:
            temp_list = []
            for cs_top in cs_top_list:
                found_cs_id = cs_top[0]
                if unique_cs_id == found_cs_id:
                    temp_list.append(cs_top[1:])

            cs_top_dict[unique_cs_id] = temp_list

        return cs_top_dict

    @classmethod
    def __get_stat_cs_id_full_file_path_dict(cls, cs_full_file_path_list, cs_id_pid_list):
        cs_dict = {}
        for cs_full_file_path in cs_full_file_path_list:
            temp_list = cs_full_file_path.split('_')
            temp_list = temp_list[len(temp_list) - 1].split('.')
            cs_pid = temp_list[0]
            for cs_id_pid in cs_id_pid_list:
                found_cs_pid = cs_id_pid[1]
                if cs_pid == found_cs_pid:
                    found_cs_id = cs_id_pid[0]
                    cs_dict[found_cs_id] = cs_full_file_path
                    break

        return cs_dict

    @classmethod
    def __get_valid_filename_list(cls, attack_dir_path, normal_dir_path, pid_list):
        attack_pid_list = pid_list[0]
        normal_pid_list = pid_list[1]

        attack_file_list = []
        cs_gs_list = os.listdir(attack_dir_path)
        for pid in attack_pid_list:
            for cs_gs in cs_gs_list:
                temp_list = cs_gs.split('_')
                temp_list = temp_list[len(temp_list) - 1].split('.')
                found_pid = temp_list[0]
                if found_pid == pid:
                    attack_file_list.append(cs_gs)

        normal_file_list = []
        cs_gs_list = os.listdir(normal_dir_path)
        for pid in normal_pid_list:
            for cs_gs in cs_gs_list:
                temp_list = cs_gs.split('_')
                temp_list = temp_list[len(temp_list) - 1].split('.')
                found_pid = temp_list[0]
                if found_pid == pid:
                    normal_file_list.append(cs_gs)

        return attack_file_list, normal_file_list

    def __load_file_list(self, stat_cs_pid_list, top_cs_pid_list, stat_gs_pid_list, top_gs_pid_list):
        attack_stat_cs_file_list, normal_stat_cs_file_list = \
            self.__get_valid_filename_list(self.__cs_attack_stat_path, self.__cs_normal_stat_path, stat_cs_pid_list)

        attack_top_cs_file_list, normal_top_cs_file_list = \
            self.__get_valid_filename_list(self.__cs_attack_top_path, self.__cs_normal_top_path, top_cs_pid_list)

        attack_stat_gs_file_list, normal_stat_gs_file_list = \
            self.__get_valid_filename_list(self.__gs_attack_stat_path, self.__gs_normal_stat_path, stat_gs_pid_list)

        attack_top_gs_file_list, normal_top_gs_file_list = \
            self.__get_valid_filename_list(self.__gs_attack_top_path, self.__gs_normal_top_path, top_gs_pid_list)

        self.__attack_stat_cs_full_file_path_list, self.__normal_stat_cs_full_file_path_list = \
            self.__get_full_file_path(attack_stat_cs_file_list, normal_stat_cs_file_list,
                                      self.__cs_attack_stat_path, self.__cs_normal_stat_path)

        self.__attack_top_cs_full_file_path_list, self.__normal_top_cs_full_file_path_list = \
            self.__get_full_file_path(attack_top_cs_file_list, normal_top_cs_file_list,
                                      self.__cs_attack_top_path, self.__cs_normal_top_path)

        self.__attack_stat_gs_full_file_path_list, self.__normal_stat_gs_full_file_path_list = \
            self.__get_full_file_path(attack_stat_gs_file_list, normal_stat_gs_file_list,
                                      self.__gs_attack_stat_path, self.__gs_normal_stat_path)

        self.__attack_top_gs_full_file_path_list, self.__normal_top_gs_full_file_path_list = \
            self.__get_full_file_path(attack_top_gs_file_list, normal_top_gs_file_list,
                                      self.__gs_attack_top_path, self.__gs_normal_top_path)

    @classmethod
    def __get_full_file_path(cls, attack_file_list, normal_file_list, attack_root_path, normal_root_path):
        attack_full_file_path_list = []
        for attack_file in attack_file_list:
            full_path = attack_root_path + '/' + attack_file
            attack_full_file_path_list.append(full_path)

        normal_full_file_path_list = []
        for normal_file in normal_file_list:
            full_path = normal_root_path + '/' + normal_file
            normal_full_file_path_list.append(full_path)

        return attack_full_file_path_list, normal_full_file_path_list

    @classmethod
    def __get_cs_id_pid_list(cls, dir_path):
        cs_id_pid_list = []
        with open(dir_path, 'r') as f:
            rdr = csv.reader(f)
            for line in rdr:
                cs_id_pid_list.append(line)

        cs_id_pid_list = cs_id_pid_list[1:]

        return cs_id_pid_list

    @classmethod
    def __get_valid_stat_cs_id_list(cls, dir_path, cs_id_pid_list):
        cs_list = os.listdir(dir_path)
        found_cs_pid_list = []
        for cs in cs_list:
            temp_list = cs.split('_')
            temp_list = temp_list[len(temp_list) - 1].split('.')
            found_cs_pid_list.append(temp_list[0])

        found_cs_id_list = []
        for found_cs_pid in found_cs_pid_list:
            for cs_id_pid in cs_id_pid_list:
                cs_pid = cs_id_pid[1]
                if cs_pid == found_cs_pid:
                    cs_id = cs_id_pid[0]
                    found_cs_id_list.append(cs_id)

        return found_cs_id_list

    def __get_valid_top_cs_id_list(self, dir_path, cs_id_pid_list):
        cs_list = os.listdir(dir_path)
        top_type_and_cs_pid_list = []
        cs_pid_list = []
        for cs in cs_list:
            temp_list = cs.split('_')
            top_type = temp_list[2]
            temp_list = temp_list[len(temp_list) - 1].split('.')
            cs_pid = temp_list[0]
            cs_pid_list.append(cs_pid)
            top_type_and_cs_pid_list.append([top_type, cs_pid])

        unique_cs_pid_list = list(set(cs_pid_list))

        unique_cs_pid_top_type_list = []
        for unique_cs_pid in unique_cs_pid_list:
            type_list = []
            for top_type_and_cs_pid in top_type_and_cs_pid_list:
                cs_pid = top_type_and_cs_pid[1]
                if unique_cs_pid == cs_pid:
                    top_type = top_type_and_cs_pid[0]
                    type_list.append(top_type)

            unique_type_list = list(set(type_list))
            unique_cs_pid_top_type_list.append([unique_cs_pid, unique_type_list])

        top_item_size = len(self.__top_item_list)
        valid_cs_pid_list = []
        for unique_cs_pid_top_type in unique_cs_pid_top_type_list:
            top_type_list = unique_cs_pid_top_type[1]
            top_type_size = len(top_type_list)
            if top_type_size == top_item_size:
                unique_cs_pid = unique_cs_pid_top_type[0]
                valid_cs_pid_list.append(unique_cs_pid)

        valid_cs_id_list = []
        for valid_cs_pid in valid_cs_pid_list:
            for cs_id_pid in cs_id_pid_list:
                cs_pid = cs_id_pid[1]
                if valid_cs_pid == cs_pid:
                    valid_cs_id = cs_id_pid[0]
                    valid_cs_id_list.append(valid_cs_id)

        return valid_cs_id_list

    @classmethod
    def __get_valid_stat_gs_pid_list(cls, dir_path):
        gs_list = os.listdir(dir_path)
        unique_gs_list = list(set(gs_list))
        unique_gs_pid_list = []
        for unique_gs in unique_gs_list:
            temp_list = unique_gs.split('_')
            temp_list = temp_list[len(temp_list) - 1].split('.')
            gs_pid = temp_list[0]
            unique_gs_pid_list.append(gs_pid)

        return list(set(unique_gs_pid_list))

    def __get_valid_top_gs_pid(self, dir_path):
        unique_gs_pid_list_1 = self.__get_valid_stat_gs_pid_list(dir_path)
        gs_list = os.listdir(dir_path)
        top_type_size = len(self.__top_item_list)

        unique_gs_pid_list_2 = []
        for unique_gs_pid in unique_gs_pid_list_1:
            gs_type_list = []
            for gs in gs_list:
                temp_list = gs.split('_')
                top_type = temp_list[2]
                temp_list = temp_list[len(temp_list) - 1].split('.')
                gs_pid = temp_list[0]
                if gs_pid == unique_gs_pid:
                    gs_type_list.append(top_type)

            unique_gs_type_list = list(set(gs_type_list))
            unique_gs_type_size = len(unique_gs_type_list)
            if top_type_size == unique_gs_type_size:
                unique_gs_pid_list_2.append(unique_gs_pid)

        return unique_gs_pid_list_2

    @classmethod
    def __get_valid_cs_pid_list(cls, attack_top_valid_cs_id_list, normal_stat_valid_cs_id_list,
                                attack_cs_id_pid_list, normal_cs_id_pid_list):
        top_valid_cs_id_set = set(attack_top_valid_cs_id_list) & set(normal_stat_valid_cs_id_list)
        top_valid_cs_id_list = list(top_valid_cs_id_set)
        attack_pid_list = []
        normal_pid_list = []
        for top_valid_cs_id in top_valid_cs_id_list:
            for attack_cs_id_pid in attack_cs_id_pid_list:
                attack_cs_id = attack_cs_id_pid[0]
                if top_valid_cs_id == attack_cs_id:
                    attack_pid = attack_cs_id_pid[1]
                    attack_pid_list.append(attack_pid)

            for normal_cs_id_pid in normal_cs_id_pid_list:
                normal_cs_id = normal_cs_id_pid[0]
                if top_valid_cs_id == normal_cs_id:
                    normal_pid = normal_cs_id_pid[1]
                    normal_pid_list.append(normal_pid)

        return attack_pid_list, normal_pid_list

    @classmethod
    def __get_valid_gs_pid_list(cls, stat_valid_gs_pid_list, top_valid_gs_pid_list):
        valid_gs_pid_set = set(stat_valid_gs_pid_list) & set(top_valid_gs_pid_list)
        valid_gs_pid_list = list(valid_gs_pid_set)

        return valid_gs_pid_list

    def __check_valid_cs_gs(self, raw_attack_root_path, raw_normal_root_path):
        self.__cs_attack_stat_path = raw_attack_root_path + '/' + Constant_Parameters.CS_STAT_PATH
        self.__cs_attack_top_path = raw_attack_root_path + '/' + Constant_Parameters.CS_TOP_PATH
        self.__gs_attack_stat_path = raw_attack_root_path + '/' + Constant_Parameters.GS_STAT_PATH
        self.__gs_attack_top_path = raw_attack_root_path + '/' + Constant_Parameters.GS_TOP_PATH
        self.__attack_time_diff_path = raw_attack_root_path + '/' + Constant_Parameters.ATTACK_TIME_DIFF_FILENAME

        self.__cs_normal_stat_path = raw_normal_root_path + '/' + Constant_Parameters.CS_STAT_PATH
        self.__cs_normal_top_path = raw_normal_root_path + '/' + Constant_Parameters.CS_TOP_PATH
        self.__gs_normal_stat_path = raw_normal_root_path + '/' + Constant_Parameters.GS_STAT_PATH
        self.__gs_normal_top_path = raw_normal_root_path + '/' + Constant_Parameters.GS_TOP_PATH
        self.__normal_time_diff_path = raw_normal_root_path + '/' + Constant_Parameters.NORMAL_TIME_DIFF_FILENAME

        attack_cs_id_pid_path = raw_attack_root_path + '/' + Constant_Parameters.CS_ID_PID_FILENAME
        normal_cs_id_pid_path = raw_normal_root_path + '/' + Constant_Parameters.CS_ID_PID_FILENAME
        self.__attack_cs_id_pid_list = self.__get_cs_id_pid_list(attack_cs_id_pid_path)
        self.__normal_cs_id_pid_list = self.__get_cs_id_pid_list(normal_cs_id_pid_path)

        attack_stat_valid_cs_id_list = \
            self.__get_valid_stat_cs_id_list(self.__cs_attack_stat_path, self.__attack_cs_id_pid_list)
        normal_stat_valid_cs_id_list = \
            self.__get_valid_stat_cs_id_list(self.__cs_normal_stat_path, self.__normal_cs_id_pid_list)
        attack_stat_cs_pid_list, normal_stat_cs_pid_list = \
            self.__get_valid_cs_pid_list(attack_stat_valid_cs_id_list, normal_stat_valid_cs_id_list,
                                         self.__attack_cs_id_pid_list, self.__normal_cs_id_pid_list)
        attack_stat_cs_pid_size = len(attack_stat_cs_pid_list)
        normal_stat_cs_pid_size = len(normal_stat_cs_pid_list)
        if attack_stat_cs_pid_size != normal_stat_cs_pid_size:
            if attack_stat_cs_pid_size < 1 or normal_stat_cs_pid_size < 1:
                print('CS STAT dataset is wrong.')
                exit(-1)

        attack_top_valid_cs_id_list = \
            self.__get_valid_top_cs_id_list(self.__cs_attack_top_path, self.__attack_cs_id_pid_list)
        normal_top_valid_cs_id_list = \
            self.__get_valid_top_cs_id_list(self.__cs_normal_top_path, self.__normal_cs_id_pid_list)
        attack_top_cs_pid_list, normal_top_cs_pid_list = \
            self.__get_valid_cs_pid_list(attack_top_valid_cs_id_list, normal_top_valid_cs_id_list,
                                         self.__attack_cs_id_pid_list, self.__normal_cs_id_pid_list)
        attack_top_cs_pid_size = len(attack_top_cs_pid_list)
        normal_top_cs_pid_size = len(normal_top_cs_pid_list)
        if attack_top_cs_pid_size != normal_top_cs_pid_size:
            if attack_top_cs_pid_size < 1 or normal_top_cs_pid_size < 1:
                print('CS TOP dataset is wrong.')
                exit(-1)

        attack_stat_valid_gs_pid_list = self.__get_valid_stat_gs_pid_list(self.__gs_attack_stat_path)
        attack_top_valid_gs_pid_list = self.__get_valid_top_gs_pid(self.__gs_attack_top_path)
        attack_gs_pid_list = self.__get_valid_gs_pid_list(attack_stat_valid_gs_pid_list, attack_top_valid_gs_pid_list)
        attack_gs_pid_size = len(attack_gs_pid_list)

        normal_stat_valid_gs_pid_list = self.__get_valid_stat_gs_pid_list(self.__gs_normal_stat_path)
        normal_top_valid_gs_pid_list = self.__get_valid_top_gs_pid(self.__gs_normal_top_path)
        normal_gs_pid_list = self.__get_valid_gs_pid_list(normal_stat_valid_gs_pid_list, normal_top_valid_gs_pid_list)
        normal_gs_pid_size = len(normal_gs_pid_list)

        attack_gs_pid = None
        normal_gs_pid = None
        if attack_gs_pid_size < 1 or normal_gs_pid_size < 1:
            print('GS STAT or TOP dataset is wrong.')
            exit(-1)
        else:
            attack_gs_pid = attack_gs_pid_list[0]
            normal_gs_pid = normal_gs_pid_list[0]

        stat_cs_pid_list = [attack_stat_cs_pid_list, normal_stat_cs_pid_list]
        top_cs_pid_list = [attack_top_cs_pid_list, normal_top_cs_pid_list]
        stat_gs_pid_list = [[attack_gs_pid], [normal_gs_pid]]
        top_gs_pid_list = stat_gs_pid_list

        return stat_cs_pid_list, top_cs_pid_list, stat_gs_pid_list, top_gs_pid_list
