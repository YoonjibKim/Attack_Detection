import Constant_Parameters


class Time_Diff_Parser:
    __time_diff_dict: dict
    __cs_id_list: list
    __attack_sim_time: float
    __normal_sim_time: float

    def __init__(self, time_diff_dict, cs_id_list, attack_sim_time, normal_sim_time):
        self.__cs_id_list = cs_id_list
        self.__attack_sim_time = attack_sim_time
        self.__normal_sim_time = normal_sim_time

        temp_time_diff_dict = time_diff_dict[Constant_Parameters.GS_ID]

        time_diff_dict = {}
        for cs_id in cs_id_list:
            attack_time_diff_list = temp_time_diff_dict[Constant_Parameters.ATTACK][cs_id]
            normal_time_diff_list = temp_time_diff_dict[Constant_Parameters.NORMAL][cs_id]

            attack_time_diff_dict = {Constant_Parameters.TIME_DIFF: attack_time_diff_list}
            normal_time_diff_dict = {Constant_Parameters.TIME_DIFF: normal_time_diff_list}

            param_dict = {Constant_Parameters.ATTACK: attack_time_diff_dict,
                          Constant_Parameters.NORMAL: normal_time_diff_dict}
            time_diff_dict[cs_id] = param_dict

        self.__time_diff_dict = time_diff_dict

    def analyzing_time_diff_sampling_analysis_dict(self):
        time_diff_analysis_dict = {}

        for cs_id in self.__cs_id_list:
            temp_dict = self.__time_diff_dict[cs_id]
            attack_data_point_list = temp_dict[Constant_Parameters.ATTACK][Constant_Parameters.TIME_DIFF]
            normal_data_point_list = temp_dict[Constant_Parameters.NORMAL][Constant_Parameters.TIME_DIFF]

            attack_sampling_count = len(attack_data_point_list)
            normal_sampling_count = len(normal_data_point_list)

            attack_sampling_resolution = attack_sampling_count / self.__attack_sim_time
            normal_sampling_resolution = normal_sampling_count / self.__normal_sim_time

            param_attack_dict = {Constant_Parameters.DATA_POINT: attack_data_point_list,
                                 Constant_Parameters.SAMPLING_COUNT: attack_sampling_count,
                                 Constant_Parameters.SIMULATION_TIME: self.__attack_sim_time,
                                 Constant_Parameters.SAMPLING_RESOLUTION: attack_sampling_resolution}
            param_normal_dict = {Constant_Parameters.DATA_POINT: normal_data_point_list,
                                 Constant_Parameters.SAMPLING_COUNT: normal_sampling_count,
                                 Constant_Parameters.SIMULATION_TIME: self.__normal_sim_time,
                                 Constant_Parameters.SAMPLING_RESOLUTION: normal_sampling_resolution}

            time_diff_analysis_dict[cs_id] = {Constant_Parameters.TIME_DIFF:
                                                  {Constant_Parameters.ATTACK: param_attack_dict,
                                                   Constant_Parameters.NORMAL: param_normal_dict}}

        return time_diff_analysis_dict
