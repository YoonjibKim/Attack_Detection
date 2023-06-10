import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import Constant_Parameters


class STAT_Analyser:
    __cs_stat_score_dict: dict
    __gs_stat_score_dict: dict

    __scenario_type: str

    def __init__(self, ml_scenario_path: str):
        temp_list = ml_scenario_path.split('/')
        self.__scenario_type = temp_list[2] + '_' + temp_list[3] + '_' + temp_list[4]

        root_dir_path = Constant_Parameters.ML_SCORE_DIR_PATH + '/' + Constant_Parameters.STAT_PATH
        cs_stat_file_path = root_dir_path + '/' + Constant_Parameters.CS + '/' + self.__scenario_type + '.json'
        gs_stat_file_path = root_dir_path + '/' + Constant_Parameters.GS + '/' + self.__scenario_type + '.json'

        with open(cs_stat_file_path, 'r') as f:
            self.__cs_stat_score_dict = json.load(f)
        with open(gs_stat_file_path, 'r') as f:
            self.__gs_stat_score_dict = json.load(f)

    @classmethod
    def building_combination_loss_rate_heatmap(cls, station_type):
        comb_loss_rate_path_list = []
        for root_path in Constant_Parameters.PROCESSED_DATASET_PATH_DICT.values():
            temp_path = root_path + '/' + Constant_Parameters.STAT_PATH + '/' + station_type + '/' + \
                        Constant_Parameters.COMBINATION_LOSS_RATE + '.json'
            comb_loss_rate_path_list.append(temp_path)

        attack_comb_loss_rate_dict = {}
        normal_comb_loss_rate_dict = {}
        for loss_rate_path in comb_loss_rate_path_list:
            with open(loss_rate_path, 'r') as f:
                temp_dict = json.load(f)
                param_attack = {}
                param_normal = {}
                for comb_type, category_dict in temp_dict.items():
                    attack_loss_rate = category_dict[Constant_Parameters.ATTACK]
                    normal_loss_rate = category_dict[Constant_Parameters.NORMAL]

                    comb_initials = cls.__get_initials(comb_type)

                    param_attack[comb_initials] = attack_loss_rate
                    param_normal[comb_initials] = normal_loss_rate

            abbreviation = ''
            for temp, temp_path in Constant_Parameters.PROCESSED_DATASET_PATH_DICT.items():
                if loss_rate_path.find(temp_path) > -1:
                    abbreviation = temp
                    break

            attack_comb_loss_rate_dict[abbreviation] = param_attack
            normal_comb_loss_rate_dict[abbreviation] = param_normal

        return attack_comb_loss_rate_dict, normal_comb_loss_rate_dict

    @classmethod
    def __get_initials(cls, full_name):
        initials = ''
        if full_name.find(Constant_Parameters.TIME_DIFF) > -1:
            initials += Constant_Parameters.TIME_DIFF_INIT + ' + '

        scenario = full_name.replace(Constant_Parameters.TIME_DIFF, '')

        temp_list = scenario.split('_')
        for temp in temp_list:
            if len(temp) > 0:
                capitals = temp.upper()
                capital = capitals[0]
                initials += capital + ' + '

        initials = initials[:-3]
        return initials

    @classmethod
    def __get_f1_score_dict(cls, score_dict):
        f1_score_dict = {}
        for scenario, ml_dict in score_dict.items():
            ml_type_dict = {}
            for ml_type, result_dict in ml_dict.items():
                f1_score = result_dict[Constant_Parameters.F1_SCORE]
                ml_type_dict[ml_type] = f1_score

            initials = cls.__get_initials(scenario)
            f1_score_dict[initials] = ml_type_dict

        return f1_score_dict

    def __drawing_f1_score(self, combined_score_dict, station_type):
        df = pd.DataFrame(combined_score_dict)
        sns.heatmap(df, cmap='YlGnBu', annot=True, fmt='1.3f', linewidths=.3)
        sns.set(font_scale=1.0)
        plt.xticks(rotation=-45)
        plt.gcf().set_size_inches(16, 13)
        plt.title('F1 Score: ' + self.__scenario_type)
        save_path = \
            Constant_Parameters.RESULT_STAT_DIR_PATH + '/' + station_type + '/' + Constant_Parameters.F1_SCORE_PATH + \
            '/' + self.__scenario_type + '_' + Constant_Parameters.F1_SCORE + '.png'
        plt.savefig(save_path)
        plt.clf()

    @classmethod
    def __extracting_score_without_statistics_dict(cls, f1_score_dict, merged_loss_rate_dict):
        score_list = []
        data = list(merged_loss_rate_dict.values())
        avg_loss_rate = np.mean(data)
        median_loss_rate = np.median(data)

        for comb_type, ml_type_dict in f1_score_dict.items():
            for ml_type, score in ml_type_dict.items():
                param_list = [comb_type, ml_type, merged_loss_rate_dict[comb_type], score]
                score_list.append(param_list)

        sorted_score_list = sorted(score_list, key=lambda x: x[3])
        worst_score_list = sorted_score_list[0]
        best_score_list = sorted_score_list[len(sorted_score_list) - 1]

        worst_score_dict = {Constant_Parameters.COMBINATION_TYPE: worst_score_list[0],
                            Constant_Parameters.ML_TYPE: worst_score_list[1],
                            Constant_Parameters.COMBINATION_LOSS_RATE: worst_score_list[2],
                            Constant_Parameters.F1_SCORE: worst_score_list[3]}
        best_score_dict = {Constant_Parameters.COMBINATION_TYPE: best_score_list[0],
                           Constant_Parameters.ML_TYPE: best_score_list[1],
                           Constant_Parameters.COMBINATION_LOSS_RATE: best_score_list[2],
                           Constant_Parameters.F1_SCORE: best_score_list[3]}

        return worst_score_dict, best_score_dict, avg_loss_rate, median_loss_rate

    @classmethod
    def merging_attack_and_normal_loss_rate_dict(cls, attack_comb_loss_rate_dict, normal_comb_loss_rate_dict):
        scenario_score_dict = {}
        for scenario, attack_dict in attack_comb_loss_rate_dict.items():
            normal_dict = normal_comb_loss_rate_dict[scenario]
            ml_avg_score_dict = {}
            for ml_type, attack_score in attack_dict.items():
                normal_score = normal_dict[ml_type]
                score_avg = (attack_score + normal_score) / 2
                ml_avg_score_dict[ml_type] = score_avg
            scenario_score_dict[scenario] = ml_avg_score_dict

        return scenario_score_dict

    @classmethod
    def drawing_loss_rate(cls, loss_rate_dict, station_type):
        df = pd.DataFrame(loss_rate_dict)
        sns.heatmap(df, cmap='YlGnBu', annot=True, fmt='1.3f', linewidths=.3)
        sns.set(font_scale=1.0)
        plt.xticks(rotation=-45)
        plt.gcf().set_size_inches(16, 13)
        plt.title('Loss Rate')
        save_path = Constant_Parameters.RESULT_STAT_DIR_PATH + '/' + station_type + '/' + \
                    Constant_Parameters.COMBINATION_LOSS_RATE + '.png'
        plt.savefig(save_path)
        plt.clf()

    def __get_best_score_with_statistics(self, f1_score_dict, merged_loss_rate_dict, station_type):
        sorted_loss_rate_tuple = sorted(merged_loss_rate_dict.items(), key=lambda x: x[1])
        smallest_loss_rate = sorted_loss_rate_tuple[0][1]
        all_param_list = []  

        score_list = []
        for temp in sorted_loss_rate_tuple:
            comb_type = temp[0]
            loss_rate = temp[1]

            ml_dict = f1_score_dict[comb_type]
            sorted_ml_tuple = sorted(ml_dict.items(), key=lambda x: x[1], reverse=True)
            max_tuple = sorted_ml_tuple[0]
            ml_type = max_tuple[0]
            score = max_tuple[1]
            score_list.append(score)

            param_list = [comb_type, ml_type, loss_rate, score]
            all_param_list.append(param_list)

        file_path = Constant_Parameters.RESULT_STAT_DIR_PATH + '/' + station_type + '/' + \
                    Constant_Parameters.BEST_FEATURE_PATH + '/' + self.__scenario_type + '_best_feature.csv'
        with open(file_path, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['combination', 'ml type', 'loss rate', 'f1 score'])
            wr.writerows(all_param_list)

        loss_rate_list = []
        for all_param in all_param_list:
            temp_loss_rate = all_param[2]
            if temp_loss_rate == smallest_loss_rate:
                loss_rate_list.append(all_param)

        sorted_loss_rate_list = sorted(loss_rate_list, key=lambda x: x[3], reverse=True)  # 이곳을 살펴 봐라
        smallest_loss_rate_list = sorted_loss_rate_list[0]

        param_comb_type = smallest_loss_rate_list[0]
        param_ml_type = smallest_loss_rate_list[1]
        param_loss_rate = smallest_loss_rate_list[2]
        param_score = smallest_loss_rate_list[3]
        param_score_avg = np.mean(score_list)
        param_score_median = np.median(score_list)

        return param_comb_type, param_ml_type, param_loss_rate, param_score, param_score_avg, param_score_median

    def run(self, cs_merged_loss_rate_dict, gs_merged_loss_rate_dict):
        print('--------------------------------------------------------------------------')
        print('scenario type: ', self.__scenario_type)

        cs_f1_score_dict = self.__get_f1_score_dict(self.__cs_stat_score_dict)
        gs_f1_score_dict = self.__get_f1_score_dict(self.__gs_stat_score_dict)

        self.__drawing_f1_score(cs_f1_score_dict, Constant_Parameters.CS)
        self.__drawing_f1_score(gs_f1_score_dict, Constant_Parameters.GS)

        print('----------------------- without statistic analysis -----------------------')
        cs_worst_score_dict, cs_best_score_dict, cs_avg_loss_rate, cs_median_loss_rate = \
            self.__extracting_score_without_statistics_dict(cs_f1_score_dict, cs_merged_loss_rate_dict)
        print('cs worst: ', cs_worst_score_dict)
        print('cs best: ', cs_best_score_dict)
        print('cs average loss rate: ', cs_avg_loss_rate)
        print('cs median loss rate: ', cs_median_loss_rate)

        gs_worst_score_dict, gs_best_score_dict, gs_avg_loss_rate, gs_median_loss_rate = \
            self.__extracting_score_without_statistics_dict(gs_f1_score_dict, gs_merged_loss_rate_dict)
        print('gs worst: ', gs_worst_score_dict)
        print('gs best: ', gs_best_score_dict)
        print('gs average loss rate: ', gs_avg_loss_rate)
        print('gs median loss rate: ', gs_median_loss_rate)

        print('------------------------- with statistic analysis ------------------------')
        cs_comb_type, cs_ml_type, cs_loss_rate, cs_f1_score, cs_f1_score_avg, cs_f1_score_median = \
            self.__get_best_score_with_statistics(cs_f1_score_dict, cs_merged_loss_rate_dict, Constant_Parameters.CS)
        print('cs best combination type: ', cs_comb_type)
        print('cs best ml type: ', cs_ml_type)
        print('cs loss rate: ', cs_loss_rate)
        print('cs f1 score: ', cs_f1_score)
        print('cs f1 score average: ', cs_f1_score_avg)
        print('cs f1 score median: ', cs_f1_score_median)

        gs_comb_type, gs_ml_type, gs_loss_rate, gs_f1_score, gs_f1_score_avg, gs_f1_score_median = \
            self.__get_best_score_with_statistics(gs_f1_score_dict, gs_merged_loss_rate_dict, Constant_Parameters.GS)
        print('gs best combination type: ', gs_comb_type)
        print('gs best ml type: ', gs_ml_type)
        print('gs loss rate: ', gs_loss_rate)
        print('gs f1 score: ', gs_f1_score)
        print('gs f1 score average: ', gs_f1_score_avg)
        print('gs f1 score median: ', gs_f1_score_median)

        cs_file_path = Constant_Parameters.RESULT_STAT_DIR_PATH + '/' + Constant_Parameters.CS + '/' + \
                       Constant_Parameters.STATISTICS_PATH + '/' + self.__scenario_type + '_statistics.csv'
        gs_file_path = Constant_Parameters.RESULT_STAT_DIR_PATH + '/' + Constant_Parameters.GS + '/' + \
                       Constant_Parameters.STATISTICS_PATH + '/' + self.__scenario_type + '_statistics.csv'

        with open(cs_file_path, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['without statistic analysis'])
            wr.writerow(['cs worst', 'cs best', 'cs average loss rate', 'cs median loss rate'])
            wr.writerow([cs_worst_score_dict, cs_best_score_dict, cs_avg_loss_rate, cs_median_loss_rate])
            wr.writerow(['with statistic analysis'])
            wr.writerow(['cs best combination type', 'cs best ml type', 'cs loss rate', 'cs f1 score',
                         'cs f1 score average', 'cs f1 score median'])
            wr.writerow([cs_comb_type, cs_ml_type, cs_loss_rate, cs_f1_score, cs_f1_score_avg, cs_f1_score_median])

        with open(gs_file_path, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['without statistic analysis'])
            wr.writerow(['gs worst', 'gs best', 'gs average loss rate', 'gs median loss rate'])
            wr.writerow([gs_worst_score_dict, gs_best_score_dict, gs_avg_loss_rate, gs_median_loss_rate])
            wr.writerow(['with statistic analysis'])
            wr.writerow(['gs best combination type', 'gs best ml type', 'gs loss rate', 'gs f1 score',
                         'gs f1 score average', 'gs f1 score median'])
            wr.writerow([gs_comb_type, gs_ml_type, gs_loss_rate, gs_f1_score, gs_f1_score_avg, gs_f1_score_median])
