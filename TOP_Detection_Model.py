import json
import numpy as np
import Constant_Parameters
from ML.Algorithm.Ada_boost import Ada_Boost
from ML.Algorithm.Agglomerative_Clustering import Agglomerative_Clustering
from ML.Algorithm.DB_Scan import DB_Scan
from ML.Algorithm.DNN import DNN
from ML.Algorithm.Decision_Tree import Decision_Tree
from ML.Algorithm.Gaussian_Mixture import Gaussian_Mixture
from ML.Algorithm.Gaussian_NB import Gaussian_NB
from ML.Algorithm.Gradient_Boost import Gradient_Boost
from ML.Algorithm.KNN import KNN
from ML.Algorithm.K_Means import K_Means
from ML.Algorithm.Linear_Regressions import Linear_Regressions
from ML.Algorithm.Logistic_Regression import Logistic_Regression
from ML.Algorithm.Random_Forest import Random_Forest
from ML.Algorithm.SVM import SVM


class TOP_Detection_Model:
    __default_filename: str

    __cs_feature_dict: dict
    __gs_feature_dict: dict

    def __init__(self, ml_scenario_path: str):
        temp_list = ml_scenario_path.split('/')
        filename = temp_list[2] + '_' + temp_list[3] + '_' + temp_list[4]
        filename += '.json'
        self.__default_filename = filename

        root_path = Constant_Parameters.ML_DATASET_PATH + '/' + Constant_Parameters.TOP_PATH
        cs_file_path = root_path + '/' + Constant_Parameters.CS + '/' + filename
        gs_file_path = root_path + '/' + Constant_Parameters.GS + '/' + filename

        with open(cs_file_path, 'r') as f:
            self.__cs_feature_dict = json.load(f)
        with open(gs_file_path, 'r') as f:
            self.__gs_feature_dict = json.load(f)

    def __run_sub_ml_algorithm(self, training_feature_array, training_label_array, testing_feature_array,
                           testing_label_array, ml_type):
        if ml_type == Constant_Parameters.ADA_BOOST:
            class_report = Ada_Boost.ada_boost_run(training_feature_array, training_label_array,
                                                   testing_feature_array, testing_label_array)
        elif ml_type == Constant_Parameters.AGGLOMERATIVE_CLUSTERING:
            class_report = Agglomerative_Clustering.agglomerative_clustering_run(testing_feature_array,
                                                                                 testing_label_array)
        elif ml_type == Constant_Parameters.DB_SCAN:
            class_report = DB_Scan.db_scan_run(testing_feature_array, testing_label_array)
        elif ml_type == Constant_Parameters.DNN:
            class_report = DNN.dnn_run(training_feature_array, training_label_array,
                                       testing_feature_array, testing_label_array)
        elif ml_type == Constant_Parameters.DECISION_TREE:
            class_report = Decision_Tree.decision_tree_run(training_feature_array, training_label_array,
                                                           testing_feature_array, testing_label_array)
        elif ml_type == Constant_Parameters.GAUSSIAN_MIXTURE:
            class_report = Gaussian_Mixture.gaussian_mixture_run(testing_feature_array, testing_label_array)
        elif ml_type == Constant_Parameters.GAUSSIAN_NB:
            class_report = Gaussian_NB.gaussian_nb_run(training_feature_array, training_label_array,
                                                       testing_feature_array, testing_label_array)
        elif ml_type == Constant_Parameters.GRADIENT_BOOST:
            class_report = Gradient_Boost.gradient_boost_run(training_feature_array, training_label_array,
                                                             testing_feature_array, testing_label_array)
        elif ml_type == Constant_Parameters.KNN:
            class_report = KNN.knn_run(training_feature_array, training_label_array,
                                       testing_feature_array, testing_label_array)
        elif ml_type == Constant_Parameters.KMEANS:
            class_report = K_Means.k_means_run(testing_feature_array, testing_label_array)
        elif ml_type == Constant_Parameters.LINEAR_REGRESSION:
            class_report_lr, class_report_ridge, class_report_lasso, class_report_elastic = \
                Linear_Regressions.linear_regressions_run(training_feature_array, training_label_array,
                                                          testing_feature_array, testing_label_array)
            class_report = class_report_lr
            self.__class_report_ridge = class_report_ridge
            self.__class_report_lasso = class_report_lasso
            self.__class_report_elastic = class_report_elastic
        elif ml_type == Constant_Parameters.LINEAR_REGRESSION_RIDGE:
            class_report = self.__class_report_ridge
        elif ml_type == Constant_Parameters.LINEAR_REGRESSION_LASSO:
            class_report = self.__class_report_lasso
        elif ml_type == Constant_Parameters.LINEAR_REGRESSION_ELASTIC:
            class_report = self.__class_report_elastic
        elif ml_type == Constant_Parameters.LOGISTIC_REGRESSION:
            class_report = Logistic_Regression.logistic_regression_run(training_feature_array, training_label_array,
                                                                       testing_feature_array, testing_label_array)
        elif ml_type == Constant_Parameters.RANDOM_FOREST:
            class_report = Random_Forest.random_forest_run(training_feature_array, training_label_array,
                                                           testing_feature_array, testing_label_array)
        elif ml_type == Constant_Parameters.SVM:
            class_report = SVM.svm_run(training_feature_array, training_label_array,
                                       testing_feature_array, testing_label_array)
        else:
            class_report = None
            print('No chosen ML')

        if class_report is not None:
            accuracy = class_report['accuracy']
            weighted_avg_dict = class_report['weighted avg']
            precision = weighted_avg_dict['precision']
            recall = weighted_avg_dict['recall']
            f1_score = weighted_avg_dict['f1-score']
            support = weighted_avg_dict['support']
        else:
            accuracy = Constant_Parameters.DUMMY_DATA
            precision = Constant_Parameters.DUMMY_DATA
            recall = Constant_Parameters.DUMMY_DATA
            f1_score = Constant_Parameters.DUMMY_DATA
            support = Constant_Parameters.DUMMY_DATA

        return accuracy, precision, recall, f1_score, support

    def __run_ml_algorithm(self, feature_dict, station_type):
        param_type_dict = {}
        for type_name, comb_dict in feature_dict.items():
            param_comb_dict = {}
            for comb_name, symbol_dict in comb_dict.items():
                param_symbol_dict = {}
                for symbol_name, category_dict in symbol_dict.items():
                    combined_sampling_resolution = category_dict[Constant_Parameters.COMBINATION_LOSS_RATE]
                    training_feature_array = np.array(category_dict[Constant_Parameters.TRAINING_FEATURE])
                    training_label_array = np.array(category_dict[Constant_Parameters.TRAINING_LABEL]).reshape(-1, 1)
                    testing_feature_array = np.array(category_dict[Constant_Parameters.TESTING_FEATURE])
                    testing_label_array = np.array(category_dict[Constant_Parameters.TESTING_LABEL]).reshape(-1, 1)

                    param_result_dict = {}
                    for ml_type in Constant_Parameters.ML_LIST:
                        print(station_type, type_name, comb_name, symbol_name, ml_type)

                        if training_feature_array.shape[0] < 1 or training_label_array.shape[0] < 1 or \
                                testing_feature_array.shape[0] < 1 or testing_label_array.shape[0] < 1:
                            print('ZERO VALUE ERROR')
                            accuracy = Constant_Parameters.DUMMY_DATA
                            precision = Constant_Parameters.DUMMY_DATA
                            recall = Constant_Parameters.DUMMY_DATA
                            f1_score = Constant_Parameters.DUMMY_DATA
                            support = Constant_Parameters.DUMMY_DATA
                            CLR = Constant_Parameters.DUMMY_DATA
                        else:
                            accuracy, precision, recall, f1_score, support = \
                                self.__run_sub_ml_algorithm(training_feature_array, training_label_array,
                                                            testing_feature_array, testing_label_array, ml_type)
                            CLR = combined_sampling_resolution

                        param_result_dict[ml_type] = \
                            {Constant_Parameters.COMBINATION_LOSS_RATE: CLR,
                             Constant_Parameters.ACCURACY: accuracy,
                             Constant_Parameters.PRECISION: precision,
                             Constant_Parameters.RECALL: recall,
                             Constant_Parameters.F1_SCORE: f1_score,
                             Constant_Parameters.SUPPORT: support}

                    param_symbol_dict[symbol_name] = param_result_dict

                param_comb_dict[comb_name] = param_symbol_dict

            param_type_dict[type_name] = param_comb_dict

        return param_type_dict

    def __saving_ml_result(self, cs_ml_result_dict, gs_ml_result_dict):
        cs_dir_path = \
            Constant_Parameters.ML_SCORE_DIR_PATH + '/' + Constant_Parameters.TOP_PATH + '/' + Constant_Parameters.CS
        gs_dir_path = \
            Constant_Parameters.ML_SCORE_DIR_PATH + '/' + Constant_Parameters.TOP_PATH + '/' + Constant_Parameters.GS

        cs_full_path = cs_dir_path + '/' + self.__default_filename
        gs_full_path = gs_dir_path + '/' + self.__default_filename

        with open(cs_full_path, 'w') as f:
            json.dump(cs_ml_result_dict, f)
        with open(gs_full_path, 'w') as f:
            json.dump(gs_ml_result_dict, f)

    def run(self):
        cs_result_dict = self.__run_ml_algorithm(self.__cs_feature_dict, Constant_Parameters.CS)
        gs_result_dict = self.__run_ml_algorithm(self.__gs_feature_dict, Constant_Parameters.GS)

        self.__saving_ml_result(cs_result_dict, gs_result_dict)
