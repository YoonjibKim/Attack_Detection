from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


class Logistic_Regression:
    @classmethod
    def logistic_regression_run(cls, training_feature_array, training_label_array, testing_feature_array,
                                testing_label_array):
        std_scale = StandardScaler()
        std_scale.fit(training_feature_array)
        X_tn_std = std_scale.transform(training_feature_array)
        X_te_std = std_scale.transform(testing_feature_array)

        clf_logi_l2 = LogisticRegression(penalty='l2')
        try:
            clf_logi_l2.fit(X_tn_std, training_label_array.ravel())
            pred_logistic = clf_logi_l2.predict(X_te_std)
            class_report = classification_report(testing_label_array, pred_logistic, output_dict=True, zero_division=0)
        except ValueError:
            class_report = None

        return class_report
