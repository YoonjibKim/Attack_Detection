from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


class Gaussian_NB:
    @classmethod
    def gaussian_nb_run(cls, training_feature_array, training_label_array, testing_feature_array,
                        testing_label_array):
        std_scale = StandardScaler()
        std_scale.fit(training_feature_array)
        X_tn_std = std_scale.transform(training_feature_array)
        X_te_std = std_scale.transform(testing_feature_array)

        clf_gnb = GaussianNB()
        try:
            clf_gnb.fit(X_tn_std, training_label_array.ravel())
            pred_gnb = clf_gnb.predict(X_te_std)
            class_report = classification_report(testing_label_array, pred_gnb, output_dict=True, zero_division=0)
        except ValueError:
            class_report = None

        return class_report