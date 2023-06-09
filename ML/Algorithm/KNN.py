from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class KNN:
    @classmethod
    def knn_run(cls, training_feature_array, training_label_array, testing_feature_array, testing_label_array):
        std_scale = StandardScaler()
        std_scale.fit(training_feature_array)
        X_tn_std = std_scale.transform(training_feature_array)
        X_te_std = std_scale.transform(testing_feature_array)

        try:
            clf_knn = KNeighborsClassifier(n_neighbors=2)
            clf_knn.fit(X_tn_std, training_label_array.ravel())
            knn_pred = clf_knn.predict(X_te_std)
            class_report = classification_report(testing_label_array, knn_pred, zero_division=0, output_dict=True)
        except ValueError:
            class_report = None

        return class_report
