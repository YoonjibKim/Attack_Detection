from sklearn.cluster import KMeans
from sklearn.metrics import classification_report


class K_Means:
    @classmethod
    def k_means_run(cls, testing_feature_array, testing_label_array):
        kmc = KMeans(n_clusters=2, init='random', n_init='auto', max_iter=100, random_state=0)
        try:
            kmc.fit(testing_feature_array)
            label_kmc = kmc.labels_
            class_report = classification_report(testing_label_array, label_kmc, output_dict=True, zero_division=0)
        except ValueError:
            class_report = None

        return class_report