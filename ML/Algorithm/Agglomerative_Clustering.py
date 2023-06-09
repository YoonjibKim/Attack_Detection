from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import classification_report


class Agglomerative_Clustering:
    @classmethod
    def agglomerative_clustering_run(cls, X, y):
        try:
            aggc2 = AgglomerativeClustering(n_clusters=2, linkage='complete')
            label_aggc2 = aggc2.fit_predict(X)
            class_report = classification_report(y, label_aggc2, output_dict=True, zero_division=0)
        except ValueError:
            class_report = None

        return class_report
