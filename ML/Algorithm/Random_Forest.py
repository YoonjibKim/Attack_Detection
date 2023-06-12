from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


class Random_Forest:
    @classmethod
    def random_forest_run(cls, X_tn, y_tn, X_te, y_te):
        std_scale = StandardScaler()
        std_scale.fit(X_tn)
        X_tn_std = std_scale.transform(X_tn)
        X_te_std = std_scale.transform(X_te)

        clf_rf = RandomForestClassifier(max_depth=2, random_state=0)
        try:
            clf_rf.fit(X_tn_std, y_tn.ravel())
            pred_rf = clf_rf.predict(X_te_std)
            class_report = classification_report(y_te, pred_rf, output_dict=True, zero_division=0)
        except ValueError:
            class_report = None

        return class_report