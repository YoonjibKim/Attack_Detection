from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


class Decision_Tree:
    @classmethod
    def decision_tree_run(cls, X_tn, y_tn, X_te, y_te):
        std_scale = StandardScaler()
        std_scale.fit(X_tn)
        X_tn_std = std_scale.transform(X_tn)
        X_te_std = std_scale.transform(X_te)

        clf_tree = tree.DecisionTreeClassifier(random_state=0)

        try:
            clf_tree.fit(X_tn_std, y_tn)
            pred_tree = clf_tree.predict(X_te_std)
            class_report = classification_report(y_te, pred_tree, output_dict=True, zero_division=0)
        except ValueError:
            class_report = None

        return class_report