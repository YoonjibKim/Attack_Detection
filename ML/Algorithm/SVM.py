from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


class SVM:
    @classmethod
    def svm_run(cls, X_tn, y_tn, X_te, y_te):
        std_scale = StandardScaler()
        std_scale.fit(X_tn)
        X_tn_std = std_scale.transform(X_tn)
        X_te_std = std_scale.transform(X_te)

        clf_svm_lr = svm.SVC(kernel='linear', random_state=0)
        try:
            clf_svm_lr.fit(X_tn_std, y_tn.ravel())
            pred_svm = clf_svm_lr.predict(X_te_std)
            class_report = classification_report(y_te, pred_svm, output_dict=True, zero_division=0)
        except ValueError:
            class_report = None

        return class_report
