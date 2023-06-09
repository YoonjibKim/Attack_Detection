import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


class Linear_Regressions:
    @classmethod
    def linear_regressions_run(cls, X_tn, y_tn, X_te, y_te):
        std_scale = StandardScaler()
        std_scale.fit(X_tn)
        X_tn_std = std_scale.transform(X_tn)
        X_te_std = std_scale.transform(X_te)

        clf_lr = LinearRegression()
        try:
            clf_lr.fit(X_tn_std, y_tn)
            pred_lr = clf_lr.predict(X_te_std)
            class_report_lr = classification_report(y_te, np.round(pred_lr), zero_division=0, output_dict=True)
        except ValueError:
            class_report_lr = None

        try:
            # 릿지 회귀분석(L2 제약식 적용)
            clf_ridge = Ridge(alpha=1)
            clf_ridge.fit(X_tn_std, y_tn)
            pred_ridge = clf_ridge.predict(X_te_std)
            class_report_ridge = classification_report(y_te, np.round(pred_ridge), zero_division=0, output_dict=True)
        except ValueError:
            class_report_ridge = None

        try:
            # 라쏘 회귀분석(L1 제약식 적용)
            clf_lasso = Lasso(alpha=0.01)
            clf_lasso.fit(X_tn_std, y_tn)
            pred_lasso = clf_lasso.predict(X_te_std)
            class_report_lasso = classification_report(y_te, np.round(pred_lasso), zero_division=0, output_dict=True)
        except ValueError:
            class_report_lasso = None

        try:
            # 엘라스틱넷
            clf_elastic = ElasticNet(alpha=0.01, l1_ratio=0.01)
            clf_elastic.fit(X_tn_std, y_tn)
            pred_elastic = clf_elastic.predict(X_te_std)
            class_report_elastic = classification_report(y_te, np.round(pred_elastic), zero_division=0,
                                                         output_dict=True)
        except ValueError:
            class_report_elastic = None

        return class_report_lr, class_report_ridge, class_report_lasso, class_report_elastic
