import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def load_and_preprocess_data(filepath, missing_strategy="median", save_original=True):
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    missing_by_column = X.isnull().sum()

    all_missing_cols = missing_by_column[missing_by_column == X.shape[0]].index.tolist()
    if all_missing_cols:
        X = X.drop(columns=all_missing_cols)

    high_missing_cols = missing_by_column[missing_by_column > X.shape[0] * 0.5].index.tolist()
    high_missing_cols = [col for col in high_missing_cols if col not in all_missing_cols]
    if high_missing_cols:
        X = X.drop(columns=high_missing_cols)

    if np.any(np.isinf(X.values)):
        X = X.replace([np.inf, -np.inf], np.nan)

    constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)

    unique_values = np.unique(y)
    if len(unique_values) != 2:
        raise ValueError(f"Target must be binary, but found {len(unique_values)} classes: {unique_values}")

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if save_original:
        train_before_impute = X_train_raw.copy()
        train_before_impute["target"] = y_train_raw.values if hasattr(y_train_raw, "values") else y_train_raw
        train_before_impute.to_csv("X_train_before_impute.csv", index=False)

        test_before_impute = X_test_raw.copy()
        test_before_impute["target"] = y_test_raw.values if hasattr(y_test_raw, "values") else y_test_raw
        test_before_impute.to_csv("X_test_before_impute.csv", index=False)

    remaining_missing = X_train_raw.isnull().sum().sum() + X_test_raw.isnull().sum().sum()

    if remaining_missing > 0:
        if missing_strategy == "mean":
            imputer = SimpleImputer(strategy="mean")
        elif missing_strategy == "median":
            imputer = SimpleImputer(strategy="median")
        elif missing_strategy == "knn":
            imputer = KNNImputer(n_neighbors=5)
        else:
            imputer = SimpleImputer(strategy="most_frequent")

        try:
            X_train_imputed = imputer.fit_transform(X_train_raw)
            X_test_imputed = imputer.transform(X_test_raw)

            X_train = pd.DataFrame(X_train_imputed, columns=X_train_raw.columns, index=X_train_raw.index)
            X_test = pd.DataFrame(X_test_imputed, columns=X_test_raw.columns, index=X_test_raw.index)
        except Exception:
            X_train = X_train_raw.dropna()
            y_train_raw = y_train_raw.loc[X_train.index]

            X_test = X_test_raw.dropna()
            y_test_raw = y_test_raw.loc[X_test.index]
    else:
        X_train, X_test = X_train_raw, X_test_raw

    y_train = y_train_raw
    y_test = y_test_raw

    if X_train.shape[0] == 0 or X_train.shape[1] == 0:
        raise ValueError(f"Training set is empty: {X_train.shape}")
    if X_test.shape[0] == 0 or X_test.shape[1] == 0:
        raise ValueError(f"Test set is empty: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns.tolist(), X_test


def get_all_models(n_samples=100):
    models = {}

    models["LogisticRegression"] = LogisticRegression(
        random_state=42,
        max_iter=2000,
        solver="lbfgs",
        penalty="l2",
    )
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=min(100, n_samples // 5),
        random_state=42,
        max_depth=5 if n_samples < 100 else None,
    )
    models["DecisionTree"] = DecisionTreeClassifier(
        random_state=42,
        max_depth=5,
    )

    if n_samples >= 50:
        models["KNN"] = KNeighborsClassifier(n_neighbors=min(5, n_samples // 10))
        models["NaiveBayes"] = GaussianNB()
        models["SVM"] = SVC(probability=True, random_state=42, kernel="linear")
        models["LDA"] = LinearDiscriminantAnalysis()

    if n_samples >= 100:
        models["GradientBoosting"] = GradientBoostingClassifier(
            n_estimators=min(100, n_samples // 10),
            random_state=42,
        )
        models["AdaBoost"] = AdaBoostClassifier(
            n_estimators=min(100, n_samples // 10),
            random_state=42,
        )
        models["XGBoost"] = XGBClassifier(
            n_estimators=min(100, n_samples // 10),
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        models["ExtraTrees"] = ExtraTreesClassifier(
            n_estimators=min(100, n_samples // 10),
            random_state=42,
        )
        models["Bagging"] = BaggingClassifier(
            n_estimators=min(100, n_samples // 10),
            random_state=42,
        )

    if n_samples >= 200:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=min(100, n_samples // 10),
            random_state=42,
            verbose=-1,
        )
        models["MLP"] = MLPClassifier(
            hidden_layer_sizes=(50,),
            max_iter=1000,
            random_state=42,
            early_stopping=True,
        )
        models["CatBoost"] = CatBoostClassifier(
            n_estimators=min(100, n_samples // 10),
            random_state=42,
            verbose=0,
        )
        models["QDA"] = QuadraticDiscriminantAnalysis()

    return models


def evaluate_model(model, model_name, X_train, X_test, y_train, y_test, cv_folds=10):
    result = {
        "model_name": model_name,
        "model": model,
    }

    try:
        model.fit(X_train, y_train)

        y_train_pred_proba = model.predict_proba(X_train)[:, 1]
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]

        y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
        y_test_pred = (y_test_pred_proba >= 0.5).astype(int)

        def calculate_metrics(y_true, y_pred, y_pred_proba, prefix):
            return {
                f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
                f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
                f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
                f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
                f"{prefix}_auc": roc_auc_score(y_true, y_pred_proba),
                f"{prefix}_ap": average_precision_score(y_true, y_pred_proba),
            }

        result.update(calculate_metrics(y_train, y_train_pred, y_train_pred_proba, "train"))
        result.update(calculate_metrics(y_test, y_test_pred, y_test_pred_proba, "test"))

        cv_scores = {
            "accuracy": cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy"),
            "precision": cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="precision"),
            "recall": cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="recall"),
            "f1": cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="f1"),
            "roc_auc": cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="roc_auc"),
        }

        for metric, scores in cv_scores.items():
            result[f"cv_{metric}_mean"] = np.mean(scores)
            result[f"cv_{metric}_std"] = np.std(scores)

        fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
        result["fpr"] = fpr
        result["tpr"] = tpr

        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_pred_proba)
        result["precision_curve"] = precision_curve
        result["recall_curve"] = recall_curve

        if hasattr(model, "feature_importances_"):
            result["feature_importance"] = model.feature_importances_
        elif hasattr(model, "coef_"):
            result["feature_importance"] = np.abs(model.coef_[0])

        result["y_test_pred_proba"] = y_test_pred_proba
        result["y_train_pred_proba"] = y_train_pred_proba
        result["y_test"] = y_test.values if hasattr(y_test, "values") else y_test
        result["y_train"] = y_train.values if hasattr(y_train, "values") else y_train

        return result
    except Exception:
        return None


def save_all_results(all_results, filename="all_models_comparison.csv"):
    if not all_results:
        return None

    records = []
    for result in all_results:
        if result is None:
            continue

        record = {
            "model_name": result["model_name"],
            "train_accuracy": result["train_accuracy"],
            "train_precision": result["train_precision"],
            "train_recall": result["train_recall"],
            "train_f1": result["train_f1"],
            "train_auc": result["train_auc"],
            "train_ap": result["train_ap"],
            "test_accuracy": result["test_accuracy"],
            "test_precision": result["test_precision"],
            "test_recall": result["test_recall"],
            "test_f1": result["test_f1"],
            "test_auc": result["test_auc"],
            "test_ap": result["test_ap"],
            "cv_accuracy_mean": result["cv_accuracy_mean"],
            "cv_accuracy_std": result["cv_accuracy_std"],
            "cv_precision_mean": result["cv_precision_mean"],
            "cv_precision_std": result["cv_precision_std"],
            "cv_recall_mean": result["cv_recall_mean"],
            "cv_recall_std": result["cv_recall_std"],
            "cv_f1_mean": result["cv_f1_mean"],
            "cv_f1_std": result["cv_f1_std"],
            "cv_roc_auc_mean": result["cv_roc_auc_mean"],
            "cv_roc_auc_std": result["cv_roc_auc_std"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        records.append(record)

    results_df = pd.DataFrame(records).sort_values("test_auc", ascending=False)
    results_df.to_csv(filename, index=False, encoding="utf-8-sig")
    results_df.to_excel(filename.replace(".csv", ".xlsx"), index=False)
    return results_df


def generate_detailed_report(all_results, results_df):
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Machine Learning Model Comparison Report")
    report_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)

    valid_results = [r for r in all_results if r is not None]

    report_lines.append("\n1. Overall Summary")
    report_lines.append("-" * 40)
    report_lines.append(f"Total models evaluated: {len(all_results)}")
    report_lines.append(f"Valid models: {len(valid_results)}")

    report_lines.append("\n2. Model Ranking by Test AUC")
    report_lines.append("-" * 40)
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        report_lines.append(
            f"{i:2d}. {row['model_name']:20s} | "
            f"AUC: {row['test_auc']:.4f} | "
            f"F1: {row['test_f1']:.4f} | "
            f"Accuracy: {row['test_accuracy']:.4f}"
        )

    best_model_row = results_df.iloc[0]

    report_lines.append("\n3. Best Model")
    report_lines.append("-" * 40)
    report_lines.append(f"Best model: {best_model_row['model_name']}")
    report_lines.append(f"Test AUC: {best_model_row['test_auc']:.4f}")
    report_lines.append(f"Test F1: {best_model_row['test_f1']:.4f}")
    report_lines.append(f"Test Accuracy: {best_model_row['test_accuracy']:.4f}")
    report_lines.append(
        f"CV AUC mean: {best_model_row['cv_roc_auc_mean']:.4f} "
        f"(±{best_model_row['cv_roc_auc_std']:.4f})"
    )

    report_lines.append("\n4. Stability Analysis")
    report_lines.append("-" * 40)
    stable_df = results_df.sort_values("cv_roc_auc_std")
    for i, (_, row) in enumerate(stable_df.head(5).iterrows(), 1):
        report_lines.append(
            f"{i:2d}. {row['model_name']:20s} | "
            f"CV AUC mean: {row['cv_roc_auc_mean']:.4f} | "
            f"SD: {row['cv_roc_auc_std']:.4f}"
        )

    report_lines.append("\n5. Overfitting Analysis")
    report_lines.append("-" * 40)
    results_df = results_df.copy()
    results_df["overfit_gap"] = results_df["train_auc"] - results_df["test_auc"]
    overfit_df = results_df.sort_values("overfit_gap", ascending=False)
    for i, (_, row) in enumerate(overfit_df.head(5).iterrows(), 1):
        report_lines.append(
            f"{i:2d}. {row['model_name']:20s} | "
            f"Gap: {row['overfit_gap']:.4f} | "
            f"Train AUC: {row['train_auc']:.4f} | "
            f"Test AUC: {row['test_auc']:.4f}"
        )

    report_lines.append("\n6. Recommended Models")
    report_lines.append("-" * 40)
    results_df["composite_score"] = (
        0.6 * results_df["test_auc"]
        + 0.3 * (1 - results_df["cv_roc_auc_std"])
        + 0.1 * (1 - results_df["overfit_gap"])
    )
    composite_df = results_df.sort_values("composite_score", ascending=False)
    for i, (_, row) in enumerate(composite_df.head(3).iterrows(), 1):
        report_lines.append(
            f"{i}. {row['model_name']}: "
            f"Composite score: {row['composite_score']:.4f}, "
            f"AUC: {row['test_auc']:.4f}, "
            f"Stability: {1 - row['cv_roc_auc_std']:.4f}"
        )

    report_lines.append("\n" + "=" * 80)
    report_lines.append("End of Report")
    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)
    with open("model_comparison_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


def main():
    filepath = "/path/to/AP.csv"
    missing_strategy = "median"

    try:
        X_train, X_test, y_train, y_test, feature_names, X_test_original = load_and_preprocess_data(
            filepath,
            missing_strategy=missing_strategy,
        )
    except Exception:
        try:
            df = pd.read_csv(filepath)
            df_clean = df.dropna()

            if df_clean.shape[0] < 20:
                raise ValueError(f"Too few rows after dropping missing values: {df_clean.shape[0]}")

            X = df_clean.iloc[:, :-1]
            y = df_clean.iloc[:, -1]

            constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
            if constant_cols:
                X = X.drop(columns=constant_cols)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )

            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index,
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index,
            )

            X_train, X_test = X_train_scaled, X_test_scaled
            feature_names = X.columns.tolist()
            X_test_original = X_test.copy()
        except Exception:
            return

    n_samples = X_train.shape[0]
    models = get_all_models(n_samples)

    cv_folds = min(10, n_samples // 10)
    if cv_folds < 3:
        cv_folds = 3

    all_results = []

    for model_name, model in models.items():
        result = evaluate_model(
            model,
            model_name,
            X_train,
            X_test,
            y_train,
            y_test,
            cv_folds=cv_folds,
        )
        all_results.append(result)

    if any(r is not None for r in all_results):
        results_df = save_all_results(all_results)
        if results_df is not None:
            generate_detailed_report(all_results, results_df)


if __name__ == "__main__":
    main()