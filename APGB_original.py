import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve, average_precision_score
import seaborn as sns
import shap
import warnings
import csv
import os
from datetime import datetime
from collections import defaultdict
import traceback
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath, missing_strategy='median', save_original=True):
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    missing_by_column = X.isnull().sum()
    total_missing = missing_by_column.sum()
    all_missing_cols = missing_by_column[missing_by_column == X.shape[0]].index.tolist()
    if all_missing_cols:
        for col in all_missing_cols:
            pass
        else:
            pass
        X = X.drop(columns=all_missing_cols)
    else:
        pass
    high_missing_cols = missing_by_column[missing_by_column > X.shape[0] * 0.5].index.tolist()
    high_missing_cols = [col for col in high_missing_cols if col not in all_missing_cols]
    if high_missing_cols:
        for col in high_missing_cols[:10]:
            missing_pct = missing_by_column[col] / X.shape[0] * 100
        else:
            pass
        X = X.drop(columns=high_missing_cols)
    else:
        pass
    has_inf = np.any(np.isinf(X.values))
    if has_inf:
        X = X.replace([np.inf, -np.inf], np.nan)
    else:
        pass
    constant_cols = []
    for col in X.columns:
        if X[col].nunique() <= 1:
            constant_cols.append(col)
        else:
            pass
    else:
        pass
    if constant_cols:
        for col in constant_cols[:10]:
            if X[col].shape[0] > 0:
                pass
            else:
                pass
        else:
            pass
        X = X.drop(columns=constant_cols)
    else:
        pass
    unique_values = np.unique(y)
    if len(unique_values) != 2:
        raise ValueError(f'Target must be binary, but found {len(unique_values)} classes: {unique_values}')
    else:
        pass
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    if save_original:
        try:
            train_before_impute = X_train_raw.copy()
            train_before_impute['target'] = y_train_raw.values if hasattr(y_train_raw, 'values') else y_train_raw
            train_before_impute.to_csv('X_train_before_impute.csv', index=False)
            test_before_impute = X_test_raw.copy()
            test_before_impute['target'] = y_test_raw.values if hasattr(y_test_raw, 'values') else y_test_raw
            test_before_impute.to_csv('X_test_before_impute.csv', index=False)
        except Exception as e:
            pass
        else:
            pass
        finally:
            pass
    else:
        pass
    remaining_missing = X_train_raw.isnull().sum().sum() + X_test_raw.isnull().sum().sum()
    if remaining_missing > 0:
        if missing_strategy == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif missing_strategy == 'median':
            imputer = SimpleImputer(strategy='median')
        elif missing_strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        else:
            imputer = SimpleImputer(strategy='most_frequent')
        try:
            X_train_imputed = imputer.fit_transform(X_train_raw)
            X_test_imputed = imputer.transform(X_test_raw)
            X_train = pd.DataFrame(X_train_imputed, columns=X_train_raw.columns, index=X_train_raw.index)
            X_test = pd.DataFrame(X_test_imputed, columns=X_test_raw.columns, index=X_test_raw.index)
        except Exception as e:
            rows_before_train = X_train_raw.shape[0]
            X_train = X_train_raw.dropna()
            y_train_raw = y_train_raw.loc[X_train.index]
            rows_removed_train = rows_before_train - X_train.shape[0]
            rows_before_test = X_test_raw.shape[0]
            X_test = X_test_raw.dropna()
            y_test_raw = y_test_raw.loc[X_test.index]
            rows_removed_test = rows_before_test - X_test.shape[0]
        else:
            pass
        finally:
            pass
    else:
        X_train, X_test = (X_train_raw, X_test_raw)
    y_train = y_train_raw
    y_test = y_test_raw
    if X_train.shape[0] == 0 or X_train.shape[1] == 0:
        raise ValueError(f'Training set is empty! X_train shape: {X_train.shape}')
    else:
        pass
    if X_test.shape[0] == 0 or X_test.shape[1] == 0:
        raise ValueError(f'Test set is empty! X_test shape: {X_test.shape}')
    else:
        pass
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return (X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns.tolist(), X_test)

def get_all_models(n_samples=100):
    models = {}
    models['LogisticRegression'] = LogisticRegression(random_state=42, max_iter=2000, solver='lbfgs', penalty='l2')
    models['RandomForest'] = RandomForestClassifier(n_estimators=min(100, n_samples // 5), random_state=42, max_depth=5 if n_samples < 100 else None)
    models['DecisionTree'] = DecisionTreeClassifier(random_state=42, max_depth=5)
    if n_samples >= 50:
        models['KNN'] = KNeighborsClassifier(n_neighbors=min(5, n_samples // 10))
        models['NaiveBayes'] = GaussianNB()
        models['SVM'] = SVC(probability=True, random_state=42, kernel='linear')
        models['LDA'] = LinearDiscriminantAnalysis()
    else:
        pass
    if n_samples >= 100:
        models['GradientBoosting'] = GradientBoostingClassifier(n_estimators=min(100, n_samples // 10), random_state=42)
        models['AdaBoost'] = AdaBoostClassifier(n_estimators=min(100, n_samples // 10), random_state=42)
        models['XGBoost'] = XGBClassifier(n_estimators=min(100, n_samples // 10), random_state=42, use_label_encoder=False, eval_metric='logloss')
        models['ExtraTrees'] = ExtraTreesClassifier(n_estimators=min(100, n_samples // 10), random_state=42)
        models['Bagging'] = BaggingClassifier(n_estimators=min(100, n_samples // 10), random_state=42)
    else:
        pass
    if n_samples >= 200:
        models['LightGBM'] = LGBMClassifier(n_estimators=min(100, n_samples // 10), random_state=42, verbose=-1)
        models['MLP'] = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42, early_stopping=True)
        models['CatBoost'] = CatBoostClassifier(n_estimators=min(100, n_samples // 10), random_state=42, verbose=0)
        models['QDA'] = QuadraticDiscriminantAnalysis()
    else:
        pass
    return models

def perform_shap_analysis(model, model_name, X_test, X_test_original, feature_names, shap_output_dir='shap_analysis_results'):
    os.makedirs(shap_output_dir, exist_ok=True)
    try:
        model_type = str(type(model)).lower()

        def model_predict_proba_wrapper(X):
            return model.predict_proba(X)
        n_samples = min(100, X_test.shape[0])
        X_test_sampled = X_test.iloc[:n_samples] if n_samples < X_test.shape[0] else X_test
        X_test_original_sampled = X_test_original.iloc[:n_samples] if n_samples < X_test_original.shape[0] else X_test_original
        if 'tree' in model_type or 'forest' in model_type or 'boost' in model_type:
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_sampled)
            except Exception as e:
                background = shap.sample(X_test_sampled, min(50, X_test_sampled.shape[0]), random_state=42)
                explainer = shap.KernelExplainer(model_predict_proba_wrapper, background)
                shap_values = explainer.shap_values(X_test_sampled, nsamples=100)
            else:
                pass
            finally:
                pass
        else:
            background = shap.sample(X_test_sampled, min(50, X_test_sampled.shape[0]), random_state=42)
            explainer = shap.KernelExplainer(model_predict_proba_wrapper, background)
            shap_values = explainer.shap_values(X_test_sampled, nsamples=100)
        shap_values_processed = process_shap_values(shap_values, X_test_sampled.shape[1])
        if shap_values_processed is None:
            return (None, None)
        else:
            pass
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values_processed, X_test_sampled, feature_names=feature_names, show=False)
            plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            summary_path = os.path.join(shap_output_dir, f'shap_summary_{model_name}.png')
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            traceback.print_exc()
        else:
            pass
        finally:
            pass
        try:
            if len(shap_values_processed.shape) == 2:
                mean_abs_shap = np.abs(shap_values_processed).mean(axis=0)
            else:
                mean_abs_shap = np.abs(shap_values_processed)
            shap_importance_df = pd.DataFrame({'feature': feature_names[:len(mean_abs_shap)], 'mean_abs_shap': mean_abs_shap}).sort_values('mean_abs_shap', ascending=True)
            max_features = min(20, len(shap_importance_df))
            shap_importance_df = shap_importance_df.tail(max_features)
            plt.figure(figsize=(10, max_features * 0.3 + 2))
            bars = plt.barh(range(len(shap_importance_df)), shap_importance_df['mean_abs_shap'])
            plt.yticks(range(len(shap_importance_df)), shap_importance_df['feature'])
            plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)')
            plt.title(f'SHAP Feature Importance - {model_name}', fontsize=14, fontweight='bold')
            for i, (bar, value) in enumerate(zip(bars, shap_importance_df['mean_abs_shap'])):
                plt.text(bar.get_width() + bar.get_width() * 0.01, i, f'{value:.4f}', va='center', fontsize=9)
            else:
                pass
            plt.tight_layout()
            bar_path = os.path.join(shap_output_dir, f'shap_bar_{model_name}.png')
            plt.savefig(bar_path, dpi=300, bbox_inches='tight')
            plt.close()
            csv_path = os.path.join(shap_output_dir, f'shap_importance_{model_name}.csv')
            shap_importance_df.sort_values('mean_abs_shap', ascending=False).to_csv(csv_path, index=False)
        except Exception as e:
            traceback.print_exc()
        else:
            pass
        finally:
            pass
        try:
            if shap_values_processed.shape[0] > 0:
                sample_idx = 1
                plt.figure(figsize=(60, 7))
                if hasattr(explainer, 'expected_value'):
                    expected_value = explainer.expected_value
                    if hasattr(expected_value, '__len__'):
                        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
                    else:
                        pass
                else:
                    expected_value = np.mean(model.predict_proba(X_test_sampled)[:, 1])
                expected_value_rounded = round(float(expected_value), 2)
                shap_values_rounded = np.round(shap_values_processed[sample_idx].astype(float), 2)
                feature_values_original = X_test_original_sampled.iloc[sample_idx].copy()
                for col in feature_values_original.index:
                    try:
                        feature_values_original[col] = round(float(feature_values_original[col]), 2)
                    except:
                        pass
                    else:
                        pass
                    finally:
                        pass
                else:
                    pass
                if len(shap_values_rounded) == len(feature_names):
                    shap.force_plot(expected_value_rounded, shap_values_rounded, feature_values_original, feature_names=feature_names, matplotlib=True, show=False)

                    def format_plot_text():
                        ax = plt.gca()
                        texts = ax.texts
                        for text in texts:
                            content = text.get_text()
                            if '=' in content:
                                parts = content.split('=')
                                if len(parts) == 2:
                                    feature_name = parts[0].strip()
                                    feature_value = parts[1].strip()
                                    try:
                                        num_value = float(feature_value)
                                        formatted_value = f'{num_value:.2f}'
                                        new_content = f'{feature_name} = {formatted_value}'
                                        text.set_text(new_content)
                                    except (ValueError, TypeError):
                                        pass
                                    else:
                                        pass
                                    finally:
                                        pass
                                else:
                                    pass
                            else:
                                pass
                        else:
                            pass
                    format_plot_text()
                    ax = plt.gca()
                    for text in ax.texts:
                        text.set_fontsize(8)
                    else:
                        pass
                    force_path = os.path.join(shap_output_dir, f'shap_force_{model_name}_original.png')
                    plt.savefig(force_path, dpi=600, bbox_inches='tight')
                    plt.close()
                else:
                    pass
            else:
                pass
        except Exception as e:
            traceback.print_exc()
        else:
            pass
        finally:
            pass
        try:
            shap_values_path = os.path.join(shap_output_dir, f'shap_values_{model_name}.npy')
            np.save(shap_values_path, shap_values_processed)
        except:
            pass
        else:
            pass
        finally:
            pass
        return (shap_values_processed, explainer)
    except Exception as e:
        traceback.print_exc()
        return (None, None)
    else:
        pass
    finally:
        pass

def process_shap_values(shap_values, n_features):
    try:
        if shap_values is None:
            return None
        else:
            pass
        if hasattr(shap_values, 'shape'):
            pass
        else:
            pass
        result = None
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                result = shap_values[1]
            else:
                result = shap_values[0]
        elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
            if shap_values.shape[2] == 2:
                result = shap_values[:, :, 1]
            else:
                result = shap_values[:, :, 0]
        elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 2:
            result = shap_values
        elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 1:
            result = shap_values.reshape(1, -1)
        else:
            pass
        if result is None:
            return None
        else:
            pass
        if len(result.shape) == 2 and result.shape[1] != n_features:
            if result.shape[1] > n_features:
                result = result[:, :n_features]
            elif result.shape[1] < n_features:
                padded = np.zeros((result.shape[0], n_features))
                padded[:, :result.shape[1]] = result
                result = padded
            else:
                pass
        else:
            pass
        return result
    except Exception as e:
        traceback.print_exc()
        return None
    else:
        pass
    finally:
        pass

def evaluate_model(model, model_name, X_train, X_test, y_train, y_test, cv_folds=10):
    results = {'model_name': model_name, 'model': model}
    try:
        model.fit(X_train, y_train)
        y_train_pred_proba = model.predict_proba(X_train)[:, 1]
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
        y_test_pred = (y_test_pred_proba >= 0.5).astype(int)

        def calculate_metrics(y_true, y_pred, y_pred_proba, prefix):
            return {f'{prefix}_accuracy': accuracy_score(y_true, y_pred), f'{prefix}_precision': precision_score(y_true, y_pred, zero_division=0), f'{prefix}_recall': recall_score(y_true, y_pred, zero_division=0), f'{prefix}_f1': f1_score(y_true, y_pred, zero_division=0), f'{prefix}_auc': roc_auc_score(y_true, y_pred_proba), f'{prefix}_ap': average_precision_score(y_true, y_pred_proba)}
        results.update(calculate_metrics(y_train, y_train_pred, y_train_pred_proba, 'train'))
        results.update(calculate_metrics(y_test, y_test_pred, y_test_pred_proba, 'test'))
        cv_scores = {'accuracy': cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy'), 'precision': cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='precision'), 'recall': cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='recall'), 'f1': cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='f1'), 'roc_auc': cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='roc_auc')}
        for metric, scores in cv_scores.items():
            results[f'cv_{metric}_mean'] = np.mean(scores)
            results[f'cv_{metric}_std'] = np.std(scores)
        else:
            pass
        fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
        results['fpr'] = fpr
        results['tpr'] = tpr
        precision, recall, _ = precision_recall_curve(y_test, y_test_pred_proba)
        results['precision_curve'] = precision
        results['recall_curve'] = recall
        if hasattr(model, 'feature_importances_'):
            results['feature_importance'] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            results['feature_importance'] = np.abs(model.coef_[0])
        else:
            pass
        results['y_test_pred_proba'] = y_test_pred_proba
        results['y_train_pred_proba'] = y_train_pred_proba
        results['y_test'] = y_test.values
        results['y_train'] = y_train.values
        return results
    except Exception as e:
        return None
    else:
        pass
    finally:
        pass

def save_all_results(all_results, feature_names, filename='all_models_comparison.csv'):
    if not all_results:
        return
    else:
        pass
    records = []
    for result in all_results:
        if result:
            record = {'model_name': result['model_name'], 'train_accuracy': result['train_accuracy'], 'train_precision': result['train_precision'], 'train_recall': result['train_recall'], 'train_f1': result['train_f1'], 'train_auc': result['train_auc'], 'train_ap': result['train_ap'], 'test_accuracy': result['test_accuracy'], 'test_precision': result['test_precision'], 'test_recall': result['test_recall'], 'test_f1': result['test_f1'], 'test_auc': result['test_auc'], 'test_ap': result['test_ap'], 'cv_accuracy_mean': result['cv_accuracy_mean'], 'cv_accuracy_std': result['cv_accuracy_std'], 'cv_precision_mean': result['cv_precision_mean'], 'cv_precision_std': result['cv_precision_std'], 'cv_recall_mean': result['cv_recall_mean'], 'cv_recall_std': result['cv_recall_std'], 'cv_f1_mean': result['cv_f1_mean'], 'cv_f1_std': result['cv_f1_std'], 'cv_roc_auc_mean': result['cv_roc_auc_mean'], 'cv_roc_auc_std': result['cv_roc_auc_std'], 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            records.append(record)
        else:
            pass
    else:
        pass
    results_df = pd.DataFrame(records)
    results_df = results_df.sort_values('test_auc', ascending=False)
    results_df.to_csv(filename, index=False, encoding='utf-8-sig')
    results_df.to_excel(filename.replace('.csv', '.xlsx'), index=False)
    return results_df

def plot_model_comparisons(all_results):
    if not all_results:
        return
    else:
        pass
    valid_results = [r for r in all_results if r is not None]
    model_names = [r['model_name'] for r in valid_results]
    test_aucs = [r['test_auc'] for r in valid_results]
    test_f1s = [r['test_f1'] for r in valid_results]
    cv_auc_means = [r['cv_roc_auc_mean'] for r in valid_results]
    cv_auc_stds = [r['cv_roc_auc_std'] for r in valid_results]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax1 = axes[0, 0]
    x_pos = np.arange(len(model_names))
    bars1 = ax1.bar(x_pos, test_aucs, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Test AUC')
    ax1.set_title('Model Comparison - Test AUC', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    for bar, auc in zip(bars1, test_aucs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f'{auc:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        pass
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x_pos, test_f1s, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Test F1 Score')
    ax2.set_title('Model Comparison - Test F1 Score', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    for bar, f1 in zip(bars2, test_f1s):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        pass
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x_pos, cv_auc_means, yerr=cv_auc_stds, alpha=0.7, color='salmon', edgecolor='black', capsize=5)
    ax3.set_xlabel('Models')
    ax3.set_ylabel('CV AUC Mean ± SD')
    ax3.set_title('10-Fold Cross Validation AUC', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    for bar, mean, std in zip(bars3, cv_auc_means, cv_auc_stds):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)
    else:
        pass
    ax4 = axes[1, 1]
    top_n = min(5, len(valid_results))
    top_indices = np.argsort(test_aucs)[-top_n:][::-1]
    metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    for idx in top_indices:
        model_data = []
        for metric in metrics:
            value = valid_results[idx][metric]
            model_data.append(value)
        else:
            pass
        model_data = np.array(model_data)
        model_data = np.concatenate((model_data, [model_data[0]]))
        ax4.plot(angles, model_data, 'o-', linewidth=2, label=valid_results[idx]['model_name'])
        ax4.fill(angles, model_data, alpha=0.1)
    else:
        pass
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(metric_labels, fontsize=10)
    ax4.set_title(f'Top {top_n} Models - Radar Chart', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax4.grid(True)
    plt.tight_layout()
    plt.savefig('model_comparison_bar_charts.png', dpi=600, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, len(valid_results)))
    for i, result in enumerate(valid_results):
        fpr = result['fpr']
        tpr = result['tpr']
        auc = result['test_auc']
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f"{result['model_name']} (AUC = {auc:.3f})")
    else:
        pass
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison - All Models', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves_comparison.png', dpi=600, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(12, 10))
    for i, result in enumerate(valid_results):
        precision = result['precision_curve']
        recall = result['recall_curve']
        ap = result['test_ap']
        plt.plot(recall, precision, color=colors[i], lw=2, label=f"{result['model_name']} (AP = {ap:.3f})")
    else:
        pass
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves Comparison - All Models', fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig('pr_curves_comparison.png', dpi=600, bbox_inches='tight')
    plt.close()
    fig, ax = plt.subplots(figsize=(16, 10))
    metrics_heatmap = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc', 'test_ap']
    metric_labels_heatmap = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'AP']
    heatmap_data = []
    for result in valid_results:
        row = [result[metric] for metric in metrics_heatmap]
        heatmap_data.append(row)
    else:
        pass
    heatmap_data = np.array(heatmap_data)
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(np.arange(len(metric_labels_heatmap)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(metric_labels_heatmap, fontsize=11)
    ax.set_yticklabels(model_names, fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    for i in range(len(model_names)):
        for j in range(len(metric_labels_heatmap)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}', ha='center', va='center', color='black', fontsize=9)
        else:
            pass
    else:
        pass
    ax.set_title('Model Performance Heatmap', fontsize=16, fontweight='bold', pad=20)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('model_performance_heatmap.png', dpi=600, bbox_inches='tight')
    plt.close()

def plot_feature_importance_comparison(all_results, feature_names, normalize=True):
    models_with_importance = []
    importances_list = []
    raw_importances_list = []
    for result in all_results:
        if result and 'feature_importance' in result:
            models_with_importance.append(result['model_name'])
            importances_list.append(result['feature_importance'])
            raw_importances_list.append(result['feature_importance'].copy())
        else:
            pass
    else:
        pass
    if not models_with_importance:
        return
    else:
        pass
    max_features = min(20, len(feature_names))
    importance_df_normalized = pd.DataFrame(index=feature_names)
    importance_df_raw = pd.DataFrame(index=feature_names)
    for model_name, importance in zip(models_with_importance, importances_list):
        if len(importance) == len(feature_names):
            importance_df_raw[model_name] = importance
            if normalize:
                importance_normalized = (importance - np.min(importance)) / (np.max(importance) - np.min(importance) + 1e-10)
                importance_df_normalized[model_name] = importance_normalized
            else:
                importance_df_normalized[model_name] = importance
        else:
            pass
    else:
        pass
    if normalize:
        importance_df_normalized['Average'] = importance_df_normalized.mean(axis=1)
        sorted_df = importance_df_normalized.sort_values('Average', ascending=False)
    else:
        temp_avg = importance_df_raw.apply(lambda x: x / (np.max(x) + 1e-10), axis=0).mean(axis=1)
        sorted_df = importance_df_normalized.copy()
        sorted_df['Average'] = temp_avg
        sorted_df = sorted_df.sort_values('Average', ascending=False)
    top_features = sorted_df.head(max_features).index
    importance_df_top = importance_df_normalized.loc[top_features]
    importance_raw_top = importance_df_raw.loc[top_features]
    plt.figure(figsize=(16, 12))
    colors = plt.cm.tab20c(np.linspace(0, 1, len(models_with_importance)))
    bottom = np.zeros(len(top_features))
    for i, model in enumerate(models_with_importance):
        if normalize:
            values = importance_df_top[model].values
            label = f'{model}'
        else:
            values = importance_df_top[model].values
            label = model
        plt.barh(range(len(top_features)), values, left=bottom, color=colors[i], edgecolor='white', label=label, alpha=0.8)
        bottom += values
    else:
        pass
    plt.yticks(range(len(top_features)), top_features, fontsize=11)
    plt.xlabel('Normalized Feature Importance Score' if normalize else 'Feature Importance Score', fontsize=12)
    title = f'Top {max_features} Feature Importance Comparison'
    if normalize:
        title += ' (Normalized)'
    else:
        pass
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9, ncol=2)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.2)
    plt.xlim([0, len(models_with_importance) * 1.1])
    plt.tight_layout()
    output_file = 'feature_importance_comparison_normalized.png' if normalize else 'feature_importance_comparison.png'
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(14, 10))
    importance_raw_scaled = importance_raw_top.copy()
    for model in models_with_importance:
        col = importance_raw_scaled[model]
        if np.max(col) - np.min(col) > 0:
            importance_raw_scaled[model] = (col - np.min(col)) / (np.max(col) - np.min(col))
        else:
            pass
    else:
        pass
    plt.imshow(importance_raw_scaled.T, aspect='auto', cmap='RdYlBu_r')
    plt.colorbar(label='Normalized Importance')
    plt.xticks(range(len(top_features)), top_features, rotation=45, ha='right', fontsize=10)
    plt.yticks(range(len(models_with_importance)), models_with_importance, fontsize=10)
    plt.title(f'Raw Feature Importance Heatmap (Row-normalized)', fontsize=12)
    plt.xlabel('Features', fontsize=11)
    plt.ylabel('Models', fontsize=11)
    for i in range(len(models_with_importance)):
        for j in range(len(top_features)):
            val = importance_raw_top.iloc[j, i]
            plt.text(j, i, f'{val:.2e}', ha='center', va='center', color='white' if importance_raw_scaled.iloc[j, i] > 0.5 else 'black', fontsize=7)
        else:
            pass
    else:
        pass
    plt.tight_layout()
    plt.savefig('feature_importance_raw_heatmap.png', dpi=600, bbox_inches='tight')
    plt.close()
    importance_df_normalized.to_csv('feature_importance_normalized.csv', encoding='utf-8-sig')
    importance_df_raw.to_csv('feature_importance_raw.csv', encoding='utf-8-sig')
    return {'normalized_df': importance_df_normalized, 'raw_df': importance_df_raw, 'top_features': top_features.tolist()}

def generate_detailed_report(all_results, results_df):
    report_lines = []
    report_lines.append('=' * 80)
    report_lines.append('Machine Learning Model Comparison Report')
    report_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append('=' * 80)
    report_lines.append('\n1. Overall Summary')
    report_lines.append('-' * 40)
    report_lines.append(f'Total models evaluated: {len(all_results)}')
    report_lines.append(f'Valid models: {len([r for r in all_results if r is not None])}')
    report_lines.append('\n2. Model Ranking (by Test AUC, descending)')
    report_lines.append('-' * 40)
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        report_lines.append(f"{i:2d}. {row['model_name']:20s} | AUC: {row['test_auc']:.4f} | F1: {row['test_f1']:.4f} | Accuracy: {row['test_accuracy']:.4f}")
    else:
        pass
    report_lines.append('\n3. Best Model Analysis')
    report_lines.append('-' * 40)
    best_model_row = results_df.iloc[0]
    report_lines.append(f"Best model: {best_model_row['model_name']}")
    report_lines.append(f"Test AUC: {best_model_row['test_auc']:.4f}")
    report_lines.append(f"Test F1: {best_model_row['test_f1']:.4f}")
    report_lines.append(f"Test Accuracy: {best_model_row['test_accuracy']:.4f}")
    report_lines.append(f"10-fold CV AUC mean: {best_model_row['cv_roc_auc_mean']:.4f} (±{best_model_row['cv_roc_auc_std']:.4f})")
    report_lines.append('\n4. Model Stability Analysis (smaller CV SD indicates higher stability)')
    report_lines.append('-' * 40)
    stable_df = results_df.sort_values('cv_roc_auc_std')
    for i, (_, row) in enumerate(stable_df.head(5).iterrows(), 1):
        report_lines.append(f"{i:2d}. {row['model_name']:20s} | CV AUC mean: {row['cv_roc_auc_mean']:.4f} | Standard deviation: {row['cv_roc_auc_std']:.4f}")
    else:
        pass
    report_lines.append('\n5. Overfitting Analysis (Train AUC - Test AUC)')
    report_lines.append('-' * 40)
    results_df['overfit_gap'] = results_df['train_auc'] - results_df['test_auc']
    overfit_df = results_df.sort_values('overfit_gap', ascending=False)
    for i, (_, row) in enumerate(overfit_df.head(5).iterrows(), 1):
        report_lines.append(f"{i:2d}. {row['model_name']:20s} | Overfitting gap: {row['overfit_gap']:.4f} | Train AUC: {row['train_auc']:.4f} | Test AUC: {row['test_auc']:.4f}")
    else:
        pass
    report_lines.append('\n6. Model Recommendation')
    report_lines.append('-' * 40)
    report_lines.append('Recommended models based on overall performance, stability, and overfitting profile:')
    results_df['composite_score'] = 0.6 * results_df['test_auc'] + 0.3 * (1 - results_df['cv_roc_auc_std']) + 0.1 * (1 - results_df['overfit_gap'])
    composite_df = results_df.sort_values('composite_score', ascending=False)
    for i, (_, row) in enumerate(composite_df.head(3).iterrows(), 1):
        report_lines.append(f"{i}. {row['model_name']}: Composite score: {row['composite_score']:.4f}, AUC: {row['test_auc']:.4f}, Stability: {1 - row['cv_roc_auc_std']:.4f}")
    else:
        pass
    report_lines.append('\n' + '=' * 80)
    report_lines.append('End of Report')
    report_lines.append('=' * 80)
    report_text = '\n'.join(report_lines)
    with open('model_comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    return report_text

def calculate_net_benefit(y_true, y_pred_proba, threshold):
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n = len(y_true)
    net_benefit = tp / n - fp / n * (threshold / (1 - threshold))
    return net_benefit

def plot_dca_curves(y_true, predictions_dict, model_names=None, thresholds=None, save_path='dca_curve.png'):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    else:
        pass
    if model_names is None:
        model_names = list(predictions_dict.keys())
    else:
        pass
    plt.figure(figsize=(12, 8))
    prevalence = np.mean(y_true)
    treat_all_nb = []
    treat_none_nb = []
    for threshold in thresholds:
        if threshold >= 0.95:
            nb = -0.1
        else:
            nb = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
            nb = max(nb, -0.1)
        treat_all_nb.append(nb)
        treat_none_nb.append(0)
    else:
        pass
    plt.plot(thresholds, treat_all_nb, 'black', linestyle='--', linewidth=2.5, alpha=0.8, label='Treat All', zorder=1)
    plt.plot(thresholds, treat_none_nb, 'black', linestyle=':', linewidth=2.5, alpha=0.8, label='Treat None', zorder=1)
    net_benefits = {}
    for name in model_names:
        if name in predictions_dict:
            y_pred_proba = predictions_dict[name]
            nb_values = []
            for threshold in thresholds:
                nb = calculate_net_benefit(y_true, y_pred_proba, threshold)
                nb = max(nb, -0.1)
                nb_values.append(nb)
            else:
                pass
            net_benefits[name] = nb_values
        else:
            pass
    else:
        pass
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    for i, (name, nb_values) in enumerate(net_benefits.items()):
        plt.plot(thresholds, nb_values, color=colors[i], linewidth=3, label=f'{name}', alpha=0.9, zorder=2)
    else:
        pass
    plt.xlabel('Threshold Probability', fontsize=12)
    plt.ylabel('Net Benefit', fontsize=12)
    plt.title('Decision Curve Analysis (DCA)', fontsize=14, fontweight='bold')
    if len(model_names) > 5:
        plt.legend(loc='upper right', fontsize=9, ncol=2)
    else:
        plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([0, 1])
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.ylim([-0.1, None])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return net_benefits

def plot_clinical_impact_curve(y_true, y_pred_proba, model_name='Model', thresholds=None, save_path='clinical_impact_curve.png'):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    else:
        pass
    n = len(y_true)
    high_risk_proportions = []
    true_positives_in_high_risk = []
    for threshold in thresholds:
        high_risk_mask = y_pred_proba >= threshold
        high_risk_count = np.sum(high_risk_mask)
        high_risk_proportion = high_risk_count / n
        true_positives = np.sum(high_risk_mask & (y_true == 1))
        true_positive_proportion = true_positives / n
        high_risk_proportions.append(high_risk_proportion)
        true_positives_in_high_risk.append(true_positive_proportion)
    else:
        pass
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, high_risk_proportions, 'b-', linewidth=2, label='High Risk Patients', alpha=0.8)
    plt.plot(thresholds, true_positives_in_high_risk, 'r-', linewidth=2, label='True Positives in High Risk', alpha=0.8)
    prevalence = np.mean(y_true)
    plt.axhline(y=prevalence, color='gray', linestyle='--', linewidth=1, alpha=0.5, label=f'Prevalence ({prevalence:.3f})')
    plt.xlabel('Threshold Probability', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    plt.title(f'Clinical Impact Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return {'thresholds': thresholds, 'high_risk_proportions': high_risk_proportions, 'true_positives_in_high_risk': true_positives_in_high_risk}

def plot_comprehensive_dca_analysis(y_true, predictions_dict, model_names=None, save_dir='dca_analysis'):
    import os
    os.makedirs(save_dir, exist_ok=True)
    if model_names is None:
        model_names = list(predictions_dict.keys())
    else:
        pass
    prevalence = np.mean(y_true)
    dca_save_path = os.path.join(save_dir, 'dca_curves.png')
    try:
        net_benefits = plot_dca_curves(y_true, predictions_dict, model_names, save_path=dca_save_path)
    except Exception as e:
        net_benefits = {}
    else:
        pass
    finally:
        pass
    for name in model_names:
        if name in predictions_dict:
            try:
                cic_save_path = os.path.join(save_dir, f'clinical_impact_{name}.png')
                plot_clinical_impact_curve(y_true, predictions_dict[name], name, save_path=cic_save_path)
            except Exception as e:
                pass
            else:
                pass
            finally:
                pass
        else:
            pass
    else:
        pass
    thresholds = np.linspace(0.01, 0.99, 99)
    threshold_analysis_results = []
    for threshold in thresholds:
        row = {'threshold': threshold}
        prevalence = np.mean(y_true)
        treat_all_nb = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
        treat_none_nb = 0
        treat_all_nb = max(treat_all_nb, -0.1)
        row['treat_all'] = treat_all_nb
        row['treat_none'] = treat_none_nb
        for name in model_names:
            if name in predictions_dict:
                nb = calculate_net_benefit(y_true, predictions_dict[name], threshold)
                nb = max(nb, -0.1)
                row[name] = nb
            else:
                pass
        else:
            pass
        threshold_analysis_results.append(row)
    else:
        pass
    threshold_df = pd.DataFrame(threshold_analysis_results)
    threshold_file = os.path.join(save_dir, 'threshold_analysis.csv')
    threshold_df.to_csv(threshold_file, index=False)
    if net_benefits:
        for name, nb_values in net_benefits.items():
            if nb_values:
                max_nb = max(nb_values)
                max_idx = np.argmax(nb_values)
                best_threshold = thresholds[max_idx]
                if best_threshold < 0.2:
                    pass
                elif best_threshold < 0.3:
                    pass
                elif best_threshold < 0.5:
                    pass
                elif best_threshold < 0.7:
                    pass
                else:
                    pass
                treat_all_nb_at_best = prevalence - (1 - prevalence) * (best_threshold / (1 - best_threshold))
                improvement = max_nb - max(treat_all_nb_at_best, 0)
                if improvement > 0:
                    pass
                else:
                    pass
            else:
                pass
        else:
            pass
    else:
        pass
    dca_report_file = os.path.join(save_dir, 'dca_analysis_report.txt')
    with open(dca_report_file, 'w') as f:
        f.write('=' * 60 + '\n')
        f.write('Decision Curve Analysis (DCA) Report\n')
        f.write('=' * 60 + '\n\n')
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f'Total samples: {len(y_true)}\n')
        f.write(f'Prevalence: {prevalence:.4f}\n')
        f.write(f'Positive samples: {np.sum(y_true)}\n')
        f.write(f'Negative samples: {np.sum(y_true == 0)}\n\n')
        f.write('Maximum Net Benefit Analysis for Each Model:\n')
        f.write('-' * 40 + '\n')
        for name, nb_values in net_benefits.items():
            if nb_values:
                max_nb = max(nb_values)
                max_idx = np.argmax(nb_values)
                best_threshold = thresholds[max_idx]
                f.write(f'{name}:\n')
                f.write(f'  Maximum net benefit: {max_nb:.4f}\n')
                f.write(f'  Best threshold: {best_threshold:.3f}\n\n')
            else:
                pass
        else:
            pass
    return net_benefits

def main():
    filepath = 'NA'
    missing_strategy = 'median'
    try:
        X_train, X_test, y_train, y_test, feature_names, X_test_original = load_and_preprocess_data(filepath, missing_strategy=missing_strategy)
        if X_train.shape[0] < 50 or X_test.shape[0] < 10:
            pass
        else:
            pass
    except Exception as e:
        try:
            df = pd.read_csv(filepath)
            df_clean = df.dropna()
            if df_clean.shape[0] < 20:
                raise ValueError(f'Too few rows after dropping missing values ({df_clean.shape[0]} rows)')
            else:
                pass
            X = df_clean.iloc[:, :-1]
            y = df_clean.iloc[:, -1]
            constant_cols = []
            for col in X.columns:
                if X[col].nunique() <= 1:
                    constant_cols.append(col)
                else:
                    pass
            else:
                pass
            if constant_cols:
                X = X.drop(columns=constant_cols)
            else:
                pass
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
            X_train, X_test = (X_train_scaled, X_test_scaled)
            feature_names = X.columns.tolist()
            X_test_original = X_test.copy()
        except Exception as e2:
            return
        else:
            pass
        finally:
            pass
    else:
        pass
    finally:
        pass
    n_samples = X_train.shape[0]
    models = get_all_models(n_samples)
    for i, name in enumerate(models.keys(), 1):
        pass
    else:
        pass
    cv_folds = min(10, n_samples // 10)
    if cv_folds < 3:
        cv_folds = 3
    else:
        pass
    all_results = []
    total_models = len(models)
    completed = 0
    shap_output_dir = 'shap_analysis_results'
    os.makedirs(shap_output_dir, exist_ok=True)
    for model_name, model in models.items():
        completed += 1
        try:
            result = evaluate_model(model, model_name, X_train, X_test, y_train, y_test, cv_folds=cv_folds)
            if result:
                all_results.append(result)
                shap_values, explainer = perform_shap_analysis(model, model_name, X_test, X_test_original, feature_names, shap_output_dir)
                if shap_values is not None:
                    result['shap_values'] = shap_values
                    result['shap_explainer'] = explainer
                else:
                    pass
            else:
                all_results.append(None)
        except Exception as e:
            all_results.append(None)
        else:
            pass
        finally:
            pass
    else:
        pass
    if all_results and any((r is not None for r in all_results)):
        results_df = save_all_results(all_results, feature_names)
        plot_model_comparisons(all_results)
        plot_feature_importance_comparison(all_results, feature_names)
        if results_df is not None:
            generate_detailed_report(all_results, results_df)
        else:
            pass
        valid_predictions = []
        valid_model_names = []
        valid_true_labels = None
        for result in all_results:
            if result and 'y_test_pred_proba' in result:
                valid_predictions.append(result['y_test_pred_proba'])
                valid_model_names.append(result['model_name'])
                if valid_true_labels is None:
                    valid_true_labels = result['y_test']
                else:
                    pass
            else:
                pass
        else:
            pass
        if valid_predictions:
            predictions_all = pd.DataFrame({'true_label': valid_true_labels})
            for model_name, pred_proba in zip(valid_model_names, valid_predictions):
                predictions_all[model_name] = pred_proba
            else:
                pass
            predictions_all.to_csv('all_models_predictions_for_dca.csv', index=False)
        else:
            pass
        predictions_dict = {}
        for model_name, pred_proba in zip(valid_model_names, valid_predictions):
            predictions_dict[model_name] = pred_proba
        else:
            pass
        try:
            plot_comprehensive_dca_analysis(y_true=valid_true_labels, predictions_dict=predictions_dict, model_names=valid_model_names, save_dir='dca_analysis_results')
        except Exception as e:
            try:
                plot_dca_curves(y_true=valid_true_labels, predictions_dict=predictions_dict, model_names=valid_model_names, save_path='simple_dca_curve.png')
            except Exception as e2:
                pass
            else:
                pass
            finally:
                pass
        else:
            pass
        finally:
            pass
    else:
        pass
    successful = len([r for r in all_results if r is not None])
    failed = len([r for r in all_results if r is None])
    if successful > 0:
        pass
    else:
        pass
if __name__ == '__main__':
    main()
else:
    pass