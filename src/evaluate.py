import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


def evaluate_all_models(models, X_test, y_test):
    results = []
    
    for model_name, model in models.items():
        print(f"\nModel: {model_name}")
        
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except:
            try:
                y_pred_proba = model.decision_function(X_test)
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
            except:
                y_pred_proba = None
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        if y_pred_proba is not None:
            auc_score = roc_auc_score(y_test, y_pred_proba)
        else:
            auc_score = None
        
        print(f"\nEvaluation Metrics:")
        print(f"  Accuracy:                    {accuracy:.4f}")
        print(f"  AUC Score:                   {auc_score:.4f}" if auc_score else f"  AUC Score:                   N/A")
        print(f"  Precision:                   {precision:.4f}")
        print(f"  Recall:                      {recall:.4f}")
        print(f"  F1 Score:                    {f1:.4f}")
        print(f"  Matthews Correlation Coeff:  {mcc:.4f}")
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:              {cm[0][0]}")
        print(f"  False Positives:             {cm[0][1]}")
        print(f"  False Negatives:             {cm[1][0]}")
        print(f"  True Positives:              {cm[1][1]}")
        
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'AUC Score': auc_score,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'MCC Score': mcc
        })

    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("ALL MODELS")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    return results_df


def save_results(results_df, filepath='results.csv'):
    results_df.to_csv(filepath, index=False)
    return filepath


def get_best_models(results_df):
    print("\nBEST MODELS BY METRIC:")
    
    metrics = ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC Score']
    
    for metric in metrics:
        best_idx = results_df[metric].idxmax()
        best_model = results_df.loc[best_idx, 'Model']
        best_score = results_df.loc[best_idx, metric]
        print(f"\n{metric:30s}: {best_model:20s} ({best_score:.4f})")

    numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC Score']
    results_df['Average Score'] = results_df[numeric_cols].mean(axis=1)
    
    best_overall_idx = results_df['Average Score'].idxmax()
    best_overall_model = results_df.loc[best_overall_idx, 'Model']
    best_overall_score = results_df.loc[best_overall_idx, 'Average Score']
    
    print(f"\n{'Overall Best Model':30s}: {best_overall_model:20s} ({best_overall_score:.4f})")
    
    return results_df
