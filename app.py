import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import load_and_preprocess_data
from src.train_models import train_all_models
from src.evaluate import evaluate_all_models, save_results, get_best_models
import joblib


def save_models(models, model_dir='model'):
    os.makedirs(model_dir, exist_ok=True)
    
    model_mapping = {
        'Logistic Regression': 'logistic_regression_model.pkl',
        'Decision Tree': 'decision_tree_model.pkl',
        'KNN': 'knn_model.pkl',
        'Naive Bayes': 'naive_bayes_model.pkl',
        'Random Forest': 'random_forest_model.pkl',
        'XGBoost': 'xgboost_model.pkl'
    }
    
    for model_name, model in models.items():
        filepath = os.path.join(model_dir, model_mapping[model_name])
        joblib.dump(model, filepath)

    return True


def main():
    data_path = 'data/heart.csv'
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(data_path)
    models = train_all_models(X_train, y_train)
    results_df = evaluate_all_models(models, X_test, y_test)
    results_df = get_best_models(results_df)

    save_results(results_df, 'results.csv')
    save_models(models)


if __name__ == '__main__':
    main()
