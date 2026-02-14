import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler


def build_preprocessors(data_path='data/heart.csv'):
    df = pd.read_csv(data_path, sep='\t')
    df = df.drop(['id', 'dataset'], axis=1)
    X = df.drop('num', axis=1)
    y = (df['num'] > 0).astype(int)

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    label_encoders = {}
    X_enc = X.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        X_enc[col] = X_enc[col].fillna('missing')
        le.fit(X_enc[col].astype(str))
        X_enc[col] = le.transform(X_enc[col].astype(str))
        label_encoders[col] = le

    for col in numerical_cols:
        if X_enc[col].isnull().sum() > 0:
            X_enc[col] = X_enc[col].fillna(X_enc[col].median())

    scaler = StandardScaler()
    scaler.fit(X_enc)

    feature_columns = X_enc.columns.tolist()
    return label_encoders, scaler, feature_columns, categorical_cols, numerical_cols


def safe_encode(value, encoder: LabelEncoder):
    v = str(value)
    if v in encoder.classes_:
        return int(encoder.transform([v])[0])
    
    return int(encoder.transform([encoder.classes_[0]])[0])


def preprocess_input(input_df, label_encoders, scaler, feature_columns, categorical_cols, numerical_cols):
    df = input_df.copy()

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('missing').apply(lambda x: safe_encode(x, label_encoders[col]))

    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                df[col] = df[col].fillna(0)

    df = df[feature_columns]
    scaled = scaler.transform(df)
    return pd.DataFrame(scaled, columns=feature_columns)


def load_models(model_dir='model'):
    import os
    mapping = {
        'Logistic Regression': 'logistic_regression_model.pkl',
        'Decision Tree': 'decision_tree_model.pkl',
        'KNN': 'knn_model.pkl',
        'Naive Bayes': 'naive_bayes_model.pkl',
        'Random Forest': 'random_forest_model.pkl',
        'XGBoost': 'xgboost_model.pkl'
    }
    models = {}
    for name, fname in mapping.items():
        path = os.path.join(model_dir, fname)
        try:
            models[name] = joblib.load(path)
        except Exception as e:
            models[name] = None
    return models


def predict_with_model(model, X_proc):
    if model is None:
        return None, None
    try:
        proba = model.predict_proba(X_proc)[:, 1]
    except Exception:
        try:
            scores = model.decision_function(X_proc)
            proba = (scores - scores.min()) / (scores.max() - scores.min())
        except Exception:
            proba = None
    pred = model.predict(X_proc)
    return pred, proba


def streamlit_app():
    st.title('Heart Disease Predictor')
    st.write('Enter patient details below and select a model to predict heart disease presence.')
    
    st.sidebar.header('Download Test Data')
    if st.sidebar.button('Download Sample Test Dataset'):
        test_df = pd.read_csv('data/sample_test.csv', sep='\t')
        csv = test_df.to_csv(index=False, sep=',')
        st.sidebar.download_button(
            label='Download sample_test.csv',
            data=csv,
            file_name='sample_test.csv',
            mime='text/csv'
        )

    label_encoders, scaler, feature_columns, categorical_cols, numerical_cols = build_preprocessors()
    models = load_models()

    st.sidebar.header('Model selection')
    model_name = st.sidebar.selectbox('Choose model', list(models.keys()))

    st.header('Patient input')
    age = st.number_input('Age', min_value=1, max_value=120, value=55)
    sex = st.selectbox('Sex', options=list(label_encoders['sex'].classes_))
    cp = st.selectbox('Chest pain type', options=list(label_encoders['cp'].classes_))
    trestbps = st.number_input('Resting blood pressure', value=130)
    chol = st.number_input('Cholesterol', value=200)
    fbs = st.selectbox('Fasting blood sugar > 120 mg/dl', options=list(label_encoders['fbs'].classes_))
    restecg = st.selectbox('Resting ECG', options=list(label_encoders['restecg'].classes_))
    thalch = st.number_input('Max heart rate achieved', value=150)
    exang = st.selectbox('Exercise induced angina', options=list(label_encoders['exang'].classes_))
    oldpeak = st.number_input('ST depression', value=1.0, format="%.2f")
    slope = st.selectbox('ST slope', options=list(label_encoders['slope'].classes_))
    ca = st.number_input('Number of major vessels', min_value=0.0, max_value=3.0, value=0.0)
    thal = st.selectbox('Thalassemia', options=list(label_encoders['thal'].classes_))

    input_df = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalch': thalch,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }])

    if st.button('Predict'):
        X_proc = preprocess_input(input_df, label_encoders, scaler, feature_columns, categorical_cols, numerical_cols)
        model = models.get(model_name)
        pred, proba = predict_with_model(model, X_proc)
        if pred is None:
            st.error(f'Model {model_name} not available.')
        else:
            label = 'Heart Disease Present (1)' if int(pred[0]) == 1 else 'No Heart Disease (0)'
            st.subheader('Prediction')
            st.write(label)
            if proba is not None:
                st.write(f'Predicted probability of disease: {proba[0]:.3f}')

    if st.checkbox('Show predictions from all models'):
        X_proc = preprocess_input(input_df, label_encoders, scaler, feature_columns, categorical_cols, numerical_cols)
        rows = []
        for name, model in models.items():
            pred, proba = predict_with_model(model, X_proc)
            if pred is None:
                rows.append({'Model': name, 'Prediction': 'N/A', 'Probability': 'N/A'})
            else:
                rows.append({'Model': name, 'Prediction': int(pred[0]), 'Probability': float(proba[0]) if proba is not None else 'N/A'})
        st.table(pd.DataFrame(rows))
    
    # Batch prediction section
    st.header('Batch Prediction')
    st.write('Upload a CSV file with multiple patient records to get predictions for all.')
    uploaded_file = st.file_uploader('Choose CSV file', type='csv')
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f'Loaded {len(batch_df)} records.')
        
        if st.button('Run batch predictions'):
            predictions = []
            for idx, row in batch_df.iterrows():
                try:
                    sample = pd.DataFrame([row.to_dict()])
                    X_proc = preprocess_input(sample, label_encoders, scaler, feature_columns, categorical_cols, numerical_cols)
                    results_row = {'Record': idx + 1}
                    for name, model in models.items():
                        pred, proba = predict_with_model(model, X_proc)
                        if pred is not None:
                            results_row[f'{name} (Pred)'] = int(pred[0])
                            results_row[f'{name} (Prob)'] = round(float(proba[0]), 3) if proba is not None else None
                    predictions.append(results_row)
                except Exception as e:
                    st.warning(f'Error processing record {idx + 1}: {str(e)[:50]}')
            
            if predictions:
                pred_df = pd.DataFrame(predictions)
                st.subheader('Batch Predictions Results')
                st.dataframe(pred_df)
                
                csv_results = pred_df.to_csv(index=False)
                st.download_button(
                    label='Download predictions (CSV)',
                    data=csv_results,
                    file_name='batch_predictions.csv',
                    mime='text/csv'
                )

if __name__ == '__main__':
    streamlit_app()
