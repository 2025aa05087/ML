from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


def train_all_models(X_train, y_train):
    models = {}
    
    models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=42)
    models['Logistic Regression'].fit(X_train, y_train)
    
    models['Decision Tree'] = DecisionTreeClassifier(random_state=42, max_depth=10)
    models['Decision Tree'].fit(X_train, y_train)
    
    models['KNN'] = KNeighborsClassifier(n_neighbors=5)
    models['KNN'].fit(X_train, y_train)
    
    models['Naive Bayes'] = GaussianNB()
    models['Naive Bayes'].fit(X_train, y_train)
    
    models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    models['Random Forest'].fit(X_train, y_train)
    
    models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, max_depth=10, use_label_encoder=False)
    models['XGBoost'].fit(X_train, y_train)
    
    return models
