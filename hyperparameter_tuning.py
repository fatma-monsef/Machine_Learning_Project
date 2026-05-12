import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('cleaned_student_portuguese.csv', sep=',')
X = df.drop(columns=['G3'])
y = df['G3']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)


# KNN 

knn_params = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2] 
}

knn_base = KNeighborsRegressor()
knn_grid = GridSearchCV(
    estimator=knn_base,
    param_grid=knn_params,
    cv=5, 
    scoring='r2',  
    n_jobs=-1,
    verbose=1
)

knn_grid.fit(X_train_std, y_train)

print(f"\n Best Parameters for KNN: {knn_grid.best_params_}")
print(f" Best Cross-Validation R^2 Score: {knn_grid.best_score_:.4f}")

best_knn = knn_grid.best_estimator_
y_pred_knn = best_knn.predict(X_test_std)

print(f"\n KNN Test Set Performance After Tuning:")
print(f"   MAE: {mean_absolute_error(y_test, y_pred_knn):.4f}")
print(f"   MSE: {mean_squared_error(y_test, y_pred_knn):.4f}")
print(f"   R^2 Score: {r2_score(y_test, y_pred_knn):.4f}")

print("----------------------------------------------------")

# SVM
svr_params = {
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'C': [0.1, 1, 10, 100, 1000],
    'epsilon': [0.01, 0.1, 0.5, 1],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}

svr_base = SVR()
svr_random = RandomizedSearchCV(
    estimator=svr_base,
    param_distributions=svr_params,
    n_iter=50, 
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

svr_random.fit(X_train_std, y_train)

print(f"\n Best Parameters for SVR: {svr_random.best_params_}")
print(f" Best Cross-Validation R^2 Score: {svr_random.best_score_:.4f}")


best_svr = svr_random.best_estimator_
y_pred_svr = best_svr.predict(X_test_std)

print(f"\n SVR Test Set Performance After Tuning:")
print(f"   MAE: {mean_absolute_error(y_test, y_pred_svr):.4f}")
print(f"   MSE: {mean_squared_error(y_test, y_pred_svr):.4f}")
print(f"   R^2 Score: {r2_score(y_test, y_pred_svr):.4f}")

print("----------------------------------------------------")

# # RandomForest
# rf_params_phase1 = {
#     'n_estimators': [50, 100, 200, 300],
#     'max_depth': [None, 10, 20, 30, 50],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2', None]
# }

# rf_base = RandomForestRegressor(random_state=42)

# rf_random = RandomizedSearchCV(
#     estimator=rf_base,
#     param_distributions=rf_params_phase1,
#     n_iter=30,
#     cv=5,
#     scoring='r2',
#     n_jobs=-1,
#     verbose=1,
#     random_state=42
# )

# rf_random.fit(X_train_std, y_train)

# print(f"\n Best Parameters (Phase 1): {rf_random.best_params_}")
# print(f" Best CV R^2 (Phase 1): {rf_random.best_score_:.4f}")


# best_params = rf_random.best_params_


# rf_params_phase2 = {
#     'n_estimators': [max(50, best_params['n_estimators'] - 50), 
#                      best_params['n_estimators'], 
#                      best_params['n_estimators'] + 50],
#     'max_depth': [best_params['max_depth'] - 5 if best_params['max_depth'] else None,
#                   best_params['max_depth'],
#                   best_params['max_depth'] + 5 if best_params['max_depth'] else None],
#     'min_samples_split': [max(2, best_params['min_samples_split'] - 1),
#                           best_params['min_samples_split'],
#                           best_params['min_samples_split'] + 1],
#     'min_samples_leaf': [max(1, best_params['min_samples_leaf'] - 1),
#                          best_params['min_samples_leaf'],
#                          best_params['min_samples_leaf'] + 1]
# }


# rf_params_phase2['max_depth'] = [d for d in rf_params_phase2['max_depth'] if d is not None or d == None]
# rf_params_phase2['max_depth'] = list(set(rf_params_phase2['max_depth']))

# rf_grid = GridSearchCV(
#     estimator=RandomForestRegressor(random_state=42),
#     param_grid=rf_params_phase2,
#     cv=5,
#     scoring='r2',
#     n_jobs=-1,
#     verbose=1
# )

# rf_grid.fit(X_train_std, y_train)

# print(f"\n Best Parameters (Phase 2 - Fine Tuning): {rf_grid.best_params_}")
# print(f" Best CV R^2 (Phase 2): {rf_grid.best_score_:.4f}")


# best_rf = rf_grid.best_estimator_
# y_pred_rf = best_rf.predict(X_test_std)

# print(f"\n Random Forest Test Set Performance After Tuning:")
# print(f"   MAE: {mean_absolute_error(y_test, y_pred_rf):.4f}")
# print(f"   MSE: {mean_squared_error(y_test, y_pred_rf):.4f}")
# print(f"   R^2 Score: {r2_score(y_test, y_pred_rf):.4f}")