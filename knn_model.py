import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor  
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('cleaned_student_portuguese.csv', sep=',')
X = df.drop(columns=['G3'])
y = df['G3']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data has been scaled for KNN!......")

print("-----------------------------------------------")

# 1. Baseline KNN Model (before tuning)
print("Before tuning:")

knn_base = KNeighborsRegressor(n_neighbors=5)  
knn_base.fit(X_train_scaled, y_train)
base_pred = knn_base.predict(X_test_scaled)

mae_base = mean_absolute_error(y_test, base_pred)
mse_base = mean_squared_error(y_test, base_pred)
r2_base = r2_score(y_test, base_pred)

print(f"Initial MAE: {mae_base:.2f}")
print(f"Initial MSE: {mse_base:.2f}")
print(f"Initial R2 Score: {r2_base*100:.2f}%")

print("-----------------------------------------------")

# 2. Hyperparameter Tuning using GridSearchCV
print("Hyperparameter Tuning using GridSearchCV:")

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],  
    'weights': ['uniform', 'distance'],        
    'algorithm': ['auto', 'ball_tree', 'kd_tree'],  
    'p': [1, 2]                              
}

# GridSearchCV 
grid_search = GridSearchCV(
    estimator=KNeighborsRegressor(), 
    param_grid=param_grid, 
    cv=5,                   
    scoring='r2',           
    n_jobs=-1,              
    verbose=1               
)

print("Searching for best parameters............")
grid_search.fit(X_train_scaled, y_train)
best_knn_model = grid_search.best_estimator_

print("Best Parameters found:", grid_search.best_params_)
print(f"Best Cross-Validation R2 Score: {grid_search.best_score_:.4f}")

print("-----------------------------------------------")

# 3. Final Model Evaluation (After Tuning)
print("After Tuning:")
final_pred = best_knn_model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, final_pred)
mse = mean_squared_error(y_test, final_pred)
r2 = r2_score(y_test, final_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score (Accuracy): {r2*100:.2f}%")

print("-----------------------------------------------")

# 4. Overfitting / Underfitting Analysis
print("Overfitting / Underfitting:")
train_score = best_knn_model.score(X_train_scaled, y_train)
test_score = best_knn_model.score(X_test_scaled, y_test)

print(f"\nTraining Score (R2): {train_score*100:.2f}%")
print(f"Testing Score (R2): {test_score*100:.2f}%")
print(f"Gap: {(train_score - test_score)*100:.2f}%")

if train_score > test_score + 0.15:
    print("\n Result: Potential OVERFITTING!")
elif train_score < 0.5 and test_score < 0.5:
    print("\n Result: Potential UNDERFITTING!")
else:
    print("\n Result: Model is BALANCED and performing well!")

print("-----------------------------------------------")

# 5. Comparison: Before vs After Tuning
print("Before vs After Tuning")
print("1. Initial Model (Before Tuning):")
print(f"   Parameters: n_neighbors=5, weights='uniform'")
print(f"   MAE: {mae_base:.2f}")
print(f"   R2 Score: {r2_base*100:.2f}%")
print("-----------------------------------------------")
print("2. Final Model (After Tuning):")
print(f"   Parameters: {grid_search.best_params_}")
print(f"   MAE: {mae:.2f}")
print(f"   R2 Score: {r2*100:.2f}%")
print("-----------------------------------------------")


print("Before Scaling vs After Scaling")


# 1. KNN WITHOUT Scaling
knn_without_scaling = KNeighborsRegressor(n_neighbors=5)
knn_without_scaling.fit(X_train, y_train)
pred_without_scaling = knn_without_scaling.predict(X_test)

mae_without = mean_absolute_error(y_test, pred_without_scaling)
mse_without = mean_squared_error(y_test, pred_without_scaling)
r2_without = r2_score(y_test, pred_without_scaling)

print("KNN Without Scaling:")
print(f"MAE: {mae_without:.2f}")
print(f"MSE: {mse_without:.2f}")
print(f"R2 Score: {r2_without*100:.2f}%")

print("------------------------------------------------")

# 2. KNN WITH Scaling
knn_with_scaling = KNeighborsRegressor(n_neighbors=5)
knn_with_scaling.fit(X_train_scaled, y_train)
pred_with_scaling = knn_with_scaling.predict(X_test_scaled)

mae_with = mean_absolute_error(y_test, pred_with_scaling)
mse_with = mean_squared_error(y_test, pred_with_scaling)
r2_with = r2_score(y_test, pred_with_scaling)

print("KNN With Scaling:")
print(f"MAE: {mae_with:.2f}")
print(f"MSE: {mse_with:.2f}")
print(f"R2 Score: {r2_with*100:.2f}%")

print("------------------------------------------------")

# Comarison
print("Before Vs After Scaling:")

if r2_with > r2_without:
    print("Scaling improved the KNN model performance.")
else:
    print("Scaling did not improve the model significantly.")


print(f"R2 Before Scaling : {r2_without*100:.2f}%")
print(f"R2 After Scaling  : {r2_with*100:.2f}%")

print("------------------------------------------------")


print("Improvement:")
mae_improvement = ((mae_base - mae) / mae_base) * 100
r2_improvement = (r2 - r2_base) * 100

print(f"   MAE Improvement: {mae_improvement:+.2f}%")
print(f"   R2 Improvement: {r2_improvement:+.2f}%")

if r2 > r2_base:
    print("\n Tuning successfully improved model performance!")
else:
    print("\n Tuning didn't improve much. Default parameters might be sufficient.")

