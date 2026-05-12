import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('cleaned_student_portuguese.csv')

# Features (X) and Target (y)
X = df.drop(columns=['G3'])
y = df['G3']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"SVM Model - Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Before Tuning
svm_base = SVR()
svm_base.fit(X_train, y_train)
base_pred = svm_base.predict(X_test)

# Hyperparameter Tuning
param_grid = {
    'kernel': ['rbf', 'linear', 'poly'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'epsilon': [0.1, 0.2, 0.5]
}

print("\nStarting Hyperparameter Tuning for SVM... please wait.")
grid_search = GridSearchCV(estimator=SVR(), 
                           param_grid=param_grid, 
                           cv=5, 
                           scoring='r2', 
                           n_jobs=-1)

grid_search.fit(X_train, y_train)

# best model
best_svm_model = grid_search.best_estimator_
print("Best Parameters found:", grid_search.best_params_)

# Evaluation
final_pred = best_svm_model.predict(X_test)

mae = mean_absolute_error(y_test, final_pred)
mse = mean_squared_error(y_test, final_pred)
r2 = r2_score(y_test, final_pred)

print("\n--- SVM Model Performance Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score (Accuracy): {r2*100:.2f}%")

# Check Overfitting / Underfitting
train_score = best_svm_model.score(X_train, y_train)
test_score = best_svm_model.score(X_test, y_test)

print(f"\nTraining Score: {train_score*100:.2f}%")
print(f"Testing Score: {test_score*100:.2f}%")

if train_score > test_score + 0.15:
    print("Result: Potential Overfitting")
elif train_score < 0.5:
    print("Result: Potential Underfitting")
else:
    print("Result: SVM Model is balanced!")

# Before vs After Tuning
print("\n--- Improvement Comparison ---")
r2_base = r2_score(y_test, base_pred)
print(f"Initial R2 Score: {r2_base*100:.2f}%")
print(f"Final R2 Score: {r2*100:.2f}%")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_test, final_pred, color='purple', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('SVM: Actual vs Predicted Grades')
plt.xlabel('Actual G3')
plt.ylabel('Predicted G3')
plt.show()