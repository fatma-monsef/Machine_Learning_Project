import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('cleaned_student_portuguese.csv', sep=',')
X = df.drop(columns=['G3'])
y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")


rf_base = RandomForestRegressor(n_estimators=100, random_state=42)
rf_base.fit(X_train, y_train)


base_pred = rf_base.predict(X_test)


param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [10, 15, None],        
    'min_samples_split': [2, 5],
    'max_features': [None]             
}

# GridSearchCV هتقوم بتجربة كل الاحتمالات دي وتختار الأحسن
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), 
                           param_grid=param_grid, 
                           cv=5, 
                           scoring='r2', 
                           n_jobs=-1)

grid_search.fit(X_train, y_train)


best_rf_model = grid_search.best_estimator_

print("Best Parameters found:", grid_search.best_params_)


final_pred = best_rf_model.predict(X_test)

mae = mean_absolute_error(y_test, final_pred)
mse = mean_squared_error(y_test, final_pred)
r2 = r2_score(y_test, final_pred)

print("\n--- Model Performance Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} degrees")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score (Accuracy): {r2*100:.2f}%")


train_score = best_rf_model.score(X_train, y_train)
test_score = best_rf_model.score(X_test, y_test)

print(f"\nTraining Score: {train_score*100:.2f}%")
print(f"Testing Score: {test_score*100:.2f}%")

if train_score > test_score + 0.15:
    print("Result: Potential Overfitting ")
elif train_score < 0.5:
    print("Result: Potential Underfitting ")
else:
    print("Result: Model is balanced and performing well! ")



print("\n--- 1. Initial Model (Before Tuning) ---")
mae_base = mean_absolute_error(y_test, base_pred)
r2_base = r2_score(y_test, base_pred)
print(f"Initial MAE: {mae_base:.2f}")
print(f"Initial R2 Score: {r2_base*100:.2f}%")

print("\n--- 2. Final Model (After Tuning) ---")
print(f"Final MAE: {mae:.2f}")
print(f"Final R2 Score: {r2*100:.2f}%")   




# y_pred = grid_search.predict(X_test)

# plt.figure(figsize=(10, 6))

# sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='#3498db', edgecolor='w', s=80)

# line_coords = [y_test.min(), y_test.max()]
# plt.plot(line_coords, line_coords, color='#e74c3c', lw=3, linestyle='--')

# plt.title('Actual vs Predicted Grades', fontsize=16, fontweight='bold')
# plt.xlabel('Actual Final Grades (G3)', fontsize=13)
# plt.ylabel('Predicted Final Grades (G3)', fontsize=13)
# plt.grid(True, linestyle=':', alpha=0.6)

# plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')


# plt.show()