import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv('cleaned_student_portuguese.csv', sep=',')

df['G3'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

X = df.drop(columns=['G3'])
y = df['G3']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
rf_base.fit(X_train, y_train)
base_pred = rf_base.predict(X_test)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 5],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                           param_grid=param_grid, 
                           cv=5, 
                           scoring='accuracy', 
                           n_jobs=-1)

grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
print("\nBest Parameters found:", grid_search.best_params_)

final_pred = best_rf_model.predict(X_test)

print("\n--- Model Performance Evaluation ---")
acc = accuracy_score(y_test, final_pred)
print(f"Accuracy Score: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, final_pred))


train_acc = best_rf_model.score(X_train, y_train)
test_acc = best_rf_model.score(X_test, y_test)

print(f"Training Accuracy: {train_acc*100:.2f}%")
print(f"Testing Accuracy: {test_acc*100:.2f}%")

if train_acc > test_acc + 0.10:
    print("Result: Potential Overfitting ")
elif train_acc < 0.6:
    print("Result: Potential Underfitting ")
else:
    print("Result: Model is balanced and performing well! ")


# plt.figure(figsize=(8, 6))
# cm = confusion_matrix(y_test, final_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Random Forest Confusion Matrix')

plt.savefig('random_forest_confusion_matrix.png')
print("\n Confusion Matrix image saved as 'random_forest_confusion_matrix.png'")

plt.show()

print("\n--- Comparison ---")
acc_base = accuracy_score(y_test, base_pred)
print(f"Initial Accuracy (Before Tuning): {acc_base*100:.2f}%")
print(f"Final Accuracy (After Tuning): {acc*100:.2f}%")