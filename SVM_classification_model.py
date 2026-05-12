import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('cleaned_student_portuguese.csv')

# Classification 
df['G3_Class'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# X and y definition
X = df.drop(columns=['G3', 'G3_Class'])
y = df['G3_Class']

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base Model
svc_base = SVC()
svc_base.fit(X_train, y_train)

# Tuning Parameters
param_grid = {
    'kernel': ['rbf', 'linear', 'poly'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto']
}

# Hyperparameter Tuning
grid_search = GridSearchCV(estimator=SVC(), 
                           param_grid=param_grid, 
                           cv=5, 
                           scoring='accuracy', 
                           n_jobs=-1)

grid_search.fit(X_train, y_train)

# Final Model
best_svc_model = grid_search.best_estimator_
final_pred = best_svc_model.predict(X_test)

# Evaluation
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, final_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, final_pred))

print(f"\nFinal Accuracy: {accuracy_score(y_test, final_pred)*100:.2f}%")

# Overfitting Check
train_score = best_svc_model.score(X_train, y_train)
test_score = best_svc_model.score(X_test, y_test)

if train_score > test_score + 0.15:
    print("Result: Potential Overfitting")
else:
    print("Result: Model is balanced!")

# Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, final_pred), annot=True, fmt='d', cmap='Purples')
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()