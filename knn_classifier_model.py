
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('cleaned_student_portuguese.csv', sep=',')

X = df.drop(columns=['G3'])
y_continuous = df['G3']


y = (y_continuous >= 10).astype(int)

print(f"Target Distribution:")
print(f"Fail (0): {(y == 0).sum()} students ({(y == 0).mean()*100:.1f}%)")
print(f"Pass (1): {(y == 1).sum()} students ({(y == 1).mean()*100:.1f}%)")

print("------------------------------------------------------")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Data Split:")
print(f"Training: {X_train.shape[0]} samples")
print(f"Testing: {X_test.shape[0]} samples")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data scaling completed")

print("------------------------------------------------------")

# BASELINE MODEL (Before Tuning)

knn_baseline = KNeighborsClassifier(n_neighbors=5)
knn_baseline.fit(X_train_scaled, y_train)

y_pred_baseline = knn_baseline.predict(X_test_scaled)

accuracy_baseline = accuracy_score(y_test, y_pred_baseline)
precision_baseline = precision_score(y_test, y_pred_baseline)
recall_baseline = recall_score(y_test, y_pred_baseline)
f1_baseline = f1_score(y_test, y_pred_baseline)
cm_baseline = confusion_matrix(y_test, y_pred_baseline)

print(f"BASELINE KNN PERFORMANCE (n_neighbors=5):")
print(f"Accuracy:{accuracy_baseline:.4f}")
print(f"Precision:{precision_baseline:.4f}")
print(f"Recall:{recall_baseline:.4f}")
print(f"F1-Score:{f1_baseline:.4f}")

print(f"Confusion Matrix:")
print(f"[{cm_baseline[0,0]:3d}{cm_baseline[0,1]:3d}]")
print(f"[{cm_baseline[1,0]:3d}{cm_baseline[1,1]:3d}]")

print("------------------------------------------------------")

# HYPERPARAMETER TUNING (Grid Search)

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

print("Searching for best parameters...")
print(f"   Parameters to test: {len(param_grid['n_neighbors']) * len(param_grid['weights']) * len(param_grid['metric'])} combinations")


knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid=param_grid,
    cv=5,                    
    scoring='f1',           
    n_jobs=-1,
    verbose=1
)

knn_grid.fit(X_train_scaled, y_train)

print(f"Best Parameters Found:{knn_grid.best_params_}")
print(f"Best Cross-Validation F1-Score:{knn_grid.best_score_:.4f}")

best_knn = knn_grid.best_estimator_

print("------------------------------------------------------")

# EVALUATION (After Tuning)

y_pred_tuned = best_knn.predict(X_test_scaled)
y_pred_train = best_knn.predict(X_train_scaled)

accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
precision_tuned = precision_score(y_test, y_pred_tuned)
recall_tuned = recall_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned)
cm_tuned = confusion_matrix(y_test, y_pred_tuned)

train_accuracy = accuracy_score(y_train, y_pred_train)

print(f"TEST SET PERFORMANCE (After Tuning):")
print(f"Accuracy:{accuracy_tuned:.4f}")
print(f"Precision:{precision_tuned:.4f}")
print(f"Recall:{recall_tuned:.4f}")
print(f"F1-Score:{f1_tuned:.4f}")

print(f"TRAINING SET PERFORMANCE:")
print(f"Accuracy:{train_accuracy:.4f}")

print(f"Confusion Matrix:")

print(f"Actual Fail: {cm_tuned[0,0]:3d}{cm_tuned[0,1]:3d}")
print(f"Actual Pass: {cm_tuned[1,0]:3d}{cm_tuned[1,1]:3d}")


print(f"Detailed Classification Report:")
print(classification_report(y_test, y_pred_tuned, target_names=['Fail (0)', 'Pass (1)']))

print("------------------------------------------------------")

# OVERFITTING / UNDERFITTING ANALYSIS

gap = train_accuracy - accuracy_tuned

print(f"Training Accuracy:{train_accuracy:.4f}")
print(f"Accuracy:{accuracy_tuned:.4f}")
print(f"Gap:{gap:.4f}")

if gap > 0.1:
    print(f"VERDICT: OVERFITTING!")
    print(f"   The model memorized training data (Gap = {gap:.4f})")
    print(f"   Suggestion: Increase n_neighbors or use 'uniform' weights")
elif train_accuracy < 0.7 and accuracy_tuned < 0.7:
    print(f"VERDICT: UNDERFITTING!")
    print(f"   Model is too simple (Low accuracy on both sets)")
    print(f"   Suggestion: Decrease n_neighbors or add more features")
elif gap < -0.05:
    print(f"VERDICT: Possible UNDERFITTING")
    print(f"   Unusual pattern (Test better than Train)")
else:
    print(f" VERDICT: GOOD GENERALIZATION!")
    print(f"   Model is balanced and performing well")

print("------------------------------------------------------")

# IMPROVEMENT & EXPERIMENTATION

# Before vs After Tuning

print(f"F1-Score:\n")
print(f"Before Tuning:{f1_baseline:.4f}")
print(f"After Tuning:{f1_tuned:.4f}")
print(f"Improvement:{(f1_tuned - f1_baseline)*100:+.2f}%")

print(f"Accuracy:\n")
print(f"Before Tuning:{accuracy_baseline:.4f}")
print(f"After Tuning:{accuracy_tuned:.4f}")
print(f"Improvement:{(accuracy_tuned - accuracy_baseline)*100:+.2f}%")

print("------------------------------------------------------")

# Effect of Scaling (KNN is sensitive to scaling)
# Train without scaling
knn_no_scale = KNeighborsClassifier(n_neighbors=best_knn.n_neighbors)
knn_no_scale.fit(X_train, y_train)  
y_pred_no_scale = knn_no_scale.predict(X_test)
acc_no_scale = accuracy_score(y_test, y_pred_no_scale)

print(f"Accuracy WITHOUT Scaling:{acc_no_scale:.4f}")
print(f"Accuracy WITH Scaling:{accuracy_tuned:.4f}")
print(f"Improvement from Scaling:{(accuracy_tuned - acc_no_scale)*100:+.2f}%")

if accuracy_tuned > acc_no_scale:
    print("Scaling significantly improved KNN performance!")
else:
    print("Scaling didn't help much for this dataset")

print("------------------------------------------------------")

# Experiment with different K values

print("EXPERIMENT: Different K Values Impact:")

k_values = [3, 5, 7, 9, 11, 13, 15]
k_results = []

for k in k_values:
    knn_test = KNeighborsClassifier(n_neighbors=k, weights=best_knn.weights, metric=best_knn.metric)
    knn_test.fit(X_train_scaled, y_train)
    y_pred_test = knn_test.predict(X_test_scaled)
    f1_test = f1_score(y_test, y_pred_test)
    k_results.append(f1_test)
    print(f"K = {k:2d} : F1-Score = {f1_test:.4f}")

best_k_from_exp = k_values[k_results.index(max(k_results))]
print(f"Best K from experiment: {best_k_from_exp}")
print(f"Best K from GridSearch: {best_knn.n_neighbors}")
