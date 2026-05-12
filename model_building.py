import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



df = pd.read_csv('cleaned_student_portuguese.csv', sep=',')
X = df.drop(columns=['G3'])
y = df['G3']


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)



# SVM
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_std, y_train)
y_pred_svr = svr_model.predict(X_test_std)

print("SVR result :")

mae_svr = mean_absolute_error(y_test, y_pred_svr)  
mse_svr = mean_squared_error(y_test, y_pred_svr)  
r2_svr = r2_score(y_test, y_pred_svr) 


print(f"Mean Absolute Error (MAE): {mae_svr}")
print(f"Mean Squared Error (MSE): {mse_svr}")
print(f"R-squared (R^2): {r2_svr}")

print("--------------------------------------------------------")

# KNN
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_std, y_train)
y_pred_knn = knn_model.predict(X_test_std)

print("KNN result :")

mae_knn = mean_absolute_error(y_test, y_pred_knn)
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)

print(f"Mean Absolute Error (MAE): {mae_knn}")
print(f"Mean Squared Error (MSE): {mse_knn}")
print(f"R-squared (R^2): {r2_knn}")

print("--------------------------------------------------------")

# RF
rf_model = RandomForestRegressor(n_estimators=100,random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("rf result :")


mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Mean Absolute Error (MAE): {mae_rf}")
print(f"Mean Squared Error (MSE): {mse_rf}")
print(f"R-squared (R^2): {r2_rf}")

print("--------------------------------------------------------")

print("SVR R2 =", r2_svr)

print("KNN R2 =", r2_knn)

print("Random Forest R2 =", r2_rf)