import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, confusion_matrix, classification_report)

# 1. LOAD AND PREPARE DATA
df = pd.read_csv(r"C:\Users\shubh\Downloads\air_pollution.csv")

print(df.head())
print(df.info())
print(df.describe())
print("Missing values:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()

num_cols = ["latitude", "longitude", "pollutant_min", "pollutant_max", "pollutant_avg"]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

pm25 = df[df["pollutant_id"] == "PM2.5"].dropna(subset=["pollutant_avg"]).copy()

print("\nPM2.5 Dataset:", pm25.shape)
print(pm25["pollutant_avg"].describe())

# 2. EXPLORATORY DATA ANALYSIS
city_pollution = pm25.groupby("city")["pollutant_avg"].mean().sort_values(ascending=False)
print("\nTop 10 Cities:\n", city_pollution.head(10))

plt.figure(figsize=(8, 5))
sns.histplot(pm25["pollutant_avg"], bins=30, kde=True)
plt.title("Distribution of PM2.5")
plt.xlabel("PM2.5 Average")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 5))
city_pollution.head(10).plot(kind="bar")
plt.title("Top 10 Cities by Average PM2.5")
plt.xlabel("City")
plt.ylabel("Average PM2.5")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
sns.boxplot(x=pm25["pollutant_avg"])
plt.title("PM2.5 Concentration Boxplot")
plt.xlabel("PM2.5 Average")
plt.show()

plt.figure(figsize=(7, 5))
sns.heatmap(pm25[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 3. MULTIPLE LINEAR REGRESSION
features = ["latitude", "longitude", "pollutant_min", "pollutant_max"]
X = pm25[features]
y = pm25["pollutant_avg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nMULTIPLE LINEAR REGRESSION")
print("MSE:", mse, "\nRMSE:", rmse, "\nR2 Score:", r2, "\nMAE:", mae)

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Multiple Linear Regression")
plt.show()

# 4. POLYNOMIAL REGRESSION
X_poly = pm25[["pollutant_max"]]
Y_poly = pm25["pollutant_avg"]

X_train_p, X_test_p, Y_train_p, Y_test_p = train_test_split(
    X_poly, Y_poly, test_size=0.2, random_state=42)

imputer_poly = SimpleImputer(strategy="median")
X_train_p = imputer_poly.fit_transform(X_train_p)
X_test_p = imputer_poly.transform(X_test_p)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_p)
X_test_poly = poly.transform(X_test_p)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, Y_train_p)
poly_pred = poly_model.predict(X_test_poly)

mse_poly = mean_squared_error(Y_test_p, poly_pred)
rmse_poly = np.sqrt(mse_poly)
r2_poly = r2_score(Y_test_p, poly_pred)
mae_poly = mean_absolute_error(Y_test_p, poly_pred)

print("\nPOLYNOMIAL REGRESSION")
print("MSE:", mse_poly, "\nRMSE:", rmse_poly, "\nR2 Score:", r2_poly, "\nMAE:", mae_poly)

sort_index = np.argsort(X_test_p[:, 0])
plt.figure(figsize=(8, 5))
plt.scatter(X_test_p, Y_test_p, label="Actual Data")
plt.plot(X_test_p[sort_index], poly_pred[sort_index], color="red", label="Polynomial Fit")
plt.xlabel("PM2.5 Maximum")
plt.ylabel("PM2.5 Average")
plt.title("Polynomial Regression")
plt.legend()
plt.show()

# 5. CLASSIFICATION DATA
bins = [-np.inf, 30, 60, 90, 120, 250, np.inf]
labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

pm25["pollution_category"] = pd.cut(pm25["pollutant_avg"], bins=bins, labels=labels)
print("\nPollution Category:\n", pm25["pollution_category"].value_counts())

X_class = pm25[features]
y_class = pm25["pollution_category"].astype(str)

# Keep the same stratified split logic as the original code
stratify_value = y_class if y_class.value_counts().min() >= 2 else None
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=stratify_value)

imputer_c = SimpleImputer(strategy="median")
X_train_c = imputer_c.fit_transform(X_train_c)
X_test_c = imputer_c.transform(X_test_c)

scaler_c = MinMaxScaler()
X_train_c = scaler_c.fit_transform(X_train_c)
X_test_c = scaler_c.transform(X_test_c)

# Common function for classifier results and confusion matrix
def show_results(name, model, color, rotate=False):
    model.fit(X_train_c, y_train_c)
    pred = model.predict(X_test_c)
    acc = accuracy_score(y_test_c, pred)

    print("\n" + name)
    print("Accuracy:", acc)
    print("Classification Report:\n",
          classification_report(y_test_c, pred, zero_division=0))

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test_c, pred, labels=labels),
                annot=True, fmt="d", cmap=color,
                xticklabels=labels, yticklabels=labels)
    plt.title(name + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if rotate:
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    return pred, acc

# 6. LOGISTIC REGRESSION
log_model = LogisticRegression(max_iter=1000)
log_pred, acc = show_results("LOGISTIC REGRESSION", log_model, "Blues", rotate=True)

# 7. DECISION TREE
dt_model = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
dt_pred, dt_accuracy = show_results("DECISION TREE RESULTS", dt_model, "Greens")

plt.figure(figsize=(16, 8))
plot_tree(dt_model, feature_names=features, class_names=dt_model.classes_, filled=True)
plt.title("Decision Tree for Air Pollution")
plt.show()

# 8. SUPPORT VECTOR MACHINE
svm_model = SVC(kernel="linear")
svm_pred, svm_accuracy = show_results("SUPPORT VECTOR MACHINE", svm_model, "Oranges")

# 9. MODEL COMPARISON
regression_result = pd.DataFrame({
    "Model": ["Multiple Linear Regression", "Polynomial Regression"],
    "MSE": [mse, mse_poly],
    "RMSE": [rmse, rmse_poly],
    "MAE": [mae, mae_poly],
    "R2 Score": [r2, r2_poly]
})
print("\nREGRESSION MODEL COMPARISON\n", regression_result)

classification_result = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "SVM"],
    "Accuracy": [acc, dt_accuracy, svm_accuracy]
})
print("\nCLASSIFICATION MODEL COMPARISON\n", classification_result)

plt.figure(figsize=(8, 5))
sns.barplot(data=regression_result, x="Model", y="R2 Score")
plt.title("Regression Model Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(data=classification_result, x="Model", y="Accuracy")
plt.title("Classification Model Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

print("\nINT234 PROJECT COMPLETED")
print("Project: Air Pollution Analysis and Prediction")
print("Target: PM2.5 Concentration")
print("Models: Multiple Linear Regression, Polynomial Regression,")
print("        Logistic Regression, Decision Tree and SVM")
print("Project Execution Completed.")
