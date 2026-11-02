📌 Project Overview

Fine particulate matter (PM2.5) is an important indicator of air pollution. This project uses air-quality station records to explore PM2.5 patterns and build machine-learning models for two tasks:

Regression: Estimate pollutant_avg using location and pollutant summary features.

Classification: Assign an air-quality category based on the average PM2.5 value.

The project demonstrates a complete introductory predictive-analytics workflow, from data preparation and visualization to model evaluation.

🎯 Objectives

Explore air-pollution data and identify PM2.5 records.

Inspect missing values, distributions, and city-level patterns.

Prepare numerical features for machine-learning models.

Compare regression models using standard evaluation metrics.

Compare classification models using accuracy and F1-score.

Present the findings and limitations of the models.

🗂️ Project Structure

PM25-Air-Pollution-Prediction-Using-Machine-Learning/
│
├── PM25_Simplified_Code.py     # Main Python project
├── requirements.txt            # Required Python libraries
├── README.md                   # Project documentation
│
└── results/                    # Optional: charts and output screenshots

📊 Dataset

The project uses an air-pollution dataset containing station-level pollutant measurements and location information.

Detail

Value

Original dataset records

3,808

Original dataset columns

11

PM2.5 records used

472

Duplicate records identified

0

Missing values noted

430 each in pollutant_min, pollutant_max, and pollutant_avg

The analysis focuses on the PM2.5 subset. The dataset file is not included in this repository; place the CSV file on your computer and update its path in the Python script.

Dataset fields used by the models:

latitude

longitude

pollutant_min

pollutant_max

pollutant_avg (regression target)

Please retain the original dataset attribution and follow its usage license or terms when using or sharing the data.

⚙️ Workflow

Load the CSV dataset.

Inspect the dataset shape, columns, missing values, and duplicates.

Filter records where the pollutant is PM2.5.

Perform exploratory data analysis (EDA).

Prepare features and handle missing numerical values using median imputation.

Scale regression features using MinMaxScaler.

Train and evaluate regression models.

Create PM2.5-based air-quality categories.

Train and evaluate classification models.

Compare model performance.

📈 Exploratory Data Analysis

The PM2.5 subset contains 472 records.

PM2.5 category distribution

Category

Number of records

Good

134

Satisfactory

199

Moderate

96

Poor

27

Very Poor

16

Severe

0

Highest city-level average PM2.5 values

City

Average PM2.5

Gummidipoondi

195

Rajkot

193

Amritsar

181

Khanna

177

Baddi

174

Chittorgarh

156

Mandi Gobindgarh

153

Jhalawar

146

Byrnihat

124.5

Sonipat

124

These are descriptive results from the dataset used in this project, not live air-quality readings.

🤖 Machine-Learning Models

Regression

The regression task estimates pollutant_avg.

Model

RMSE

MAE

R²

Multiple Linear Regression

9.51

6.35

0.895

Polynomial Regression (degree 2)

17.04

13.63

0.662

Best regression result: Multiple Linear Regression, with an R² score of approximately 0.895 on the test split.

Classification

The classification task predicts the air-quality category.

Model

Accuracy

Macro F1

Weighted F1

Logistic Regression

65.26%

0.359

0.595

Decision Tree

72.63%

0.547

0.711

Support Vector Machine (linear kernel)

69.47%

0.416

0.658

Best classification result: Decision Tree, with 72.63% accuracy and a weighted F1-score of 0.711.

Results are based on the project's train-test split and may vary with changes to the dataset, package versions, or execution settings.

🧰 Tools & Libraries

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

🚀 Run the Project

1. Clone the repository

git clone https://github.com/shingneshubham/PM25-Air-Pollution-Prediction-Using-Machine-Learning.git
cd PM25-Air-Pollution-Prediction-Using-Machine-Learning

2. Install dependencies

pip install -r requirements.txt

3. Add the dataset

Place air_pollution.csv in your preferred local folder.

Open PM25_Simplified_Code.py and update the CSV path to match your computer. For example:

data = pd.read_csv(r"C:\Users\YourName\Downloads\air_pollution.csv")

Use the actual filename and location of your dataset.

4. Run the script

python PM25_Simplified_Code.py

The script performs the analysis, trains the models, prints evaluation results, and displays the project visualizations.

⚠️ Limitations

The PM2.5 subset contains 472 records, which is relatively small for building generalizable models.

The classification dataset is imbalanced; the Poor and Very Poor categories have fewer examples, and the Severe category has no examples.

The regression model uses pollutant_min and pollutant_max to estimate pollutant_avg. These are related summaries from the same measurement record.

This is not a future air-quality forecasting system. The project does not use time-series, weather, traffic, or emissions data to predict future PM2.5 levels.

Evaluation is based on a single train-test split, so results may vary across different splits.

🔭 Future Improvements

Add time-series and weather features for forecasting.

Use a larger dataset covering more cities and dates.

Apply cross-validation and hyperparameter tuning.

Address class imbalance and evaluate per-class performance.

Compare additional machine-learning models.

Build an interactive dashboard for exploring air-quality patterns.

👨‍🎓 Author

Shubham Shingne
B.Tech – Computer Science and Engineering (Data Science)

🙏 Acknowledgment

Developed as part of the INT234 – Predictive Analytics coursework.
