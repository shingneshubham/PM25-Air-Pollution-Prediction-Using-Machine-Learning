PM2.5 Air Pollution Analysis and Prediction Using Machine Learning

An academic Predictive Analytics project that explores PM2.5 air pollution measurements across Indian cities and applies machine learning models for concentration estimation and air-quality category classification.

Project Overview

Fine particulate matter (PM2.5) is an important air-quality indicator. This project analyzes PM2.5 records, explores city-wise pollution patterns, and compares regression and classification algorithms.

Course: INT234 – Predictive Analytics
Project title: Prediction of Fine Particulate Matter Concentration and Air Quality Category in Indian Cities Using Machine Learning

Objectives

Clean and prepare the air-pollution dataset.

Explore PM2.5 concentration patterns.

Compare average PM2.5 levels across cities.

Study relationships between numerical features.

Build and evaluate regression models.

Classify PM2.5 observations into air-quality categories.

Compare model performance using suitable evaluation metrics.

Dataset

The project uses the file air_pollution.csv, containing air-pollution station records. The analysis filters records where pollutant_id is PM2.5.

The working PM2.5 subset contains 472 rows after filtering and removing rows with a missing pollutant_avg value.

Expected columns used in the analysis:

city

latitude

longitude

pollutant_id

pollutant_min

pollutant_max

pollutant_avg

Add the exact dataset source, download page, license, and attribution here before publishing the dataset. Do not upload the raw CSV unless its license and sharing terms permit redistribution.

Workflow

Load and inspect the dataset.

Check missing values and duplicate records.

Remove duplicate rows and convert numerical columns to numeric values.

Filter PM2.5 observations and remove rows without pollutant_avg.

Perform exploratory data analysis (EDA).

Train regression models.

Create air-quality categories and train classification models.

Evaluate and compare the models.

Exploratory Data Analysis

The notebook/script produces the following visualizations:

PM2.5 concentration distribution

Top 10 cities by average PM2.5

PM2.5 concentration boxplot

Correlation heatmap

Selected EDA Findings

PM2.5 subset size: 472 records

Mean PM2.5 average: 50.42

Median PM2.5 average: 46

Minimum PM2.5 average: 2

Maximum PM2.5 average: 195

Top cities by mean PM2.5 in this dataset include Gummidipoondi, Rajkot, Amritsar, Khanna, and Baddi.

Machine Learning Models

Regression

The target variable is pollutant_avg. The input features are:

latitude

longitude

pollutant_min

pollutant_max

Models:

Multiple Linear Regression

Polynomial Regression (degree 2, using pollutant_max)

Model

MSE

RMSE

MAE

R² Score

Multiple Linear Regression

90.44

9.51

6.35

0.8946

Polynomial Regression

290.43

17.04

13.63

0.6615

Multiple Linear Regression achieved the higher R² score on the selected test split.

Classification

PM2.5 observations are grouped using these project-defined intervals:

PM2.5 interval

Project category

Up to 30

Good

30–60

Satisfactory

60–90

Moderate

90–120

Poor

120–250

Very Poor

Above 250

Severe

Models:

Logistic Regression

Decision Tree Classifier

Support Vector Machine (linear kernel)

Model

Accuracy

Logistic Regression

65.26%

Decision Tree

72.63%

Support Vector Machine (SVM)

69.47%

The Decision Tree Classifier achieved the highest accuracy among the three classification models on the selected test split.

Evaluation and Visualizations

The project includes:

Regression metrics: MSE, RMSE, MAE, and R² score

Classification accuracy and classification reports

Confusion matrices for classification models

Actual-versus-predicted scatter plot

Polynomial regression fit plot

Decision Tree visualization

Regression and classification model comparison charts

Important Limitations

The regression model estimates pollutant_avg using pollutant_min and pollutant_max, which are related summaries of the same pollution measurements. This is not a genuine future-time forecast.

The available PM2.5 subset is relatively small.

Some air-quality categories have few or no observations. In particular, the Severe category has no observations in this subset, so its performance cannot be meaningfully evaluated.

Results are based on the project's train-test split and may vary with dataset or library versions.

The project does not include time-series forecasting or weather-related input features.

Technologies and Libraries

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Project Files

Suggested repository structure:

PM25-Air-Pollution-Prediction-Using-Machine-Learning/
│
├── PM25_Simplified_Code.py
├── air_pollution.csv             # Include only if redistribution is permitted
├── requirements.txt
├── README.md
└── results/
    ├── pm25_distribution.png
    ├── top_10_cities.png
    ├── correlation_heatmap.png
    ├── regression_comparison.png
    ├── classification_comparison.png
    ├── logistic_confusion_matrix.png
    ├── decision_tree_confusion_matrix.png
    ├── svm_confusion_matrix.png
    └── decision_tree.png

Use the actual filenames of your saved images. Remove any entries that you do not upload.

Installation and Usage

1. Clone the repository

git clone https://github.com/YOUR-USERNAME/PM25-Air-Pollution-Prediction-Using-Machine-Learning.git
cd PM25-Air-Pollution-Prediction-Using-Machine-Learning

2. Install dependencies

pip install -r requirements.txt

3. Add the dataset

Place air_pollution.csv in the project folder, or update the CSV path in PM25_Simplified_Code.py to match its location.

4. Run the project

python PM25_Simplified_Code.py

Future Improvements

Add weather features such as temperature, humidity, and wind speed.

Use time-stamped observations for genuine forecasting.

Evaluate models with cross-validation.

Address class imbalance and collect more examples for rare categories.

Test additional algorithms and compare their performance.

Author

Shubham Shingne
B.Tech – Computer Science and Engineering (Data Science)
Lovely Professional University

Acknowledgment

This project was completed as part of the INT234 Predictive Analytics coursework. Thanks to the faculty and learning resources that supported the project.

If you use or extend this project, please verify the dataset source, license, and attribution requirements before redistributing the data.
