Sleep Quality Prediction Using Machine Learning
Project Overview
This project aims to predict sleep quality using machine learning based on lifestyle factors.
Sleep quality is an important factor for health and productivity. This model helps to predict whether a person will have Poor, Average, or Good sleep quality.

Problem Statement
Many people suffer from sleep problems due to lifestyle habits such as stress, screen time, caffeine intake, and lack of exercise.
The goal of this project is to build a machine learning model that predicts sleep quality using these factors.

Dataset
The dataset contains 700 records of Indian users with the following features:

Feature	Description
Sleep_Hours	Number of hours of sleep
Screen_Time	Daily screen time in hours
Caffeine_Intake	Coffee/tea intake (Yes/No)
Exercise	Daily exercise (Yes/No)
Stress	Stress level (Low/Medium/High)
Gender	Male/Female
Sleep_Quality	Target variable (Poor/Average/Good)
Data Cleaning
Data cleaning includes:

Removing missing values
Removing unnecessary columns
Correcting invalid values
Ensuring data consistency
Data Preprocessing
Before training the model, the following preprocessing steps were done:

Convert categorical columns into numeric values using encoding
Convert target column Sleep_Quality into numeric values
Split the data into training and testing sets
Apply feature scaling
Model Training
Several classification models were trained:

Logistic Regression
Decision Tree
Random Forest
K-Nearest Neighbors (KNN)
Model Evaluation
Models were evaluated using accuracy, confusion matrix, and classification report.

Model	Accuracy
Logistic Regression	0.94
Decision Tree	1.00
Random Forest	0.99
KNN	0.86
Best Model: Random Forest
It gives high accuracy and better performance on unseen data.

Prediction
After training the model, prediction can be done for new data values.

Example: Sleep_Hours = 7 Screen_Time = 2 Caffeine_Intake = Yes Exercise = Yes Stress = Medium Gender = Male

The model predicts the sleep quality (Poor/Average/Good).

Conclusion
This project successfully predicts sleep quality using machine learning.
The Random Forest model performed best and can help people understand how lifestyle affects sleep quality.

Future Work
Possible improvements:

Use real-world data from sleep trackers
Add more features such as age, diet, health conditions
Build a web or mobile application
Use advanced models like Neural Networks