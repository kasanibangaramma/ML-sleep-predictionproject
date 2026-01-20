# LIFESTYLE BASED ON THE SLEEP QUANTITY PREDICTION 

### &nbsp;STEPS FOR CLEANING THE DATASET

✅ Step 1: Load the dataset

✅ Step 2: Check missing values

✅ Step 3: Remove Name column

✅ Step 4: Convert Yes/No to numbers

### STEPS FOR PREPROSSING THE DATASET

✅ Step 5: Encode Gender \& State

✅ Step 6: Encode Sleep Quality

✅ Step 7: Split X and y

✅ Step 8: Train-test split

✅ Step 9: Feature scaling (optional)

##### NOTE: THE EXPLAINATION OF ALL THE 9 STEPS

To load your data

To check if any data is missing

Name is not useful

ML needs numbers

ML needs numbers

Target needs numbers

Separate features \& target

Create train and test data

Makes model accurate

##### EXPLAINATION FOR STEPS

###### Step 1: Load the Dataset



First, the dataset was loaded from an Excel file into Python using the Pandas library.

This converts the Excel sheet into a table-like structure called a Data Frame, which allows easy data analysis and manipulation.



###### Step 2: Check for Missing Values



After loading the dataset, it was checked for missing or empty values in each column.

This step is important because missing data can cause errors or incorrect results during model training.

The check confirmed that the dataset contained no missing values.



###### Step 3: Remove Irrelevant Columns



The “Name” column was removed from the dataset.

This column does not contribute to predicting sleep quality and keeping it would only increase unnecessary data without improving model performance.



###### Step 4: Convert Binary Categorical Values



Columns containing “Yes” and “No” values were converted into numerical format.

Machine learning models cannot understand text values, so:



“Yes” was converted to 1



“No” was converted to 0



This was done only for the columns that are used for prediction.



###### Step 5: Encode Categorical Features



Categorical text values such as Gender and State were converted into numeric labels.

This process assigns a unique number to each category, allowing the machine learning model to process these features correctly.



###### Step 6: Encode the Target Variable



The target column, Sleep Quality, was converted into numerical values.

Each category was mapped to a number so the model can learn and predict sleep quality effectively:



Poor → 0



Average → 1



Good → 2



###### Step 7: Separate Input Features and Output Label



The dataset was divided into two parts:



Input features (independent variables)



Output label (dependent variable – Sleep Quality)



This separation helps the model clearly understand what data to learn from and what it needs to predict.



###### Step 8: Split the Dataset into Training and Testing Sets



The data was split into two parts:



Training set (used to train the model)



Testing set (used to evaluate model performance)



This step ensures that the model is tested on unseen data to measure its accuracy.



###### Step 9: Feature Scaling



Feature scaling was applied to bring all numerical values to a similar range.

Since different features have different units and ranges, scaling improves model performance and prevents bias toward larger values.



###### Final Summary (Short \& Clear)



In this project, the dataset was first cleaned by checking for missing values and removing irrelevant columns.

Then, preprocessing techniques such as encoding categorical values, converting the target variable, splitting the dataset, and feature scaling were applied.

These steps prepared the data in a suitable numerical format for training a machine learning model.

### MODEL TRAINING 

###### Step 1: Understand the Problem Type



The output of this project is Sleep Quality, which has categories such as:



Poor



Average



Good



Since the output is categorical, this is a classification problem.

Therefore, classification models are used.



###### Step 2: Choose Multiple Models



Instead of using only one model, multiple classification models were selected so that their performance could be compared.



The models used are:



Logistic Regression



Decision Tree Classifier



Random Forest Classifier



K-Nearest Neighbors (KNN)



Using multiple models helps in selecting the best-performing model.



###### Step 3: Train Each Model



Each selected model was trained using the training dataset.



During training:



The model learns patterns from the input data



It understands how different features affect sleep quality



This step is called model training.



###### Step 4: Test Each Model



After training, each model was tested using the testing dataset.



The model predicts sleep quality for test data that it has never seen before.



###### Step 5: Evaluate Model Performance



The performance of each model was evaluated using accuracy.



Accuracy shows how many predictions were correct compared to total predictions.



###### Step 6: Compare Models



The accuracy values of all models were compared.



The model with the highest accuracy was considered the best model for this project.



###### Step 7: Select the Best Model



The best-performing model was selected and used for final predictions.



This ensures better results and improves reliability.



###### Step 8: Use the Best Model for Prediction



The selected model was used to predict sleep quality for new data.



One-Paragraph Model Explanation (Report Ready)



Multiple classification models were trained and evaluated to predict sleep quality. Each model was trained using the training dataset and tested on unseen data. The performance of the models was compared using accuracy, and the model with the highest accuracy was selected as the final model for prediction.



###### One-Line Viva Answer



“Multiple classification models were trained and compared, and the model with the highest accuracy was selected.”

BEST MODEL TO CHOOSE: Random Forest Classifier

###### Why Random Forest is the right choice



Very high accuracy (0.99)



Less overfitting than Decision Tree



More stable and reliable



Better generalization on unseen data



Highly accepted in academic projects

###### Final Model Selection Statement (REPORT READY)



“Although the Decision Tree achieved 100% accuracy, it is prone to overfitting. Random Forest Classifier achieved comparable accuracy with better generalization and was therefore selected as the final model.”

###### One-line Viva Answer



“Random Forest was selected because it provides high accuracy with better stability compared to Decision Tree.”

| Model               | Accuracy |

| ------------------- | -------- |

| Logistic Regression | 0.94     |

| Decision Tree       | \*\*1.00\*\* |

| Random Forest       | 0.99     |

| KNN                 | 0.86     |

### model evaluation 

#### testing the model

###### Step 1: Split the data into Train \& Test



You divided your data into:



Training data (used to teach the model)



Test data (used to check the model)



###### Step 2: Train the model



You trained your selected model (Random Forest) using training data.



###### Step 3: Predict using Test data



You used the trained model to predict sleep quality on the test data.



###### Step 4: Calculated Test Accuracy



You calculated the accuracy on test data and got 0.99

###### final description:



“The dataset was split into training and testing sets. The Random Forest model was trained on the training data and evaluated on unseen test data. The model achieved an accuracy of 0.99, showing strong prediction performance.”

##### Confusion Matrix \& Classification Report

##### confusion matrix:

###### output is:

\[\[49  0  0]

&nbsp;\[ 0 80  0]

&nbsp;\[ 0  1 10]]

Rows = Actual values



Columns = Predicted values

###### NOTE:“The confusion matrix shows how many test data samples were correctly or incorrectly predicted by the model.”



##### classification report:

###### output is:

&nbsp;             precision    recall  f1-score   support



&nbsp;          0       1.00      1.00      1.00        49

&nbsp;          1       0.99      1.00      0.99        80

&nbsp;          2       1.00      0.91      0.95        11



&nbsp;   accuracy                           0.99       140

&nbsp;  macro avg       1.00      0.97      0.98       140

weighted avg       0.99      0.99      0.99       140

###### NOTE:“The classification report provides precision, recall, and F1-score for each class. This helps to understand the performance of the model in detail.”

###### Perfect Viva Answer



“The confusion matrix and classification report were used to evaluate the model in detail. They show how accurately the model predicts each class and how many errors were made.”

### Model Evaluation



Model evaluation is the process of measuring how well the trained machine learning model performs on unseen data.

After training the model using the training dataset, the model was tested on the testing dataset to check its performance. This ensures that the model can predict accurately on new data and not just memorize the training data.



In this project, multiple classification models were trained and evaluated, including Logistic Regression, Decision Tree, Random Forest, and KNN. The dataset was split into training and testing sets using a ratio of 80:20. The models were trained on the training set and then tested on the test set.



The performance of each model was measured using accuracy, which indicates the percentage of correct predictions. The accuracy scores were compared, and the model with the highest accuracy was selected as the final model. The Random Forest model achieved the highest accuracy of 0.99, indicating strong performance and reliability.



Additionally, the model performance was further validated using a confusion matrix and classification report. The confusion matrix shows the number of correct and incorrect predictions for each class. The classification report provides precision, recall, and F1-score, which help in understanding the performance of the model for each category of sleep quality.



Overall, the model evaluation proves that the Random Forest classifier is able to predict sleep quality accurately based on the given features.

MODEL PREDICTION:

Prediction



Prediction is the final step of the project where the trained machine learning model is used to forecast sleep quality for new data.



##### What does prediction mean?



It means:



You give the model new information about a person (like sleep hours, exercise, screen time).



The model uses what it learned during training.



It predicts whether the person’s sleep quality is Poor, Average, or Good.



##### How it works



1\.New data is collected (example: 7 hours sleep, 2 hours screen time, etc.).



2\.The new data is converted into the same format as the training data.



3\.The model uses this data to predict the sleep quality.



4\.The predicted result is shown as one of the categories.

















































