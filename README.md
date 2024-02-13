# Predicting-E-Commerce-Website-Visitor-to-Customer-Conversion
In the rapidly evolving E-commerce industry, understanding customer behavior is crucial for business growth. The goal of this project is to analyze an E-commerce dataset and develop a predictive model that can accurately classify visitors based on their likelihood to convert into customers.

* The dataset contains a variety of features, including a target variable has_converted which indicates whether a visitor has converted into a customer.  
* The challenge is to use this data to train a classification model that can predict the has_converted status for future visitors. The outcome of this project will provide valuable insights that can help in strategizing effective customer conversion techniques.

* DataSet: https://drive.google.com/drive/folders/1ATULlRKrSensZHs2SxaT7y0b68Rc1vQA


## Data Preprocessing: 
Preprocess the data like handing missing values, treat outliers if necessary, checking duplicates etc.. to clean and structure it for machine learning model.

## Feature Engineering:
Extract relevant features from the dataset and Create any additional features that may enhance prediction accuracy.

## Encoding Categorical Variables:
Encode the categorical variables to numerical ones using label encoding or one-hot-encoding.

## Feature Selection:
select the suitable features which can contribute more for prediction.

## Model Selection and Training: 
Choose an appropriate machine learning model for classification (e.g.random forests, XGBClassifier, logistic regression or any other algorithms). Train the model using a portion of the dataset for training and the rest of the data for testing.

## Model Evaluation:
Evaluate the model's predictive performance for classification using classification metrics such as accuracy, recall, precision and F1 Score.

## Choosing Model:
After model evaluation, choosing logistic regression model for predicting whether user will convert as a customer or not (convert / not convert).

## Streamlit Web Application: 
Develop a user-friendly web application using Streamlit that allows users to input details (transaction revenue, device operating system, time on site, quality session etc). Utilize the trained machine learning model to predict whether user will convert as a customer or not based on user inputs.

* input - transaction_revenue, device_operating_system, time_on_site, session_quality, products_array
* output - convert/not convert


