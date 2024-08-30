import os
# Base Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data Analysis Libraries
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report,confusion_matrix,mean_squared_error,accuracy_score,recall_score,precision_score,f1_score,roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV, RandomizedSearchCV
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import scipy.stats as stats
from scipy.stats import shapiro, levene, kruskal, ttest_ind, f_oneway
#from tabulate import tabulate

# Machine Learning Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("parkinsons_disease_data.csv")
st.title('Parkinson Disease')
df = df.drop("DoctorInCharge", axis = 1) 
if st.checkbox('Show dataframe'):
    st.write(df)
st.subheader("Description of Data")
def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # Create a DataFrame for the results
    results = {
        "Category": ["Observations", "Variables", "cat_cols", "num_cols", "cat_but_car", "num_but_cat"],
        "Count": [
            dataframe.shape[0],
            dataframe.shape[1],
            len(cat_cols),
            len(num_cols),
            len(cat_but_car),
            len(num_but_cat)
        ]
    }
    
    results_df = pd.DataFrame(results)
    
    # Display the DataFrame using Streamlit
    st.dataframe(results_df)

    return cat_cols, num_cols, cat_but_car

# Assuming you have a DataFrame `df`
cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    # Display summary table
    st.write(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                           "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        # Plot frequency count
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        
        # Count plot
        plt.subplot(1, 2, 1)
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.title("Frequency of " + col_name)
        plt.xticks(rotation=90)
        
        # Pie chart
        plt.subplot(1, 2, 2)
        values = dataframe[col_name].value_counts()
        plt.pie(x=values, labels=values.index, autopct=lambda p: '{:.2f}% ({:.0f})'.format(p, p/100 * sum(values)))
        plt.title("Frequency of " + col_name)
        plt.legend(labels=['{} - {:.2f}%'.format(index, value/sum(values)*100) for index, value in zip(values.index, values)],
                   loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1)
        
        st.pyplot(fig)

# Sample DataFrame
# df = pd.read_csv('your_data.csv')  # Uncomment this line to load your data

# Streamlit app
st.title('Categorical Data Summary')

# Dropdown menu to select column
cat_cols = [
    'Gender', 'Ethnicity', 'EducationLevel', 'Smoking', 'FamilyHistoryParkinsons',
    'TraumaticBrainInjury', 'Hypertension', 'Diabetes', 'Depression', 'Stroke',
    'Tremor', 'Rigidity', 'Bradykinesia', 'PosturalInstability', 'SpeechProblems',
    'SleepDisorders', 'Constipation', 'Diagnosis'
]

selected_col = st.selectbox('Select a categorical column', cat_cols)

# Display summary for the selected column
if selected_col:
    cat_summary(df, selected_col, plot=True)
    
def num_summary(dataframe, numerical_col, plot=False):
    # Display summary statistics
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    summary_stats = dataframe[numerical_col].describe(quantiles).T
    st.write(summary_stats)

    if plot:
        # Plot histogram and boxplot
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        
        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(y=numerical_col, data=dataframe)
        plt.title("Distribution of " + numerical_col)
        plt.xticks(rotation=90)
        
        st.pyplot(fig)
        st.write("______________________________________________________")

# Sample DataFrame
# df = pd.read_csv('your_data.csv')  # Uncomment this line to load your data

# Streamlit app
st.title('Numerical Data Summary')

# Dropdown menu to select column
num_cols = [
    'PatientID', 'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 
    'DietQuality', 'SleepQuality', 'SystolicBP', 'DiastolicBP', 
    'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 
    'CholesterolTriglycerides', 'UPDRS', 'MoCA', 'FunctionalAssessment'
]

selected_col = st.selectbox('Select a numerical column', num_cols)

# Display summary for the selected column
if selected_col:
    num_summary(df, selected_col, plot=True)
    
    
    
def target_summary_with_num(dataframe, target, numerical_col):
    results=dataframe.groupby(target).agg({numerical_col: "mean"})
    results_df = pd.DataFrame(results)
    
    
def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


cat_cols, num_cols, cat_but_car = grab_col_names(df)
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
y = df["Diagnosis"]
X = df.drop("Diagnosis", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

rf_model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)



def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "Model": [model.__class__.__name__],
        "Accuracy": [round(accuracy_score(y_test, y_pred), 2)],
        "Recall": [round(recall_score(y_test, y_pred), 3)],
        "Precision": [round(precision_score(y_test, y_pred), 2)],
        "F1 Score": [round(f1_score(y_test, y_pred), 2)],
        "AUC": [round(roc_auc_score(y_test, y_pred), 2)]
    }
    return metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Example usage
st.title('Model Evaluation Metrics')
models = [
    LogisticRegression(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200),
    SVC(),
    XGBClassifier(learning_rate=0.06),
]

# Evaluate all models
results = []
for model in models:
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    results.append(metrics)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display results in a table
st.write("Model Performance Metrics:")
st.dataframe(results_df)


def display_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    
    # Display the plot in Streamlit
    st.pyplot(fig)

# Add dropdown for model selection
selected_model_name = st.selectbox('Select a model', [model.__class__.__name__ for model in models])

# Find the selected model
selected_model = next(model for model in models if model.__class__.__name__ == selected_model_name)

# Display confusion matrix for the selected model
if selected_model:
    st.write(f"Confusion Matrix for {selected_model_name}:")
    display_confusion_matrix(selected_model, X_test, y_test)