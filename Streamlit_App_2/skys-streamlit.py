import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import time

# Başlık ve açıklama
st.title("Sky Data Classification and Visualization")
st.write("This project explores classification techniques applied to a dataset containing observations of stars, galaxies, and quasars.")

# Veri yükleme
sky = pd.read_csv("C:\\Users\\tetik\\Downloads\\SkyXGboost\\Skyserver_SQL2_27_2018 6_51_39 PM.csv", skiprows=0)
sky.drop(['objid', 'run', 'rerun', 'camcol', 'field', 'specobjid'], axis=1, inplace=True)

# Veri setini göster
st.write("### Dataset Overview")
if st.checkbox('Show dataframe'):
    st.write(sky)

# PCA işlemi ve veri hazırlığı
le = LabelEncoder()
sky['class'] = le.fit_transform(sky['class'])
pca = PCA(n_components=3)
pca_components = pca.fit_transform(sky[['u', 'g', 'r', 'i', 'z']])
sky[['PCA_1', 'PCA_2', 'PCA_3']] = pca_components
sky.drop(['u', 'g', 'r', 'i', 'z'], axis=1, inplace=True)

# Veri normalizasyonu
scaler = MinMaxScaler()
sky_scaled = scaler.fit_transform(sky.drop('class', axis=1))
X_train, X_test, y_train, y_test = train_test_split(sky_scaled, sky['class'], test_size=0.33)

# Sayısal ve Kategorik Veriler için Görselleştirmeler
st.write("### Numerical Features Visualization")
numerical_features = sky.select_dtypes(include=[np.number]).columns

option = st.selectbox("Select numerical feature for visualization:", numerical_features)

if option:
    st.write(f"#### Distribution of {option}")
    fig, ax = plt.subplots()
    sns.histplot(sky[option], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    st.write(f"#### Box Plot of {option}")
    fig, ax = plt.subplots()
    sns.boxplot(x=sky['class'], y=sky[option], ax=ax)
    ax.set_title(f'{option} by Class')
    st.pyplot(fig)

    st.write(f"#### Violin Plot of {option}")
    fig, ax = plt.subplots()
    sns.violinplot(x=sky['class'], y=sky[option], ax=ax)
    ax.set_title(f'{option} by Class')
    st.pyplot(fig)


# Model değerlendirme fonksiyonu
def evaluate_model(model, model_name):
    training_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_end = time.perf_counter()
    prediction_start = time.perf_counter()
    preds = model.predict(X_test)
    prediction_end = time.perf_counter()
    
    accuracy = (preds == y_test).sum().astype(float) / len(preds) * 100
    train_time = training_end - training_start
    prediction_time = prediction_end - prediction_start

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    # Sonuçlar için veri çerçevesi
    results = {
        'Model': model_name,
        'Accuracy (%)': [accuracy],
        'Training Time (s)': [train_time],
        'Prediction Time (s)': [prediction_time]
    }
    results_df = pd.DataFrame(results)

    return results_df, cm_df

# Model eğitim ve değerlendirme
results_list = []

# KNN Modeli
knn = KNeighborsClassifier()
knn_results, knn_cm = evaluate_model(knn, "K-Nearest Neighbors")
results_list.append(knn_results)

# Gaussian Naive Bayes Modeli
gnb = GaussianNB()
gnb_results, gnb_cm = evaluate_model(gnb, "Gaussian Naive Bayes")
results_list.append(gnb_results)

# Random Forest Modeli
rfc = RandomForestClassifier(n_estimators=100)
rfc_results, rfc_cm = evaluate_model(rfc, "Random Forest")
results_list.append(rfc_results)

# XGBoost Modeli
xgb = XGBClassifier(n_estimators=100)
xgb_results, xgb_cm = evaluate_model(xgb, "XGBoost")
results_list.append(xgb_results)

# Tüm model sonuçlarını birleştir
all_results_df = pd.concat(results_list, ignore_index=True)

st.write("### Model Results")
st.table(all_results_df)

# Model karşılaştırma sonuçları
st.write("### Model Comparison")
st.table(all_results_df[['Model', 'Accuracy (%)']])

# Confusion Matrix Görselleştirmeleri
def plot_cm(cm_df, model_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(f'{model_name} Confusion Matrix')
    return fig

# Confusion Matrix seçim menüsü
model_names = ["K-Nearest Neighbors", "Gaussian Naive Bayes", "Random Forest", "XGBoost"]
selected_model = st.selectbox("**Select a model to view the confusion matrix:**", model_names)

if selected_model == "K-Nearest Neighbors":
    st.write("### K-Nearest Neighbors Confusion Matrix")
    st.pyplot(plot_cm(knn_cm, "K-Nearest Neighbors"))
elif selected_model == "Gaussian Naive Bayes":
    st.write("### Gaussian Naive Bayes Confusion Matrix")
    st.pyplot(plot_cm(gnb_cm, "Gaussian Naive Bayes"))
elif selected_model == "Random Forest":
    st.write("### Random Forest Confusion Matrix")
    st.pyplot(plot_cm(rfc_cm, "Random Forest"))
elif selected_model == "XGBoost":
    st.write("### XGBoost Confusion Matrix")
    st.pyplot(plot_cm(xgb_cm, "XGBoost"))
