# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 07:52:15 2024

@author: DELL
"""

## Anomaly Detection Isolation Forest ##
import pandas as pd
import streamlit as st

df=pd.read_csv("C:\\Users\\DELL\\Downloads\\Final Projects\\sensor_data.csv")
df.head(5)

import matplotlib.pyplot as plt
plt.boxplot(df['Temperature'])
df=df.drop('Anomaly',axis=1)

# Categorical Features: Bar Plots
#It provides insights into the frequency of each category.
import seaborn as sn
categorical_columns = ['Boiler Name']

# Creating subplots for categorical features
plt.figure(figsize=(15, 5))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(1, 1, i)
    sn.countplot(x=col, data=df, palette='pastel')
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
plt.show()

## Data Exploration and Analysis
#Understand the distribution of the data through visualizations.

# Numerical Features: Histogram
#This helps to understand the range, central tendency, and spread of each variable.

numerical_columns = ['Temperature']

# Create subplots for numerical features
import seaborn as sn
plt.figure(figsize=(5, 5))
for i, col in enumerate(numerical_columns,1):
    plt.subplot(1, 1, i)
    sn.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

df_1=pd.DataFrame(df['Temperature'])

# Model Building
from sklearn.ensemble import IsolationForest
#ValueError: The feature names should match those that were passed during fit.Feature names unseen at fit time:- Anomaly_Scores
#The error you're encountering suggests that the feature names used during prediction with the Isolation Forest
#--model do not match the feature names used during the model fitting. 
#--In this case, it appears that 'Anomaly_Scores' is a new feature that was not present in the dataset 
#--when the model was initially trained.
# Select the features used for training (excluding 'Anomaly_Scores' and 'Anomaly' if present)
features_for_training = df_1.drop(['Anomaly_Scores', 'Anomaly'], axis=1, errors='ignore')

iso = IsolationForest(contamination=0.02, random_state=0)
iso.fit(features_for_training)

# Getting Anomaly Scores
df_1['Anomaly_Scores'] = iso.decision_function(features_for_training)
df_1['Anomaly'] = iso.predict(features_for_training)


# Calculate Anomaly Detection Rate (ADR)
adr = sum(df_1['Anomaly'] == -1) / len(df_1['Anomaly'])
print(f'Anomaly Detection Rate: {adr:.4f}')

# Confusion 
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix([-1] * len(df_1['Anomaly']), df_1['Anomaly'], labels=[-1, 1])
print('Confusion Matrix:')
print(conf_matrix)

#Density-Based Spatial Clustering of Applications with Noise(DBSCAN)
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Assuming df_1 is a DataFrame with numerical features
# Select the columns you want to use for clustering, for example, the first two columns
features_for_clustering = df_1[['Temperature', 'Anomaly']].values

db = DBSCAN(eps=0.2)
db_labels = db.fit_predict(features_for_clustering)

# Scatter plot
plt.scatter(features_for_clustering[:, 0], features_for_clustering[:, 1], c=db_labels, cmap='viridis')
plt.xlabel('Temperature')
plt.ylabel('Anomaly')
plt.title('DBSCAN Clustering')
plt.show()

# Taking Anomaly and Anomaly scores 
from sklearn.metrics import auc,roc_curve
true_labels = df_1['Anomaly']
anomaly_scores = df_1['Anomaly_Scores']

# Calculating false positive rate (FPR), true positive rate (TPR), and threshold
fpr, tpr, thresholds = roc_curve(true_labels, anomaly_scores)

# Calculate the Area Under the Curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Display the AUC value
print('AUC:', roc_auc)

## Streamlit App
import base64
import plotly.express as px
import pandas as pd

# Assume df_1 is your DataFrame with 'Timestamp', 'Temperature', 'Anomaly', and 'Boiler Name' columns

# Function to preprocess and predict anomalies
def predict_anomalies(temperature):
    data = {'Temperature': [temperature]}
    df_input = pd.DataFrame(data)
    prediction = iso.predict(df_input)
    return prediction[0]

# Map predictions to "Anomaly Detected" or "Normal"
def map_to_label(prediction):
    return "Anomaly Detected" if prediction == -1 else "Normal"

# Load background image once
with open('C:/Users/DELL/Downloads/Anomaly_Img.png', "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    st.title("Anomaly Detection System")

    # Input for user
    temperature_input = st.number_input("Enter Temperature:", value=0.0, step=0.1)

    # Predict and display result
    if st.button("Predict Anomaly"):
        result = predict_anomalies(temperature_input)
        result_label = map_to_label(result)
        st.success(f"Prediction: {result_label}")

    # Scatter Plot
    df_1['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df_1['Anomaly Label'] = df_1['Anomaly'].map({-1: 'Detected', 1: 'Normal'})
    df_1['Boiler Name'] = df['Boiler Name']

    # Create an interactive scatter plot with Plotly
    fig = px.scatter(df_1, x='Timestamp', y='Temperature', title='Scatter Plot of Temperature',
                     labels={'Temperature': 'Temperature'},
                     template='plotly_dark',  # Dark mode
                     hover_data={'Temperature': ':.2f'},
                     color='Anomaly Label',  # Color points based on the 'Anomaly' column
                     color_discrete_map={'Detected': 'red', 'Normal': 'blue'}  # Map colors for 'Detected' and 'Normal'
                     )

    # Customize layout
    fig.update_layout(
        xaxis=dict(title='Timestamp', showgrid=False),
        yaxis=dict(title='Temperature', showgrid=False),
    )

    # Scatter Plot for Each Boiler
    fig_boiler = px.scatter(df_1, x='Boiler Name', y='Temperature', color='Anomaly',
                            title='Temperature in Each Boiler with Anomalies',
                            labels={'Temperature': 'Temperature', 'Anomaly': 'Anomaly'},
                            color_discrete_map={'1': 'blue', '-1': 'red'},  # Map colors for '0' and '1'
                            category_orders={'Boiler Name': df_1['Boiler Name'].unique()},  # Specify category order
                            template='plotly_dark'  # Dark mode
                            )

    # Customize layout
    fig_boiler.update_layout(
        xaxis=dict(title='Boiler Name', showgrid=False),
        yaxis=dict(title='Temperature', showgrid=False),
    )

    # Create a radio button for selecting the graph
    selected_graph = st.radio("Select Graph", ("Temperature vs. Timestamp", "Anomalies detected in Each Boiler"))

    # Display the selected graph
    if selected_graph == "Temperature vs. Timestamp":
        st.plotly_chart(fig)
    elif selected_graph == "Anomalies detected in Each Boiler":
        st.plotly_chart(fig_boiler)

if __name__ == "__main__":
    main()
