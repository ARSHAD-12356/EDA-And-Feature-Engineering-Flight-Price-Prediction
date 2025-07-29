import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

# Streamlit app title
st.title("✈️ Flight Price EDA & Feature Engineering")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload your Flight Price Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.success("File successfully loaded!")

    # Initial data preview
    st.subheader("📄 Raw Dataset Preview")
    st.write(df.head())

    # Extracting date, month, year
    df['Date'] = df['Date_of_Journey'].str.split('/').str[0].astype(int)
    df['Month'] = df['Date_of_Journey'].str.split('/').str[1].astype(int)
    df['Year'] = df['Date_of_Journey'].str.split('/').str[2].astype(int)
    df.drop('Date_of_Journey', axis=1, inplace=True)

    # Splitting arrival time
    df['Arrival_hour'] = df['Arrival_Time'].str.split(' ').str[0].str.split(':').str[0].astype(int)
    df['Arrival_min'] = df['Arrival_Time'].str.split(' ').str[0].str.split(':').str[1].astype(int)
    df.drop('Arrival_Time', axis=1, inplace=True)

    # Splitting departure time
    df['Dept_hour'] = df['Dep_Time'].str.split(':').str[0].astype(int)
    df['Dept_min'] = df['Dep_Time'].str.split(':').str[1].astype(int)
    df.drop('Dep_Time', axis=1, inplace=True)

    # Dropping Route
    df.drop('Route', axis=1, inplace=True)

    # Duration feature engineering
    df['Duration'] = df['Duration'].fillna('0h 0m').str.strip()
    df['Duration'] = df['Duration'].apply(lambda x: '0h ' + x if 'h' not in x else x)
    df['Duration'] = df['Duration'].apply(lambda x: x + ' 0m' if 'm' not in x else x)
    df['Dur_hour'] = df['Duration'].str.extract(r'(\d+)h')[0].fillna(0).astype(int)
    df['Dur_min'] = df['Duration'].str.extract(r'(\d+)m')[0].fillna(0).astype(int)
    df.drop('Duration', axis=1, inplace=True)

    # Total Stops mapping
    df['Total_Stops'] = df['Total_Stops'].map({
        'non-stop': 0,
        '1 stop': 1,
        '2 stops': 2,
        '3 stops': 3,
        '4 stops': 4,
        np.nan: 1
    })

    st.subheader("🧹 Cleaned Dataset Preview")
    st.write(df.head())

    # Columns overview for debugging
    st.subheader("📌 Columns Available in DataFrame:")
    st.write(df.columns)

    # Categorical overview
    st.subheader("📊 Categorical Feature Distribution")
    fig, ax = plt.subplots(figsize=(12, 6))
    df['Airline'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Airline Distribution")
    st.pyplot(fig)

    # Duration vs Price Boxplot
    st.subheader("📦 Boxplot: Class vs Price")
    if 'Class' in df.columns and 'Price' in df.columns:
        if df['Class'].isnull().sum() == 0 and df['Price'].isnull().sum() == 0:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            sns.boxplot(x='Class', y='Price', data=df, ax=ax2)
            st.pyplot(fig2)
        else:
            st.warning("⚠️ 'Class' या 'Price' column में null values हैं। कृपया dataset को जांचें।")
    else:
        st.warning("⚠️ Dataset में 'Class' या 'Price' column मौजूद नहीं हैं।")

    # One-hot encoding preview
    st.subheader("🧠 OneHotEncoder Preview")
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(df[['Airline', 'Source', 'Destination']]).toarray()
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
    st.write(encoded_df.head())

else:
    st.warning("📎 Please upload your Excel file to proceed.")
