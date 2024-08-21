# app.py
import streamlit as st
from backend import load_and_analyze_data, clean_and_format_dataframe, univariate_analysis


def main():
    st.title('Bank Churn Prediction Analysis')

    # Sidebar navigation
    st.sidebar.title('Navigation')
    page = st.sidebar.radio("Select a Step", [
                            "About Project", "Data Load & Analysis", "Clean & Format DataFrame", "Univariate Analysis"])

    # Load data
    data_url = './bank_churn_prediction_modeling.csv'
    df, initial_check_summary, data_quality_summary, potential_categorical_from_numerical = load_and_analyze_data(
        data_url)

    # Define df_cleaned in the outer scope
    df_cleaned = None

    if page == "About Project":
        st.header("About the Project")
        st.markdown("""
        ## Bank Churn Prediction 🔚🏃💼🚪❌

        This project focuses on predicting customer churn in a bank. Churn refers to customers leaving the bank, and predicting it is crucial for businesses to take preemptive actions to retain valuable customers. The data used in this project is a set of features that describe the customers' demographics, their relationship with the bank, and their financial status. By applying various machine learning models, we aim to predict whether a customer will churn based on these features.

        ### Objective

        The primary goal of this project is to develop a predictive model that accurately identifies customers at risk of leaving the bank. By analyzing the factors contributing to customer churn, the bank can implement strategies to enhance customer retention, thereby improving profitability and customer satisfaction.

        ### Dataset Overview

        The dataset consists of 13 columns that provide detailed information about each customer. These features include demographic information, account details, and the status of their relationship with the bank.

        ### Dataset Columns Description:

        - **Customer ID** 🆔: A unique identifier for each customer.
        - **Surname** 👤: The customer's surname or last name.
        - **Credit Score** 💳: A numerical value representing the customer's credit score.
        - **Geography** 🌍: The country where the customer resides (France, Spain, or Germany).
        - **Gender** ⚤: The customer's gender (Male or Female).
        - **Age** 🎂: The customer's age.
        - **Tenure** 🏦: The number of years the customer has been with the bank.
        - **Balance** 💰: The customer's account balance.
        - **NumOfProducts** 🛠: The number of bank products the customer uses (e.g., savings account, credit card).
        - **HasCrCard** 💳: Whether the customer has a credit card (1 = yes, 0 = no).
        - **IsActiveMember** ✅: Whether the customer is an active member (1 = yes, 0 = no).
        - **EstimatedSalary** 💼: The estimated salary of the customer.
        - **Exited** 🚪: Whether the customer has churned (1 = yes, 0 = no).

        ### Project Flow

        1. **Data Loading and Exploration**: The project starts by loading and exploring the dataset to understand its structure and key characteristics.
        2. **Data Cleaning and Preprocessing**: The data is cleaned, formatted, and preprocessed to ensure it is ready for modeling.
        3. **Exploratory Data Analysis**: Univariate and bivariate analyses are conducted to explore the relationships between different features and the target variable (churn).
        4. **Modeling**: Various machine learning models are applied to predict customer churn.
        5. **Evaluation**: The models are evaluated based on their performance metrics, such as accuracy, precision, recall, and F1-score.
        6. **Conclusion and Recommendations**: Based on the model's performance, conclusions are drawn, and recommendations are provided to reduce churn rates.
        """)

    elif page == "Data Load & Analysis":
        st.header("1. Data Load & Initial Analysis")

        st.markdown("""
        - 1.1. Import necessary libraries
        - 1.2. Load the dataset
        - 1.3. Check the shape of the dataset, duplicated rows, statistics summary
        - 1.4. Assess data quality: datatypes, number and % of unique values and missing values
        - 1.5. Identify categorical variables from numerical formats (less than 20 unique values)
        """)

        # Display initial data analysis results
        st.subheader("Shape of the DataFrame and Duplicate Rows")
        st.write(f"Shape of the DataFrame: {df.shape}")
        st.write(f"Duplicate Rows Number: {df.duplicated().sum()}")

        st.subheader("Summary Statistics")
        st.table(initial_check_summary)

        st.subheader("Data Quality Summary")
        st.table(data_quality_summary)

        st.subheader("Potential Categorical Variables from Numerical Columns")
        st.dataframe(potential_categorical_from_numerical)

    elif page == "Clean & Format DataFrame":
        st.header("Clean & Format DataFrame")

        st.markdown("""
        - 2.1. Dealing with Duplicates.
        - 2.2. Standardize Column Names: Remove leading and trailing spaces and underscores. Replace non-alphanumeric characters with underscores. Convert column names from CamelCase to snake_case.
        - 2.3. Data Types Correction.
        - 2.4. Check Domain-Specific Inconsistencies: Check for inconsistencies or errors specific to the domain (e.g., unrealistic values, mismatches in categorical data).
        """)

        # Clean and format the dataframe
        integer_columns = ['age', 'balance', 'has_cr_card',
                           'is_active_member', 'estimated_salary']
        df_cleaned, df_cleaned_path = clean_and_format_dataframe(
            df, integer_columns)

        st.subheader("Cleaned DataFrame")
        st.dataframe(df_cleaned.head())

        st.subheader("Path to Cleaned CSV")
        st.write(df_cleaned_path)

    elif page == "Univariate Analysis":
        st.header("Univariate Analysis")

        st.markdown("""
        - 3.1. Separate categorical and numerical columns
        - 3.2. Categorical variables. Frequency tables: counts and proportions
        - 3.3. Categorical variables. Barplots
        - 3.4. Categorical variables. Pie charts
                """)

        # Check if df_cleaned is available, if not, clean the data
        if df_cleaned is None:
            integer_columns = ['age', 'balance', 'has_cr_card',
                               'is_active_member', 'estimated_salary']
            df_cleaned, _ = clean_and_format_dataframe(df, integer_columns)

        # Perform univariate analysis
        univariate_analysis(df_cleaned)


if __name__ == '__main__':
    main()
