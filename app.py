import streamlit as st
from backend import load_and_analyze_data, clean_and_format_dataframe, univariate_analysis, bivariate_analysis1, bivariate_analysis2, bivariate_analysis3, feature_engineering


def main():
    st.title('Customer Churn Prediction')

    st.sidebar.title('Navigation')
    page = st.sidebar.radio("Project Process", [
        "About Project", "Data Load & Analysis", "Clean & Format", "Univariate Analysis", "Bivariate Analysis", "Feature Engineering", "Model Building & Evaluation"])

    # Sidebar user information and navigation
    st.sidebar.markdown("&nbsp;")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Project realized by:")
    st.sidebar.markdown("**Natalia Mikhieieva Gravereaux**")

    # GitHub link with clickable text
    st.sidebar.markdown(
        '<a href="https://github.com/NGravereaux" target="_blank">'
        '<img src="https://img.icons8.com/ios-filled/50/000000/github.png" alt="GitHub" style="width:30px;height:30px;">'
        '</a> <a href="https://github.com/NGravereaux" target="_blank">NGravereaux</a>',
        unsafe_allow_html=True
    )

    # LinkedIn link with clickable text
    st.sidebar.markdown(
        '<a href="https://www.linkedin.com/in/nmikh/" target="_blank">'
        '<img src="https://img.icons8.com/ios-filled/50/000000/linkedin.png" alt="LinkedIn" style="width:30px;height:30px;">'
        '</a> <a href="https://www.linkedin.com/in/nmikh/" target="_blank">nmikh</a>',
        unsafe_allow_html=True
    )

    # Load data
    data_url = './bank_churn_prediction_modeling.csv'
    df, initial_check_summary, data_quality_summary, potential_categorical_from_numerical = load_and_analyze_data(
        data_url)

    # Define df_cleaned in the outer scope
    df_cleaned = None

    if page == "About Project":
        st.header("🔚🏃💼")
        st.markdown("""
        ### Customer Churn Prediction in the Banking Sector Using Machine Learning-Based Classification Models 

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
        # Add a horizontal line divider
        st.markdown("---")
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

    elif page == "Clean & Format":
        st.header("2. Clean & Format")

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
        st.header("3. Univariate Analysis")

        st.markdown("""
        - 3.1. Separate categorical and numerical columns
        - 3.2. Categorical variables. Frequency tables: counts and proportions
        - 3.3. Categorical variables. Barplots
        - 3.4. Categorical variables. Pie charts
        - 3.5. Numerical variables. Summary Statistics
        - 3.6. Numerical variables. Shape of the distribution: Skewness and Kurtosis
        - 3.7. Plot Histograms for Numerical Variables
        - 3.8. Plot Boxplots
                """)

        # Add a horizontal line divider
        st.markdown("---")

        # Check if df_cleaned is available, if not, clean the data
        if df_cleaned is None:
            integer_columns = ['age', 'balance', 'has_cr_card',
                               'is_active_member', 'estimated_salary']
            df_cleaned, _ = clean_and_format_dataframe(df, integer_columns)

        # Perform univariate analysis
        univariate_analysis(df_cleaned)

    elif page == "Bivariate Analysis":
        st.header("4. Bivariate Analysis")

        st.markdown("""
        - 4.1. Categorical (including Discrete Numerical) vs Categorical
          - 4.1.1. Chi-square test. Test the independence of variables
          - 4.1.2. Cramér's V. Test the independence of variables
          - 4.1.3. Visualization stacked bar. Analysis of proportional relationships between pairs of categorical.
          - 4.1.4. Visualization Frequency heat maps. The density or count of occurrences for combinations of two categorical variables, with color intensity indicating the frequency of each combination
        - 4.2. Categorical vs Continuous
          - 4.2.1. Violin Plots
          - 4.2.2. Bar Charts
          - 4.2.3. Side by side Box Plots
        - 4.3. Continuous vs Continuous
          - 4.3.1. Pearson Correlation coefficients
          - 4.3.2. Spearman Correlation coefficients
          - 4.3.3. Scatter plots/ pairplot
          - 4.3.4. Correlation Heatmaps
        """)
        # Add a horizontal line divider
        st.markdown("---")

        # Check if df_cleaned is available, if not, clean the data
        if df_cleaned is None:
            integer_columns = ['age', 'balance', 'has_cr_card',
                               'is_active_member', 'estimated_salary']
            df_cleaned, _ = clean_and_format_dataframe(df, integer_columns)

        # Separate categorical and numerical variables
        df_categorical = df_cleaned[['geography', 'gender', 'tenure',
                                     'num_of_products', 'has_cr_card', 'is_active_member', 'exited']]
        df_numerical = df_cleaned[['credit_score',
                                   'age', 'balance', 'estimated_salary']]
        df_for_spearman_and_heatmap = df_cleaned[['credit_score', 'age', 'tenure', 'balance',
                                                  'num_of_products', 'has_cr_card',
                                                  'is_active_member', 'estimated_salary', 'exited']]

        # Perform bivariate analysis
        bivariate_analysis1(df_categorical)
        bivariate_analysis2(df_categorical, df_numerical)
        bivariate_analysis3(df_numerical, df_for_spearman_and_heatmap)

    elif page == "Feature Engineering":
        st.header("5. Feature Engineering")

        st.markdown("""
        - 5.1. Create new features: categorization for credit score, age, tenure, balance, salary.
        - 5.2. Encode categorical variables.
        - 5.3. Check and Remove highly correlated variables: (Pearson correlation coefficient > 0.8).
        - 5.4. Assemble the final dataset (ABT).
        """)
        # Add a horizontal line divider
        st.markdown("---")

        # Check if df_cleaned is available, if not, clean the data
        if df_cleaned is None:
            integer_columns = ['age', 'balance', 'has_cr_card',
                               'is_active_member', 'estimated_salary']
            df_cleaned, _ = clean_and_format_dataframe(df, integer_columns)
        # Perform Feature Engineering
        feature_engineering(df_cleaned)

    elif page == "Model Building & Evaluation":
        st.header("6. Model Building & Evaluation")
        st.markdown("""
        - 6.1. Feature selection.
        - 6.2. Data Splitting into training and testing sets.
        - 6.3. Define features importance.
        - 6.4. Build and evaluate 5 Models No-Resampling: CatBoost, Random Forest, XGBoost, LightGBM, Neural Network.
        - 6.5. Hyperparameters tuning for 5 Models No-Resampling: CatBoost, Random Forest, XGBoost, LightGBM, Neural Network.
        - 6.6. Build and evaluate 5 Models With Resampling: CatBoost, Random Forest, XGBoost, LightGBM, Neural Network.
        - 6.7. Hyperparameters Tuning 5 Models with Resampling: CatBoost, Random Forest, XGBoost, LightGBM, Neural Network.
        - 6.8. Plot ROC AUC curve.
        """)
        # Add a horizontal line divider
        st.markdown("---")

        st.subheader("Overview")
        st.markdown("""
        In this phase, multiple machine learning models were built and evaluated to predict customer churn. 
        The models included:
        
        - **CatBoost**
        - **Random Forest**
        - **XGBoost**
        - **LightGBM**
        - **Neural Network**

        Initially, these models were trained and tested without any resampling to gauge their performance on the original data distribution.

        Following this, various resampling strategies were applied to address class imbalance and enhance model performance. The resampling techniques used were:

        - **Original Data**: No resampling was applied.
        - **Undersampling**: Reducing the majority class to balance the dataset.
        - **Oversampling**: Increasing the minority class to balance the dataset.
        - **SMOTE**: Synthetic Minority Over-sampling Technique to create synthetic samples of the minority class.

        After extensive evaluation, the best results were obtained with the **Random Forest** model using the **Original Data** strategy, demonstrating robust performance across multiple metrics.
        """)
    # Add Results After Resampling
        st.subheader("Results after Resampling")
        st.markdown("""
            **Best performing model on cross-validation:**
            - Fold: 2.0
            - CV Accuracy: 0.925486
            - CV Precision: 0.891806
            - CV Recall: 0.968534
            - CV F1 Score: 0.928573
            - CV AUC: 0.975634
            - Model: Random Forest
            - Resampling Strategy: Oversampling

            **Best performing model on test set:**
            - Resampling Strategy: Original
            - Model: LightGBM
            - Test Accuracy: 0.846
            - Test Precision: 0.730769
            - Test Recall: 0.443925
            - Test F1 Score: 0.552326
            - Test AUC: 0.865342
            """)

        # Add Results After ReSampling and Tuning
        st.subheader("Results After ReSampling and Tuning")
        st.markdown("""
            **Best performing model on test set:**
            - Resampling Strategy: Original
            - Model: Random Forest
            - Test Accuracy: 0.8355
            - Test Precision: 0.609272
            - Test Recall: 0.64486
            - Test F1 Score: 0.626561
            - Test AUC: 0.859508
            """)

        # Display ROC Curve Image
        st.subheader("Plot ROC AUC Curve")
        roc_curve_url = 'https://github.com/NGravereaux/bank_churn_streamlit/blob/main/roc_curve.png?raw=true'
        st.image(roc_curve_url, caption="ROC AUC Curve", use_column_width=True)


if __name__ == '__main__':
    main()
