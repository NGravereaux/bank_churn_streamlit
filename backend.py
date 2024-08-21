# backend.py
# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import re
import math
import statsmodels.api as sm
import scipy.stats as stats
import streamlit as st
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, validation_curve, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OrdinalEncoder
from numpy import log1p
from scipy.stats.contingency import association, chi2_contingency
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


# Optional: Conditional import with logging for better debugging
try:
    from sklearn.feature_selection import SelectFromModel
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, precision_recall_curve, roc_curve, make_scorer, auc
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
except ImportError as e:
    st.error(f"Error importing some sklearn modules: {e}")


# Ensure to define and check your environment, Python version, and package versions
import sys
st.write(f"Python version: {sys.version}")
st.write(f"NumPy version: {np.__version__}")


# Function to load and analyze data
def load_and_analyze_data(file_path):
    # 1.2. Load the dataset
    df = pd.read_csv(file_path)

    # 1.3. Check the shape of the dataset, duplicated rows, and statistics summary
    def initial_data_checking(df):
        # Print the shape of the DataFrame (number of rows and columns)
        print(f"1. Shape of the DataFrame: {df.shape}")

        # Print the count of duplicate rows
        print(f"2. Duplicate Rows Number: {df.duplicated().sum()}")

        # Print summary statistics for numerical columns
        print("3. Summary Statistics:")
        return pd.DataFrame(df.describe())

    initial_check_summary = initial_data_checking(df)
    print(initial_check_summary)  # Print summary statistics

    # 1.4. Assess data quality: datatypes, number and % of unique values and missing values
    def unique_and_missing_values_dtype(df):
        # Non-null counts and data types
        non_null_counts = df.notnull().sum()
        dtypes = df.dtypes

        # Count of unique values
        unique_count = df.nunique()

        # Percentage of unique values
        unique_percentage = (df.nunique() / len(df)) * 100

        # Count of missing values
        missing_count = df.isnull().sum()

        # Percentage of missing values
        missing_percentage = df.isnull().mean() * 100

        # Combine into a DataFrame
        summary = pd.DataFrame({
            'non-Null_count': non_null_counts,
            'dtype': dtypes,
            'unique_values': unique_count,
            '%_unique': unique_percentage.round(2).astype(str) + '%',
            'missing_values': missing_count,
            '%_missing': missing_percentage.round(2).astype(str) + '%'
        })

        return summary

    data_quality_summary = unique_and_missing_values_dtype(df)
    print("\n4. Data Quality Summary:")
    print(data_quality_summary)  # Print data quality summary

    # 1.5. Identify categorical variables from numerical formats (less than 20 unique values)
    potential_categorical_from_numerical = df.select_dtypes(
        "number").loc[:, df.select_dtypes("number").nunique() < 20]
    print("\n5. Potential Categorical Variables from Numerical Columns:")
    # Print potential categorical variables
    print(potential_categorical_from_numerical.head())

    return df, initial_check_summary, data_quality_summary, potential_categorical_from_numerical


# Function to clean and format the DataFrame
def clean_and_format_dataframe(df, integer_columns):
    # 2.1. Delete Duplicates
    initial_row_count = df.shape[0]
    df.dropna(inplace=True)
    final_row_count = df.shape[0]
    print(f"2.1.Deleted {initial_row_count - final_row_count} "
          f"duplicate/missing rows.")

    # 2.2. Standardize Column Names
    def clean_column(name):
        # Convert camel case to snake case
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
        name = name.strip()  # Remove leading and trailing spaces
        # Replace non-alphanumeric characters with underscores
        name = re.sub(r'[^0-9a-zA-Z]+', '_', name)
        # Replace multiple underscores with a single underscore
        name = re.sub(r'_+', '_', name)
        name = name.lower()  # Convert to lowercase
        return name.strip('_')  # Remove leading and trailing underscores

    df.columns = [clean_column(col) for col in df.columns]
    print(f"2.2.Standardized column names: {df.columns.tolist()}")

    # 2.3. Data Types Correction
    def convert_float_to_integer(df, columns):
        for column in columns:
            df[column] = df[column].astype(int)
        return df

    df_cleaned = convert_float_to_integer(df, integer_columns)
    print(f"2.3.Converted columns to integer: {integer_columns}")

    # Export cleaned dataset
    df_cleaned_path = './df_cleaned.csv'
    df.to_csv(df_cleaned_path, index=False)
    print(df_cleaned.head())

    return df_cleaned, df_cleaned_path


# Function for Univariate Analysis
def univariate_analysis(df_cleaned):
    # 3.1. Separate categorical and numerical columns
    categorical_columns = ['geography', 'gender', 'tenure',
                           'num_of_products', 'has_cr_card', 'is_active_member', 'exited']
    numerical_columns = ['credit_score', 'age', 'balance', 'estimated_salary']

    df_categorical = df_cleaned[categorical_columns]
    df_numerical = df_cleaned[numerical_columns]
    st.subheader("Define Categorical and Numerical Variables")
    st.write(f"Categorical Variables: {df_categorical.columns}.")
    st.write(f"Numerical Variables: {df_numerical.columns}.")

    # 3.2. Categorical variables. Frequency tables: counts and proportions
    def generate_frequency_proportion_tables(df_categorical):
        frequency_proportion_results = {}

        for col in df_categorical.columns:
            # Calculate frequency and proportion
            frequency = df_categorical[col].value_counts()
            proportion = df_categorical[col].value_counts(
                normalize=True).round(2)

            # Combine into a single DataFrame
            result_table = pd.DataFrame({
                'Frequency': frequency,
                'Proportion': proportion
            })

            # Add the result to the dictionary
            frequency_proportion_results[col] = result_table

        return frequency_proportion_results
    st.subheader(
        "Frequency and Proportion Tables for Categorical Variables")
    frequency_proportion_tables = generate_frequency_proportion_tables(
        df_categorical)
    for col, table in frequency_proportion_tables.items():
        st.write(f"{col}:\n")
        st.table(table)
    st.markdown("""
        **Inference from Frequency Proportion Results**
        - **Geography:** France has the largest customer base (50%), with Germany and Spain equally represented (25% each).
        - **Gender:** The customer base is slightly male-dominated (55% male, 45% female).
        - **Tenure:** Tenure is evenly spread across the first nine years, with fewer customers at 10 years (5%) and new customers (4%).
        - **Number of Products:** Most customers use 1 or 2 products (96%), with very few using 3 or 4 products (4%).
        - **Has Credit Card:** The majority of customers (71%) have a credit card.
        - **Is Active Member:** The customer base is evenly split between active (51%) and inactive (49%) members.
        - **Exited:** Most customers have not churned (80%), exited (20%).
        """)

    # 3.3. Categorical variables. Barplots
    st.subheader("Plot Categorical Barplots")
    categorical_barplots_url = 'https://github.com/NGravereaux/bank_churn_streamlit/blob/main/1_univariate_plot_categorical_barplots.png?raw=true'
    st.image(categorical_barplots_url,
             caption="categorical barplots", use_column_width=True)

    # 3.4. Categorical variables. Pie charts
    st.subheader("Plot Categorical Pie Charts")
    categorical_piecharts_url = 'https://github.com/NGravereaux/bank_churn_streamlit/blob/main/2_Univariate%20Plot%20Categorical%20Pie_charts.png?raw=true'
    st.image(categorical_piecharts_url,
             caption="Plot Categorical Pie Charts", use_column_width=True)

    # 3.5. Numerical variables. Summary Statistics

    def summary_statistics(df_numerical):
        return pd.DataFrame(df_numerical.describe())
    st.subheader("Summary Statistics for Numerical Variables")
    summary_stats = summary_statistics(df_numerical)
    st.table(summary_stats)
    st.markdown("""
        **Inference from Summary Statistics**

        - **Balance:** The minimum value for the balance column is 0.0, but the 25th percentile is also 0.0. This suggests that at least 25% of the customers have a zero balance, which could indicate that a significant portion of the customers do not use or have funds in their accounts.
        - **Estimated Salary:** The estimated_salary column has a minimum value of 11.0, which is unusually low for a salary estimate. This could be a data entry error or an outlier. Additionally, the mean salary is 100,099.29, while the median (50th percentile) salary is 100,218.00, which suggests that the salary distribution is fairly symmetrical, but the low minimum value might still be an anomaly.
        - **Age:** The range of age (minimum of 18 and maximum of 92) seems reasonable, but the mean age of around 39 might indicate a relatively younger customer base.
        """)

    # 3.6. Numerical variables. Shape of the distribution: Skewness and Kurtosis
    def calculate_skewness_kurtosis(df_numerical):
        results = {'Column': [], 'Skewness': [], 'Kurtosis': []}

        for column in df_numerical.columns:
            skewness = round(df_numerical[column].skew(), 2)
            kurtosis = round(df_numerical[column].kurtosis(), 2)

            results['Column'].append(column)
            results['Skewness'].append(skewness)
            results['Kurtosis'].append(kurtosis)

        return pd.DataFrame(results)
    st.subheader("Skewness and Kurtosis for Numerical Variables")
    skewness_kurtosis = calculate_skewness_kurtosis(df_numerical)
    st.table(skewness_kurtosis)
    st.markdown("""
        **Interpretation of Shape of the Distribution Analysis**

        1. **Skewness (Shape of the Distribution)**
            - Skewness = 0: Symmetrical distribution (No action needed).
            - Skewness > 0: Right-skewed (Consider log or square root transformation).
            - Skewness < 0: Left-skewed (Consider inverse or square transformation).
            - -0.5 to 0.5: Fairly symmetrical (Generally acceptable).
            - -1 to -0.5 or 0.5 to 1: Moderately skewed (Might require transformation).
            - <-1 or >1: Highly skewed (Transformation recommended).

        2. **Kurtosis (Outliers)**
            - Kurtosis = 3: Normal distribution (No action needed).
            - Kurtosis > 3: Heavy tails (Check for outliers, consider robust methods).
            - Kurtosis < 3: Light tails (Typically acceptable, fewer outliers).
        """)

    # 3.7. Numerical variables. Plot Histograms for Numerical Variables
    st.subheader("Plot Histograms for Numerical Variables")
    histograms_url = 'https://github.com/NGravereaux/bank_churn_streamlit/blob/main/3_Univariate_Plot%20Histograms%20for%20Numerical%20Variables.png?raw=true'
    st.image(histograms_url, caption="Plot Histograms for Numerical Variables",
             use_column_width=True)

    # 3.8. Numerical variables. Plot Boxplots
    st.subheader("Plot Boxplots")
    boxplots_url = 'https://github.com/NGravereaux/bank_churn_streamlit/blob/main/4_Univariate_Plot%20Boxplots.png?raw=true'
    st.image(boxplots_url, caption="Plot Boxplots", use_column_width=True)


def bivariate_analysis1(df_categorical):
    st.header("4.1. Categorical (including Discrete Numerical) vs Categorical")
    st.subheader("Define Categorical Variables")
    st.write(f"**Categorical Variables**: {df_categorical.columns.tolist()}")

    # 4.1.1. Chi-square tests for categorical variables
    def calculate_and_sort_chi2(df_categorical):
        chi2_results = []
        for i, col1 in enumerate(df_categorical.columns):
            for col2 in df_categorical.columns[i+1:]:
                crosstab_result = pd.crosstab(
                    df_categorical[col1], df_categorical[col2])
                try:
                    chi2_statistic, chi2_p_value, _, _ = chi2_contingency(
                        crosstab_result)
                    chi2_results.append({
                        'Variable 1': col1,
                        'Variable 2': col2,
                        'Chi2 Statistic': round(chi2_statistic, 4),
                        'P-Value': round(chi2_p_value, 4)
                    })
                except ValueError:
                    chi2_results.append({
                        'Variable 1': col1,
                        'Variable 2': col2,
                        'Chi2 Statistic': None,
                        'P-Value': None
                    })
        return pd.DataFrame(chi2_results).sort_values(by='P-Value')

    st.subheader("Chi-square Tests for Categorical Variables")
    chi2_df = calculate_and_sort_chi2(df_categorical)
    st.table(chi2_df)

    # 4.1.2. Cramér's V calculation for categorical variables
    def calculate_cramers_v_for_all_pairs(df_categorical):
        def cramers_v(crosstab):
            chi2_statistic, _, _, _ = chi2_contingency(crosstab)
            n = crosstab.sum().sum()
            phi2 = chi2_statistic / n
            r, k = crosstab.shape
            return np.sqrt(phi2 / min(k-1, r-1))

        cramers_v_results = []
        for i, col1 in enumerate(df_categorical.columns):
            for col2 in df_categorical.columns[i+1:]:
                crosstab_result = pd.crosstab(
                    df_categorical[col1], df_categorical[col2])
                cramers_v_value = cramers_v(crosstab_result)
                cramers_v_results.append({
                    "Variable Pair": f"{col1} vs {col2}",
                    "Cramér's V": cramers_v_value
                })

        return pd.DataFrame(cramers_v_results).sort_values(by="Cramér's V", ascending=False)

    st.subheader("Cramér's V for Categorical Variables")
    df_cramers_v_results = calculate_cramers_v_for_all_pairs(df_categorical)
    st.table(df_cramers_v_results)

    # 4.1.3. Stacked bar chart visualization for categorical variables
    def plot_stacked_bar_charts_for_all_pairs(df_categorical):
        num_plots = len(df_categorical.columns) * \
            (len(df_categorical.columns) - 1) // 2
        num_cols = 3
        num_rows = (num_plots // num_cols) + (num_plots % num_cols > 0)
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(20, num_rows * 5))
        axes = axes.flatten()

        plot_idx = 0
        for i, col1 in enumerate(df_categorical.columns):
            for col2 in df_categorical.columns[i+1:]:
                ax = axes[plot_idx]
                crosstab_result = pd.crosstab(
                    df_categorical[col1], df_categorical[col2])
                crosstab_result.plot(kind="bar", stacked=True, ax=ax)
                ax.set_title(f'Stacked Bar Chart of {col1} vs {col2}')
                ax.set_xlabel(col1)
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
                plot_idx += 1

        # Hide any unused subplots
        for j in range(plot_idx, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        return fig

    st.subheader("Stacked Bar Charts for Categorical Variables")
    fig = plot_stacked_bar_charts_for_all_pairs(df_categorical)
    st.pyplot(fig)


def bivariate_analysis2(df_categorical, df_numerical):
    st.header("4.2. Categorical vs Continuous")

    # 4.2.1. Violin plot visualization for categorical vs numerical variables
    def plot_violin_plots(df_categorical, df_numerical):
        num_plots = len(df_categorical.columns) * len(df_numerical.columns)
        num_rows = (num_plots // 3) + (num_plots % 3 > 0)
        fig, axes = plt.subplots(
            nrows=num_rows, ncols=3, figsize=(18, num_rows * 6))
        axes = axes.flatten()

        plot_index = 0
        for cat_col in df_categorical.columns:
            for num_col in df_numerical.columns:
                sns.violinplot(
                    x=df_categorical[cat_col], y=df_numerical[num_col], ax=axes[plot_index])
                axes[plot_index].set_title(
                    f'Violin Plot: {num_col} by {cat_col}')
                plot_index += 1

        for i in range(plot_index, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig

    st.subheader("Violin Plots for Categorical vs Numerical Variables")
    fig = plot_violin_plots(df_categorical, df_numerical)
    st.pyplot(fig)

    # 4.2.2. Bar chart visualization for categorical vs numerical variables
    def plot_bar_charts(df_categorical, df_numerical):
        num_plots = len(df_categorical.columns) * len(df_numerical.columns)
        num_rows = (num_plots // 3) + (num_plots % 3 > 0)
        fig, axes = plt.subplots(
            nrows=num_rows, ncols=3, figsize=(18, num_rows * 6))
        axes = axes.flatten()

        plot_index = 0
        for cat_col in df_categorical.columns:
            for num_col in df_numerical.columns:
                sns.barplot(
                    x=df_categorical[cat_col], y=df_numerical[num_col], ci=None, ax=axes[plot_index])
                axes[plot_index].set_title(
                    f'Bar Chart: {num_col} by {cat_col}')
                plot_index += 1

        for i in range(plot_index, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig

    st.subheader("Bar Charts for Categorical vs Numerical Variables")
    fig = plot_bar_charts(df_categorical, df_numerical)
    st.pyplot(fig)

    # 4.2.3. Box plot visualization for categorical vs numerical variables
    def plot_box_plots(df_categorical, df_numerical):
        num_plots = len(df_categorical.columns) * len(df_numerical.columns)
        num_rows = (num_plots // 3) + (num_plots % 3 > 0)
        fig, axes = plt.subplots(
            nrows=num_rows, ncols=3, figsize=(18, num_rows * 6))
        axes = axes.flatten()

        plot_index = 0
        for cat_col in df_categorical.columns:
            for num_col in df_numerical.columns:
                sns.boxplot(
                    x=df_categorical[cat_col], y=df_numerical[num_col], ax=axes[plot_index])
                axes[plot_index].set_title(f'Box Plot: {num_col} by {cat_col}')
                plot_index += 1

        for i in range(plot_index, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig

    st.subheader(
        "Side-by-Side Box Plots for Categorical vs Numerical Variables")
    fig = plot_box_plots(df_categorical, df_numerical)
    st.pyplot(fig)


def bivariate_analysis3(df_numerical, df_for_spearman_and_heatmap):
    st.header("4.3. Continuous vs Continuous")

    # 4.3.1. Pearson correlation calculation and display
    def calculate_pearson_correlations(df):
        correlations = []
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                correlation = df[col1].corr(df[col2])
                correlations.append((col1, col2, correlation))
        correlation_df = pd.DataFrame(
            correlations, columns=['Variable 1', 'Variable 2', 'Correlation'])
        sorted_correlation_df = correlation_df.sort_values(
            by='Correlation', ascending=False)
        return sorted_correlation_df

    st.subheader("Pearson Correlation Coefficients for Numerical Variables")
    sorted_correlations = calculate_pearson_correlations(df_numerical)
    st.table(sorted_correlations)

    # 4.3.2. Spearman correlation calculation and display
    def calculate_spearman_correlations(df):
        correlations = []
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                correlation = df[col1].corr(df[col2], method='spearman')
                correlations.append((col1, col2, correlation))
        correlation_df = pd.DataFrame(
            correlations, columns=['Column 1', 'Column 2', 'Correlation'])
        sorted_correlation_df = correlation_df.sort_values(
            by='Correlation', ascending=False)
        return sorted_correlation_df

    st.subheader("Spearman Correlation Coefficients for Numerical Variables")
    spearman_correlations_sorted = calculate_spearman_correlations(
        df_numerical)
    st.table(spearman_correlations_sorted)

    # 4.3.4. Spearman correlation heatmap

    def plot_sorted_spearman_heatmap(df):
        correlation_matrix = df.corr(method='spearman')
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm",
                    square=True, linewidths=.5, ax=ax)
        ax.set_title("Correlation Heatmap for Selected Numerical Variables")
        return fig

    st.subheader(
        "Spearman Correlation Heatmap for Selected Numerical Variables")
    fig = plot_sorted_spearman_heatmap(df_for_spearman_and_heatmap)
    st.pyplot(fig)


# Function for Feature Engineering
def feature_engineering(df_cleaned):
    # 5.1. Create new features: categorization for credit score, age, tenure, balance, and salary
    def create_features(df):
        st.subheader("5.1. Create New Features")
        # Binning credit score with letters
        credit_score_bins = [349, 579, 669, 739, 799, 851]
        credit_score_labels = ['E', 'D', 'C', 'B', 'A']
        df['credit_score_cat'] = pd.cut(
            df['credit_score'], bins=credit_score_bins, labels=credit_score_labels, right=True)
        st.write(
            "Categorized `credit_score` into `credit_score_cat`:'E', 'D', 'C', 'B', 'A'")

        # Categorize age into segments
        age_bins = [17, 25, 35, 45, 55, 65, np.inf]
        age_labels = ['A', 'B', 'C', 'D', 'E', 'F']
        df['age_cat'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
        st.write("Categorized `age` into `age_cat`:'A', 'B', 'C', 'D', 'E', 'F'")

        # Categorize tenure into segments
        tenure_bins = [-1, 3, 5, 7, np.inf]
        tenure_labels = ['A', 'B', 'C', 'D']
        df['tenure_cat'] = pd.cut(
            df['tenure'], bins=tenure_bins, labels=tenure_labels)
        st.write("Categorized `tenure` into `tenure_cat`:'A', 'B', 'C', 'D'")

        # Categorize balance
        balance_bins = [-1, 50000, 90000, 127000, np.inf]
        balance_labels = ['A', 'B', 'C', 'D']
        df['balance_cat'] = pd.cut(
            df['balance'], bins=balance_bins, labels=balance_labels)
        st.write("Categorized `balance` into `balance_cat`:'A', 'B', 'C', 'D'")

        # Categorize estimated_salary into segments
        salary_bins = [11, 40000, 80000, 120000, 160000, 200000]
        salary_labels = ['A', 'B', 'C', 'D', 'E']
        df['salary_cat'] = pd.cut(
            df['estimated_salary'], bins=salary_bins, labels=salary_labels, right=False)
        st.write(
            "Categorized `estimated_salary` into `salary_cat`: 'A', 'B', 'C', 'D', 'E'")

        # Drop unnecessary columns
        df = df.drop(columns=['row_number', 'customer_id', 'surname'], axis=1)
        st.write(
            "Dropped unnecessary columns: `row_number`, `customer_id`, `surname`")

        st.dataframe(df.head())
        return df

    # 5.2. Encode non-ordinal categorical variables with one_hot_encoder
    def one_hot_encoder(df, drop_first=True):
        st.subheader("5.2. One-Hot Encode Non-Ordinal Categorical Variables")
        categorical_cols = ['geography']
        df = pd.get_dummies(df, columns=categorical_cols,
                            drop_first=drop_first, dtype=int)
        st.write("One-hot encoded `geography` column:")
        st.dataframe(df.head())
        return df

    # 5.3. Encode ordinal categorical variables with ordinal encoder
    def ordinal_encoder(df):
        st.subheader("5.3. Ordinal Encode Ordinal Categorical Variables")
        categorical_cols = ['credit_score_cat', 'age_cat',
                            'tenure_cat', 'balance_cat', 'salary_cat']

        # Initialize the OrdinalEncoder
        encoder = OrdinalEncoder()

        # Apply the OrdinalEncoder to the specified columns
        df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
        # Convert the encoded values to integers
        df[categorical_cols] = df[categorical_cols].astype(int)

        st.write(
            "Ordinal encoded columns: `credit_score_cat`, `age_cat`, `tenure_cat`, `balance_cat`, `salary_cat`")
        st.dataframe(df.head())
        return df

    # 5.4. Encode binary non-ordinal categorical variables with label_encoder
    def label_encoder(df, info=False):
        st.subheader(
            "5.4. Label Encode Binary Non-Ordinal Categorical Variables")
        binary_cols = ['gender', 'has_cr_card', 'is_active_member', 'exited']
        labelencoder = LabelEncoder()

        for col in binary_cols:
            df[col] = labelencoder.fit_transform(df[col])
            if info:
                d1, d2 = labelencoder.inverse_transform([0, 1])
                st.write(f'{col}\n0:{d1}, 1:{d2}')

        st.write(
            "Label encoded columns: `gender`, `has_cr_card`, `is_active_member`, `exited`")
        st.dataframe(df.head())
        return df

    # Apply all transformations
    df = create_features(df_cleaned)
    df = one_hot_encoder(df)
    df = ordinal_encoder(df)
    df = label_encoder(df)

    # 5.5. Assemble final dataset by dropping the original non-categorical columns
    st.subheader("5.5. Assemble Final Dataset")
    df_final = df.drop(
        columns=['credit_score', 'age', 'tenure', 'balance', 'estimated_salary'])
    st.write("Dropped original non-categorical columns: `credit_score`, `age`, `tenure`, `balance`, `estimated_salary`")
    st.dataframe(df_final.head())

    # Save the final DataFrame
    df_final_path = './df_final.csv'
    df_final.to_csv(df_final_path, index=False)
    st.subheader("Final DataFrame Saved")
    st.write(f"The final DataFrame has been saved to: {df_final_path}")

    return df_final
