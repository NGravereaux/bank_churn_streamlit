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

    # 3.3. Categorical variables. Barplots
    def plot_categorical_barplots(df_categorical):
        num_cols = 3
        num_plots = len(df_categorical.columns)
        num_rows = math.ceil(num_plots / num_cols)

        fig, ax = plt.subplots(
            num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
        ax = ax.flatten()

        for i, col in enumerate(df_categorical.columns):
            sns.countplot(data=df_categorical, x=col, ax=ax[i])
            ax[i].set_title(f'Distribution of {col}')
            ax[i].set_xlabel(col)
            ax[i].set_ylabel('Count')
            plt.xticks(rotation=45)

        plt.tight_layout()
        return fig

    # 3.4. Categorical variables. Pie charts
    def plot_categorical_pie_charts(df_categorical):
        num_cols = 3
        num_plots = len(df_categorical.columns)
        num_rows = math.ceil(num_plots / num_cols)

        fig, ax = plt.subplots(
            num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
        ax = ax.flatten()

        for i, col in enumerate(df_categorical.columns):
            df_categorical[col].value_counts().plot.pie(
                autopct='%1.1f%%', colors=sns.color_palette("Set3"), startangle=90, ax=ax[i])
            ax[i].set_title(f'Distribution of {col}')
            ax[i].set_ylabel('')  # Hide the y-label for better aesthetics

        plt.tight_layout()
        return fig

    # 3.5. Numerical variables. Summary Statistics
    def summary_statistics(df_numerical):
        return pd.DataFrame(df_numerical.describe())

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

    # 3.7. Plot Histograms for Numerical Variables
    def plot_histograms(df_numerical):
        num_cols = 2
        num_plots = len(df_numerical.columns)
        num_rows = math.ceil(num_plots / num_cols)

        fig, ax = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
        ax = ax.flatten()

        for i, col in enumerate(df_numerical.columns):
            df_numerical[col].plot.hist(
                bins=60, ax=ax[i], color="skyblue", edgecolor="black")
            ax[i].set_title(f'Distribution of {col}')
            ax[i].set_xlabel(col)
            ax[i].set_ylabel('Frequency')

        plt.tight_layout()
        return fig

    # 3.8. Plot Boxplots for Numerical Variables
    def plot_boxplots(df_numerical):
        num_cols = 2
        num_rows = math.ceil(len(df_numerical.columns) / num_cols)

        fig, ax = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
        ax = ax.flatten()

        for i, column in enumerate(df_numerical.columns):
            sns.boxplot(data=df_numerical[column], ax=ax[i], color="lightblue")
            ax[i].set_title(f'Boxplot of {column}')

        plt.tight_layout()
        return fig

    # Call all functions and display results in Streamlit
    st.subheader("Define Categorical and Numerical Variables")
    st.write(f"Categorical Variables: {df_categorical.columns}.")
    st.write(f"Numerical Variables: {df_numerical.columns}.")

    st.subheader(
        "Frequency and Proportion Tables for Categorical Variables")
    frequency_proportion_tables = generate_frequency_proportion_tables(
        df_categorical)
    for col, table in frequency_proportion_tables.items():
        st.write(f"{col}:\n")
        st.table(table)

    st.subheader("Plot Categorical Barplots")
    st.pyplot(plot_categorical_barplots(df_categorical))

    st.subheader("Plot Categorical Pie Charts")
    st.pyplot(plot_categorical_pie_charts(df_categorical))

    st.subheader("Summary Statistics for Numerical Variables")
    summary_stats = summary_statistics(df_numerical)
    st.table(summary_stats)

    st.subheader("Skewness and Kurtosis for Numerical Variables")
    skewness_kurtosis = calculate_skewness_kurtosis(df_numerical)
    st.table(skewness_kurtosis)

    st.subheader("Plot Histograms for Numerical Variables")
    st.pyplot(plot_histograms(df_numerical))

    st.subheader("Plot Boxplots")
    st.pyplot(plot_boxplots(df_numerical))


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
