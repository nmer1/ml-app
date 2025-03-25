import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import json
import io
import sys
from zipfile import ZipFile
from sklearn.utils.validation import check_is_fitted
from scipy.stats import skew
from sklearn.preprocessing import PolynomialFeatures
import inspect
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def safe_model_init(m_class):
    try:
        params = inspect.signature(m_class).parameters
        if 'verbose' in params:
            return m_class(verbose=0)
        else:
            return m_class()
    except:
        return m_class()

# Optional advanced libraries
try:
    import plotly.express as px
    plotly_installed = True
except ImportError:
    plotly_installed = False

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    statsmodels_installed = True
except ImportError:
    statsmodels_installed = False

try:
    from sklearn.impute import KNNImputer
    knn_imputer_installed = True
except ImportError:
    knn_imputer_installed = False

try:
    import shap
    shap_installed = True
except ImportError:
    shap_installed = False

try:
    from flaml import AutoML
    flaml_installed = True
except ImportError:
    flaml_installed = False

try:
    import lime
    import lime.lime_tabular
    lime_installed = True
except ImportError:
    lime_installed = False

warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score,
    f1_score
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# --------------------------------------------------------------------------------
# Custom Transformers
# --------------------------------------------------------------------------------
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        numeric_df = X.select_dtypes(include=[np.number])
        for col in numeric_df.columns:
            self.lower_bounds_[col] = numeric_df[col].quantile(self.lower_quantile)
            self.upper_bounds_[col] = numeric_df[col].quantile(self.upper_quantile)
        return self

    def transform(self, X):
        X = X.copy()
        numeric_df = X.select_dtypes(include=[np.number])
        for col in numeric_df.columns:
            lower_cap = self.lower_bounds_[col]
            upper_cap = self.upper_bounds_[col]
            X[col] = np.where(X[col] < lower_cap, lower_cap, X[col])
            X[col] = np.where(X[col] > upper_cap, upper_cap, X[col])
        return X

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        numeric_df = X.select_dtypes(include=[np.number])
        for col in numeric_df.columns:
            min_val = X[col].min()
            if min_val <= 0:
                shift_val = 1 - min_val
                X[col] = X[col] + shift_val
            X[col] = np.log1p(X[col])
        return X

# --------------------------------------------------------------------------------
# Recommendation Functions
# --------------------------------------------------------------------------------
def get_missing_value_recommendation(col_data):
    missing_pct = col_data.isna().mean() * 100
    if missing_pct == 0:
        return "No action needed"
    elif pd.api.types.is_numeric_dtype(col_data):
        skewness = skew(col_data.dropna())
        if abs(skewness) > 1:
            return "Impute with median"
        else:
            return "Impute with mean"
    else:
        return "Impute with mode"

def get_outlier_recommendation(col_data):
    if not pd.api.types.is_numeric_dtype(col_data):
        return "Not applicable"
    skewness = skew(col_data.dropna())
    q1 = col_data.quantile(0.25)
    q3 = col_data.quantile(0.75)
    iqr = q3 - q1
    outliers = col_data[(col_data < q1 - 1.5 * iqr) | (col_data > q3 + 1.5 * iqr)]
    outlier_pct = len(outliers) / len(col_data) * 100
    if abs(skewness) > 1:
        return "Log Transform"
    elif outlier_pct > 5:
        return "Winsorize (1%-99%)"
    else:
        return "No action needed"

# --------------------------------------------------------------------------------
# Streamlit Configuration
# --------------------------------------------------------------------------------
st.set_page_config(page_title="Enhanced ML Pro App", layout="wide")

st.title("Enhanced All-in-One ML App (with Cross-Validation)")

st.markdown("""
**Features**:
- **Original**: Data upload, cleaning, EDA, target selection, basic train/test split, multi-model training, etc.
- **Cross-Validation** option
- **Export to Jupyter Notebook**
- And all the advanced features: AutoML, advanced cleaning, interpretability, etc.
""")

# --------------------------------------------------------------------------------
# Initialize Session State
# --------------------------------------------------------------------------------
if "data" not in st.session_state:
    st.session_state.data = None
if "target_col" not in st.session_state:
    st.session_state.target_col = None
if "problem_type" not in st.session_state:
    st.session_state.problem_type = None
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_train" not in st.session_state:
    st.session_state.y_train = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "num_cols" not in st.session_state:
    st.session_state.num_cols = []
if "cat_cols" not in st.session_state:
    st.session_state.cat_cols = []
if "final_models" not in st.session_state:
    st.session_state.final_models = {}
if "transformed_feature_names" not in st.session_state:
    st.session_state.transformed_feature_names = None
if "action_log" not in st.session_state:
    st.session_state.action_log = []
if "true_num_cols" not in st.session_state:
    st.session_state.true_num_cols = []

if "true_cat_cols" not in st.session_state:
    st.session_state.true_cat_cols = []


# --------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    df_sample = pd.DataFrame({
        'age': np.random.randint(18, 90, 100),
        'income': np.random.randint(20000, 150000, 100),
        'education_years': np.random.randint(8, 20, 100),
        'gender': np.random.choice(['Male', 'Female'], 100),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], 100),
        'target': np.random.choice([0, 1], 100)
    })
    return df_sample

def dataframe_info(df):
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

def get_transformed_feature_names(column_transformer, num_cols, cat_cols):
    check_is_fitted(column_transformer)
    feature_names = []
    for name, transformer, cols in column_transformer.transformers_:
        if name == "num":
            if "poly" in transformer.named_steps:
                poly = transformer.named_steps["poly"]
                feature_names.extend(poly.get_feature_names_out(cols))
            else:
                feature_names.extend(cols)
        elif name == "cat":
            ohe = transformer.named_steps["onehot"]
            try:
                check_is_fitted(ohe)
                cat_names = ohe.get_feature_names_out(cols)
                feature_names.extend(cat_names)
            except:
                feature_names.extend([f"{col}_cat_{i}" for i, col in enumerate(cols)])
    return feature_names

def get_models_dict(problem_type):
    if problem_type == "Classification":
        models = {
            "LogisticRegression": LogisticRegression,
            "RandomForestClassifier": RandomForestClassifier,
            "SVC": SVC,
            "KNeighborsClassifier": KNeighborsClassifier,
            "XGBClassifier": XGBClassifier,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "MLPClassifier": MLPClassifier,
            "LGBMClassifier": LGBMClassifier,
            "CatBoostClassifier": CatBoostClassifier
        }
    else:
        models = {
            "LinearRegression": LinearRegression,
            "RandomForestRegressor": RandomForestRegressor,
            "SVR": SVR,
            "KNeighborsRegressor": KNeighborsRegressor,
            "XGBRegressor": XGBRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor,
            "MLPRegressor": MLPRegressor,
            "LGBMRegressor": LGBMRegressor,
            "CatBoostRegressor": CatBoostRegressor
        }
    return models

# --------------------------------------------------------------------------------
# Sidebar Configuration
# --------------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration Panel")
    
    st.subheader("Data Upload")
    file_type = st.selectbox("File Type", ["CSV", "Excel", "JSON"])
    uploaded_file = st.file_uploader(f"Upload your {file_type} file", type=[file_type.lower()])
    if st.button("Use Sample Data"):
        with st.spinner("Loading sample data..."):
            st.session_state.data = generate_sample_data()
            st.success("Sample data loaded!")
            st.session_state.action_log.append({"action": "load_data", "code": "df = generate_sample_data()  # Using sample data"})

    st.subheader("Preprocessing Options")
    scaler_choice = st.selectbox("Feature Scaling", ["StandardScaler", "MinMaxScaler"])
    encoder_choice = st.selectbox("Feature Encoding", ["OneHotEncoder"])
    use_poly_features = st.checkbox("Generate Polynomial Features", help="Adds polynomial and interaction terms to numeric features.")

    st.subheader("Train/Test Split")
    test_size = st.slider("Test Size (%)", 10, 40, 20)
    random_state = st.number_input("Random State", 0, 100, 42)
    stratify = st.checkbox("Stratify Split (Classification)", value=True)
    
    st.subheader("Cross-Validation")
    use_cross_val = st.checkbox("Use Cross-Validation?", help="Enable to evaluate with multiple folds.")
    cv_folds = st.slider("Number of CV Folds", 2, 10, 5, help="Number of folds for cross-validation.")
    if st.session_state.get("problem_type") == "Classification":
        cv_scoring = st.selectbox("Scoring Metric", ["accuracy", "f1", "roc_auc", "precision", "recall"], index=0)
    elif st.session_state.get("problem_type") == "Regression":
        cv_scoring = st.selectbox("Scoring Metric", ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"], index=0)
    else:
        cv_scoring = None

# --------------------------------------------------------------------------------
# Main App Logic
# --------------------------------------------------------------------------------
if uploaded_file is not None and st.session_state.data is None:
    with st.spinner("Loading data..."):
        try:
            if file_type == "CSV":
                st.session_state.data = pd.read_csv(uploaded_file)
            elif file_type == "Excel":
                st.session_state.data = pd.read_excel(uploaded_file)
            elif file_type == "JSON":
                st.session_state.data = pd.read_json(uploaded_file)
            if st.session_state.data.empty:
                st.error("Uploaded file is empty. Please upload a valid dataset.")
            else:
                st.success(f"Data loaded: {st.session_state.data.shape}")
                st.session_state.action_log.append({"action": "load_data", "code": f"df = pd.read_{file_type.lower()}('your_data.{file_type.lower()}')  # Shape: {st.session_state.data.shape}"})
        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty. Please upload a valid dataset.")
        except pd.errors.ParserError:
            st.error("Error parsing the file. Please ensure it's a valid CSV, Excel, or JSON file.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Data Cleaning & EDA
if st.session_state.data is not None:
    st.header("Data Cleaning & EDA")
    st.info("Clean your data by removing duplicates, handling missing values, and managing outliers. Use EDA to understand your data's structure.")
    data = st.session_state.data.copy()

    num_cols = data.select_dtypes(include=["int", "float"]).columns
    potential_cat = [col for col in num_cols if data[col].nunique() < 20]
    if potential_cat:
        st.write("**Potential Categorical Columns (numeric with <20 unique values):**", potential_cat)

    with st.expander("Preview Original Data & Info", expanded=True):
        st.dataframe(data.head(10))
        st.write(f"Shape: {data.shape}")
        st.text(dataframe_info(data))
        st.write("Missing Values:", data.isna().sum())
        st.write("Unique Values:", data.nunique())

    st.subheader("Duplicates")
    dup_cols = st.multiselect("Columns to define duplicates (leave empty to check all)", data.columns.tolist())
    keep_opt = st.selectbox("Keep", ["first", "last", "none"])
    if st.checkbox("Preview duplicates"):
        if dup_cols:
            mask_dup = data.duplicated(subset=dup_cols, keep=False)
        else:
            mask_dup = data.duplicated(keep=False)
        dup_df = data[mask_dup]
        if dup_df.empty:
            st.write("No duplicates found.")
        else:
            st.warning(f"Found {dup_df.shape[0]} duplicates.")
            st.dataframe(dup_df)

    if st.checkbox("Remove duplicates"):
        if st.button("Confirm duplicate removal"):
            orig_shape = data.shape
            if dup_cols:
                data = data.drop_duplicates(subset=dup_cols, keep=keep_opt if keep_opt != "none" else False)
            else:
                data = data.drop_duplicates(keep=keep_opt if keep_opt != "none" else False)
            st.success(f"Removed duplicates. {orig_shape} -> {data.shape}")
            st.subheader("Preview after duplicate removal")
            st.dataframe(data.head(10))
            st.session_state.data = data
            subset_str = f"subset={dup_cols}" if dup_cols else ""
            st.session_state.action_log.append({"action": "remove_duplicates", "code": f"df = df.drop_duplicates({subset_str}, keep='{keep_opt}')  # {orig_shape} -> {data.shape}"})

    st.subheader("Missing Values")
    if st.checkbox("Show missing-value summary"):
        st.write(data.isna().sum())

    if st.checkbox("Show missing-value recommendations"):
        from scipy.stats import skew
        recommendations = []
        for col in data.columns:
            rec = get_missing_value_recommendation(data[col])
            col_type = "Numerical" if pd.api.types.is_numeric_dtype(data[col]) else "Categorical"
            missing_pct = data[col].isna().mean() * 100
            recommendations.append({
                "Column": col,
                "Type": col_type,
                "Missing %": f"{missing_pct:.2f}%",
                "Recommendation": rec
            })
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(rec_df)

    if st.button("Apply Recommended Missing Value Imputations"):
        with st.spinner("Applying imputations..."):
            orig_shape = data.shape
            code_lines = []
            for col in data.columns:
                rec = get_missing_value_recommendation(data[col])
                if rec == "Impute with mean":
                    data[col] = data[col].fillna(data[col].mean())
                    code_lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].mean())")
                elif rec == "Impute with median":
                    data[col] = data[col].fillna(data[col].median())
                    code_lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].median())")
                elif rec == "Impute with mode":
                    mode_val = data[col].mode()[0] if not data[col].mode().empty else "Missing"
                    data[col] = data[col].fillna(mode_val)
                    code_lines.append(f"df['{col}'] = df['{col}'].fillna('{mode_val}')")
            st.success(f"Applied strategies. {orig_shape} -> {data.shape}")
            st.subheader("Preview after missing-value handling")
            st.dataframe(data.head(10))
            st.session_state.data = data
            st.session_state.action_log.append({"action": "handle_missing_recommended", "code": "\n".join(code_lines)})

    num_miss_strategies = ["None", "Drop rows", "Impute Mean", "Impute Median", "KNN Impute"]
    cat_miss_strategies = ["None", "Drop rows", "Impute Mode", "Impute with 'Missing'"]
    num_choice = st.selectbox("Choose strategy for numerical columns", num_miss_strategies)
    cat_choice = st.selectbox("Choose strategy for categorical columns", cat_miss_strategies)

    if st.button("Apply missing-value strategies"):
        with st.spinner("Applying missing-value strategies..."):
            orig_shape = data.shape
            code_lines = ["num_cols = df.select_dtypes(include=['number']).columns", "cat_cols = df.select_dtypes(include=['object', 'category']).columns"]
            if num_choice != "None":
                numC = data.select_dtypes(include=[np.number]).columns
                if num_choice == "Drop rows":
                    data = data.dropna(subset=numC)
                    code_lines.append("df = df.dropna(subset=num_cols)")
                elif num_choice == "Impute Mean":
                    for c in numC:
                        data[c] = data[c].fillna(data[c].mean())
                        code_lines.append(f"for col in num_cols:\n    df[col] = df[col].fillna(df[col].mean())")
                    
                elif num_choice == "Impute Median":
                    for c in numC:
                        data[c] = data[c].fillna(data[c].median())
                        code_lines.append(f"for col in num_cols:\n    df[col] = df[col].fillna(df[col].median())")
                    
                elif num_choice == "KNN Impute" and knn_imputer_installed:
                    from sklearn.impute import KNNImputer
                    imputer = KNNImputer(n_neighbors=5)
                    data[numC] = imputer.fit_transform(data[numC])
                    code_lines.append("from sklearn.impute import KNNImputer\nimputer = KNNImputer(n_neighbors=5)\ndf[num_cols] = imputer.fit_transform(df[num_cols])")

            if cat_choice != "None":
                catC = data.select_dtypes(include=['object', 'category']).columns
                if cat_choice == "Drop rows":
                    data = data.dropna(subset=catC)
                    code_lines.append("df = df.dropna(subset=cat_cols)")
                elif cat_choice == "Impute Mode":
                    for c in catC:
                        mode_val = data[c].mode()[0] if not data[c].mode().empty else "Missing"
                        data[c] = data[c].fillna(mode_val)
                        code_lines.append(f"for col in cat_cols:\n    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Missing')")
                    
                elif cat_choice == "Impute with 'Missing'":
                    for c in catC:
                        data[c] = data[c].fillna('Missing')
                        code_lines.append(f"for col in cat_cols:\n    df[col] = df[col].fillna('Missing')")
                    

            st.success(f"Applied strategies. {orig_shape} -> {data.shape}")
            st.session_state.data = data
            st.session_state.action_log.append({"action": "handle_missing", "code": "\n".join(code_lines)})

    st.subheader("Outliers")
    st.info("Outlier handling can improve model performance by reducing the impact of extreme values.")
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if st.session_state.target_col is not None and st.session_state.target_col in data.columns:
        numeric_feature_cols = [col for col in numeric_cols if col != st.session_state.target_col]
    else:
        numeric_feature_cols = numeric_cols

    selected_cols = st.multiselect("Select numeric columns for outlier handling", numeric_cols, default=numeric_feature_cols)

    if selected_cols:
        if st.checkbox("Visualize Outliers") and plotly_installed:
            import plotly.express as px
            fig = px.box(data, y=selected_cols)
            st.plotly_chart(fig)

        if st.checkbox("Show outlier recommendations"):
            recommendations = []
            for col in selected_cols:
                rec = get_outlier_recommendation(data[col])
                from scipy.stats import skew
                skewness = skew(data[col].dropna())
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                outliers = data[col][(data[col] < q1 - 1.5 * iqr) | (data[col] > q3 + 1.5 * iqr)]
                outlier_pct = len(outliers) / len(data[col]) * 100
                recommendations.append({
                    "Column": col,
                    "Skewness": f"{skewness:.2f}",
                    "Outlier %": f"{outlier_pct:.2f}%",
                    "Recommendation": rec
                })
            rec_df = pd.DataFrame(recommendations)
            st.dataframe(rec_df)

        if st.button("Apply Recommended Outlier Transformations"):
            with st.spinner("Handling outliers..."):
                orig_shape = data.shape
                code_lines = []
                for col in selected_cols:
                    rec = get_outlier_recommendation(data[col])
                    if rec == "Winsorize (1%-99%)":
                        win = Winsorizer()
                        data[[col]] = win.fit_transform(data[[col]])
                        code_lines.append(f"from sklearn.base import BaseEstimator, TransformerMixin\nclass Winsorizer(BaseEstimator, TransformerMixin):\n    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):\n        self.lower_quantile = lower_quantile\n        self.upper_quantile = upper_quantile\n        self.lower_bounds_ = {{}}\n        self.upper_bounds_ = {{}}\n    def fit(self, X, y=None):\n        numeric_df = X.select_dtypes(include=[np.number])\n        for col in numeric_df.columns:\n            self.lower_bounds_[col] = numeric_df[col].quantile(self.lower_quantile)\n            self.upper_bounds_[col] = numeric_df[col].quantile(self.upper_quantile)\n        return self\n    def transform(self, X):\n        X = X.copy()\n        numeric_df = X.select_dtypes(include=[np.number])\n        for col in numeric_df.columns:\n            lower_cap = self.lower_bounds_[col]\n            upper_cap = self.upper_bounds_[col]\n            X[col] = np.where(X[col] < lower_cap, lower_cap, X[col])\n            X[col] = np.where(X[col] > upper_cap, upper_cap, X[col])\n        return X\nwin = Winsorizer()\ndf[['{col}']] = win.fit_transform(df[['{col}']])")
                    elif rec == "Log Transform":
                        lt = LogTransformer()
                        data[[col]] = lt.fit_transform(data[[col]])
                        code_lines.append(f"from sklearn.base import BaseEstimator, TransformerMixin\nclass LogTransformer(BaseEstimator, TransformerMixin):\n    def __init__(self):\n        pass\n    def fit(self, X, y=None):\n        return self\n    def transform(self, X):\n        X = X.copy()\n        numeric_df = X.select_dtypes(include=[np.number])\n        for col in numeric_df.columns:\n            min_val = X[col].min()\n            if min_val <= 0:\n                shift_val = 1 - min_val\n                X[col] = X[col] + shift_val\n            X[col] = np.log1p(X[col])\n        return X\nlt = LogTransformer()\ndf[['{col}']] = lt.fit_transform(df[['{col}']])")
                st.success(f"Applied transformations. {orig_shape} -> {data.shape}")
                st.subheader("Preview after transformations")
                st.dataframe(data.head(10))
                st.session_state.data = data
                st.session_state.action_log.append({"action": "handle_outliers_recommended", "code": "\n".join(code_lines)})

        out_action = st.selectbox("Outlier handling method", ["None", "Winsorize (1%-99%)", "Log Transform", "Remove (IQR)"])
        if out_action != "None" and st.button("Apply outlier method"):
            with st.spinner("Applying outlier method..."):
                orig_shape = data.shape
                code_lines = []
                if out_action == "Winsorize (1%-99%)":
                    win = Winsorizer()
                    data[selected_cols] = win.fit_transform(data[selected_cols])
                    code_lines.append(f"from sklearn.base import BaseEstimator, TransformerMixin\nclass Winsorizer(BaseEstimator, TransformerMixin):\n    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):\n        self.lower_quantile = lower_quantile\n        self.upper_quantile = upper_quantile\n        self.lower_bounds_ = {{}}\n        self.upper_bounds_ = {{}}\n    def fit(self, X, y=None):\n        numeric_df = X.select_dtypes(include=[np.number])\n        for col in numeric_df.columns:\n            self.lower_bounds_[col] = numeric_df[col].quantile(self.lower_quantile)\n            self.upper_bounds_[col] = numeric_df[col].quantile(self.upper_quantile)\n        return self\n    def transform(self, X):\n        X = X.copy()\n        numeric_df = X.select_dtypes(include=[np.number])\n        for col in numeric_df.columns:\n            lower_cap = self.lower_bounds_[col]\n            upper_cap = self.upper_bounds_[col]\n            X[col] = np.where(X[col] < lower_cap, lower_cap, X[col])\n            X[col] = np.where(X[col] > upper_cap, upper_cap, X[col])\n        return X\nwin = Winsorizer()\ndf[{selected_cols}] = win.fit_transform(df[{selected_cols}])")
                elif out_action == "Log Transform":
                    lt = LogTransformer()
                    data[selected_cols] = lt.fit_transform(data[selected_cols])
                    code_lines.append(f"from sklearn.base import BaseEstimator, TransformerMixin\nclass LogTransformer(BaseEstimator, TransformerMixin):\n    def __init__(self):\n        pass\n    def fit(self, X, y=None):\n        return self\n    def transform(self, X):\n        X = X.copy()\n        numeric_df = X.select_dtypes(include=[np.number])\n        for col in numeric_df.columns:\n            min_val = X[col].min()\n            if min_val <= 0:\n                shift_val = 1 - min_val\n                X[col] = X[col] + shift_val\n            X[col] = np.log1p(X[col])\n        return X\nlt = LogTransformer()\ndf[{selected_cols}] = lt.fit_transform(df[{selected_cols}])")
                elif out_action == "Remove (IQR)":
                    for col in selected_cols:
                        Q1 = data[col].quantile(0.25)
                        Q3 = data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        data = data[(data[col] >= lower) & (data[col] <= upper)]
                    code_lines.append(f"for col in {selected_cols}:\n    Q1 = df[col].quantile(0.25)\n    Q3 = df[col].quantile(0.75)\n    IQR = Q3 - Q1\n    lower = Q1 - 1.5 * IQR\n    upper = Q3 + 1.5 * IQR\n    df = df[(df[col] >= lower) & (df[col] <= upper)]")
                st.success(f"Outlier method '{out_action}' applied. {orig_shape} -> {data.shape}")
                st.subheader("Preview after outlier handling")
                st.dataframe(data.head(10))
                st.session_state.data = data
                st.session_state.action_log.append({"action": "handle_outliers", "code": "\n".join(code_lines)})

    if st.button("Finalize Cleaning"):
        st.success("Data cleaning applied step by step. No additional changes.")
        st.write("Action Log:", [log["action"] for log in st.session_state.action_log])

    st.subheader("Enhanced EDA")
    with st.expander("Summary Statistics"):
        st.write(data.describe())

    with st.expander("Correlation Heatmap (numeric)"):
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_cols) > 1:
            corr = data[numerical_cols].corr()
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(corr, annot=True, cmap='Blues', ax=ax)
            st.pyplot(fig)
        else:
            st.write("Not enough numeric columns for correlation heatmap.")

    if plotly_installed:
        import plotly.express as px
        with st.expander("Interactive Pair Plot"):
            pair_cols = st.multiselect("Select columns for pair plot", data.columns, default=data.columns[:4])
            if len(pair_cols) >= 2:
                fig_pair = px.scatter_matrix(data[pair_cols])
                st.plotly_chart(fig_pair)

        with st.expander("Interactive Histogram"):
            if numerical_cols:
                hist_col = st.selectbox("Select numeric column for histogram", numerical_cols)
                figp = px.histogram(data, x=hist_col, nbins=30, marginal='box')
                st.plotly_chart(figp)

    if statsmodels_installed:
        with st.expander("Check VIF"):
            if st.button("Compute VIF"):
                df_vif = data[numerical_cols].dropna()
                vif_df = pd.DataFrame()
                vif_df["feature"] = df_vif.columns
                vif_df["VIF"] = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]
                st.dataframe(vif_df)

# Target Selection & Problem Type
if st.session_state.data is not None:
    st.header("Target Selection & Problem Type")
    st.info("Select your target variable and specify if it's a classification or regression task.")
    data = st.session_state.data.copy()
    all_cols = data.columns.tolist()
    if len(all_cols) >= 2:
        suggested_targets = [col for col in all_cols if data[col].nunique() < 10]
        combined_targets = list(dict.fromkeys(suggested_targets + all_cols))
        target_sel = st.selectbox("Choose target column", combined_targets)
        prob_sel = st.selectbox("Choose problem type", ["Classification", "Regression"])
        if st.button("Confirm Target"):
            if target_sel not in data.columns:
                st.error(f"'{target_sel}' is not in data columns.")
            else:
                st.session_state.target_col = target_sel
                st.session_state.problem_type = prob_sel
                st.success(f"Target set to '{target_sel}' as {prob_sel} problem.")
                st.session_state.action_log.append({"action": "select_target", "code": f"# Selected target column: '{target_sel}' for {prob_sel} problem"})
                if prob_sel == "Classification" and plotly_installed:
                    import plotly.express as px
                    fig = px.bar(data[target_sel].value_counts(), title="Class Distribution")
                    st.plotly_chart(fig)
                    if (data[target_sel].value_counts().min() / data[target_sel].value_counts().sum()) < 0.1:
                        st.warning("Class imbalance detected.")
    else:
        st.warning("Need at least 2 columns to pick a target + have features.")

# Train/Test Split & Data Validation
if st.session_state.target_col is not None and st.session_state.target_col in st.session_state.data.columns:
    st.header("Train/Test Split")
    data = st.session_state.data.copy()
    st.write(f"Using cleaned data for splitting. Shape: {data.shape}")
    st.subheader("Preview data before splitting")
    st.dataframe(data.head(10))

    if st.button("Perform Split"):
        with st.spinner("Splitting data..."):
            X = data.drop(labels=[st.session_state.target_col], axis=1, errors='ignore')
            y = data[st.session_state.target_col]
            stratify_y = y if (stratify and st.session_state.problem_type == "Classification") else None

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=random_state, stratify=stratify_y
            )
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            st.session_state.cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            categorical_like_nums = [col for col in numeric_cols if X_train[col].nunique() < 20]
            st.session_state.true_num_cols = [col for col in numeric_cols if col not in categorical_like_nums]
            st.session_state.true_cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist() + categorical_like_nums
            st.success(f"Split done. Train: {X_train.shape}, Test: {X_test.shape}")
            st.subheader("Preview training data")
            st.dataframe(X_train.head(10))
            stratify_code = "stratify=y" if stratify and st.session_state.problem_type == "Classification" else ""
            st.session_state.action_log.append({"action": "train_test_split", "code": f"X = df.drop('{st.session_state.target_col}', axis=1)\ny = df['{st.session_state.target_col}']\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size/100}, random_state={random_state}, {stratify_code})"})
            if st.session_state.problem_type == "Classification" and plotly_installed:
                import plotly.express as px
                fig_train = px.bar(y_train.value_counts(), title="Train Class Distribution")
                fig_test = px.bar(y_test.value_counts(), title="Test Class Distribution")
                st.plotly_chart(fig_train)
                st.plotly_chart(fig_test)

    if st.session_state.X_train is not None:
        st.header("Data Validation")
        st.info("Validating data to ensure it's suitable for modeling.")
        if st.session_state.problem_type == "Regression" and statsmodels_installed:
            num_cols = st.session_state.num_cols
            if len(num_cols) > 1:
                df_vif = st.session_state.X_train[num_cols].dropna()
                vif_df = pd.DataFrame()
                vif_df["feature"] = df_vif.columns
                vif_df["VIF"] = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]
                high_vif = vif_df[vif_df["VIF"] > 10]
                if not high_vif.empty:
                    st.warning("High multicollinearity detected. Consider removing or combining these features:")
                    st.dataframe(high_vif)

        # Check target type consistency
        if st.session_state.problem_type == "Classification" and pd.api.types.is_numeric_dtype(st.session_state.y_train):
            st.warning("Target is numeric but problem type is classification. Ensure it's categorical or change problem type.")
        elif st.session_state.problem_type == "Regression" and not pd.api.types.is_numeric_dtype(st.session_state.y_train):
            st.error("Target must be numeric for regression. Please select a different target or problem type.")

        # Check for infinite values in numeric features
        X_numeric = st.session_state.X_train.select_dtypes(include=[np.number])
        if np.isinf(X_numeric).values.any():
            st.error("Infinite values found in numeric features. Please handle them before training.")

        # Check for infinite values in target (if numeric)
        y_train = st.session_state.y_train
        if pd.api.types.is_numeric_dtype(y_train):
            if np.isinf(y_train).values.any():
                st.error("Infinite values found in the target column. Please handle them before training.")

# Model Selection & Training
if st.session_state.X_train is not None:
    st.header("Model Selection & Training")
    st.info("Select models, configure hyperparameters, and optionally evaluate with cross-validation.")
    model_dict = get_models_dict(st.session_state.problem_type)
    model_options = ["AutoML"] + list(model_dict.keys()) if flaml_installed else list(model_dict.keys())
    chosen_models = st.multiselect("Select models to train", model_options, default=[])

    adv_mode = st.checkbox("Advanced: Hyperparameter Tuning", help="Enable to set model-specific hyperparameters.")
    param_grids = {}
    if adv_mode:
        search_type = st.selectbox("Search type", ["GridSearch", "RandomSearch"])
        if search_type == "RandomSearch":
            n_iter_val = st.slider("Number of random parameter combos", 5, 50, 10)
        for model_name in chosen_models:
            if model_name != "AutoML":
                with st.expander(f"Hyperparameters for {model_name}"):
                    if model_name in ["RandomForestClassifier", "RandomForestRegressor"]:
                        n_estimators = st.multiselect("Number of Trees", [50, 100, 150, 200], default=[100], key=f"{model_name}_n_estimators")
                        max_depth = st.multiselect("Max Depth", [5, 10, 15, 20], default=[10], key=f"{model_name}_max_depth")
                        param_grids[model_name] = {
                            "model__n_estimators": n_estimators,
                            "model__max_depth": max_depth
                        }

                    elif model_name in ["LogisticRegression"]:
                        C = st.multiselect("Regularization Strength (C)", [0.1, 1, 10], default=[1], key=f"{model_name}_C")
                        penalty = st.multiselect("Penalty", ["l1", "l2"], default=["l2"], key=f"{model_name}_penalty")
                        param_grids[model_name] = {
                            "model__C": C,
                            "model__penalty": penalty,
                            "model__solver": ["liblinear"]  # Needed for l1
                        }

                    elif model_name in ["SVC", "SVR"]:
                        C = st.multiselect("Regularization Strength (C)", [0.1, 1, 10], default=[1], key=f"{model_name}_C")
                        kernel = st.multiselect("Kernel", ["linear", "rbf"], default=["rbf"], key=f"{model_name}_kernel")
                        gamma = st.multiselect("Gamma", ["scale", "auto"], default=["scale"], key=f"{model_name}_gamma")
                        param_grids[model_name] = {
                            "model__C": C,
                            "model__kernel": kernel,
                            "model__gamma": gamma
                        }

                    elif model_name in ["KNeighborsClassifier", "KNeighborsRegressor"]:
                        n_neighbors = st.multiselect("Number of Neighbors", [3, 5, 7, 9], default=[5], key=f"{model_name}_n_neighbors")
                        weights = st.multiselect("Weights", ["uniform", "distance"], default=["uniform"], key=f"{model_name}_weights")
                        param_grids[model_name] = {
                            "model__n_neighbors": n_neighbors,
                            "model__weights": weights
                        }

                    elif model_name in ["XGBClassifier", "XGBRegressor",
                                        "GradientBoostingClassifier", "GradientBoostingRegressor"]:
                        n_estimators = st.multiselect("Number of Estimators", [50, 100, 150], default=[100], key=f"{model_name}_n_estimators")
                        learning_rate = st.multiselect("Learning Rate", [0.01, 0.1, 0.2], default=[0.1], key=f"{model_name}_learning_rate")
                        max_depth = st.multiselect("Max Depth", [3, 5, 7], default=[3], key=f"{model_name}_max_depth")
                        param_grids[model_name] = {
                            "model__n_estimators": n_estimators,
                            "model__learning_rate": learning_rate,
                            "model__max_depth": max_depth
                        }

                    elif model_name in ["MLPClassifier", "MLPRegressor"]:
                        hidden_layer_sizes = st.multiselect("Hidden Layer Sizes", [(50,), (100,), (100, 50)], default=[(100,)], key=f"{model_name}_hls")
                        alpha = st.multiselect("Alpha (L2 Penalty)", [0.0001, 0.001, 0.01], default=[0.0001], key=f"{model_name}_alpha")
                        solver = st.multiselect("Solver", ["adam", "lbfgs", "sgd"], default=["adam"], key=f"{model_name}_solver")
                        param_grids[model_name] = {
                            "model__hidden_layer_sizes": hidden_layer_sizes,
                            "model__alpha": alpha,
                            "model__solver": solver
                        }

                    elif model_name in ["LGBMClassifier", "LGBMRegressor"]:
                        n_estimators = st.multiselect("Number of Estimators", [50, 100, 150], default=[100], key=f"{model_name}_n_estimators")
                        learning_rate = st.multiselect("Learning Rate", [0.01, 0.1, 0.3], default=[0.1], key=f"{model_name}_learning_rate")
                        max_depth = st.multiselect("Max Depth", [5, 10, 15], default=[10], key=f"{model_name}_max_depth")
                        param_grids[model_name] = {
                            "model__n_estimators": n_estimators,
                            "model__learning_rate": learning_rate,
                            "model__max_depth": max_depth
                        }

                    elif model_name in ["CatBoostClassifier", "CatBoostRegressor"]:
                        iterations = st.multiselect("Iterations", [100, 200, 300], default=[100], key=f"{model_name}_iterations")
                        depth = st.multiselect("Depth", [4, 6, 8], default=[6], key=f"{model_name}_depth")
                        learning_rate = st.multiselect("Learning Rate", [0.01, 0.05, 0.1], default=[0.1], key=f"{model_name}_learning_rate")
                        param_grids[model_name] = {
                            "model__iterations": iterations,
                            "model__depth": depth,
                            "model__learning_rate": learning_rate
                        }

                    else:
                        st.write("⚠️ No hyperparameter tuning options defined for this model.")
                        param_grids[model_name] = {}

    if st.button("Train Selected Models"):
        with st.spinner("Training models..."):
            st.session_state.final_models = {}
            X_train = st.session_state.X_train
            y_train = st.session_state.y_train
            num_cols = st.session_state.num_cols
            cat_cols = st.session_state.cat_cols

            # Preprocessing pipeline
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy='median')),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)) if use_poly_features else ("identity", "passthrough"),
                ("scaler", StandardScaler() if scaler_choice == "StandardScaler" else MinMaxScaler())
            ])
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("onehot", OneHotEncoder(handle_unknown='ignore'))
            ])
            preprocessor = ColumnTransformer([
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols)
            ])
            preprocessor.fit(X_train)
            st.session_state.transformed_feature_names = get_transformed_feature_names(preprocessor, num_cols, cat_cols)

            total_models = len(chosen_models)
            progress_bar = st.progress(0)
            for i, model_name in enumerate(chosen_models):
                st.write(f"**Training {model_name}**...")
                try:
                    if model_name == "AutoML" and flaml_installed:
                        automl = AutoML()
                        automl.fit(
                            X_train, y_train,
                            task="classification" if st.session_state.problem_type == "Classification" else "regression",
                            time_budget=60
                        )
                        st.session_state.final_models["AutoML"] = automl
                        st.session_state.action_log.append({"action": "train_model", "code": "from flaml import AutoML\nautoml = AutoML()\nautoml.fit(X_train, y_train, task='{st.session_state.problem_type.lower()}', time_budget=60)"})

                    else:
                        m_class = model_dict[model_name]
                        base_pipe = Pipeline([
                            ("preprocessor", preprocessor),
                            ("model", safe_model_init(m_class))
                        ])

                        # (A) Hyperparam Tuning
                        if adv_mode and model_name in param_grids and param_grids[model_name]:
                            if search_type == "GridSearch":
                                gs = GridSearchCV(base_pipe, param_grid=param_grids[model_name], cv=cv_folds, scoring=cv_scoring, n_jobs=-1, verbose=0)
                                gs.fit(X_train, y_train)
                                best_model = gs.best_estimator_
                                st.write(f"Best Params for {model_name}: {gs.best_params_}")
                                st.write(f"Best CV Score ({cv_scoring}): {gs.best_score_:.4f}")
                                st.session_state.action_log.append({"action": "hyperparam_tuning", "code": f"gs = GridSearchCV(base_pipe, param_grid={param_grids[model_name]}, cv={cv_folds}, scoring='{cv_scoring}', n_jobs=-1, verbose=0)\ngs.fit(X_train, y_train)\nbest_model = gs.best_estimator_"})
                            else:
                                rs = RandomizedSearchCV(
                                    base_pipe, param_distributions=param_grids[model_name], 
                                    n_iter=n_iter_val, cv=cv_folds, scoring=cv_scoring, n_jobs=-1, verbose=0, random_state=42
                                )
                                rs.fit(X_train, y_train)
                                best_model = rs.best_estimator_
                                st.write(f"Best Params for {model_name}: {rs.best_params_}")
                                st.write(f"Best CV Score ({cv_scoring}): {rs.best_score_:.4f}")
                                st.session_state.action_log.append({"action": "hyperparam_tuning", "code": f"rs = RandomizedSearchCV(base_pipe, param_distributions={param_grids[model_name]}, n_iter={n_iter_val}, cv={cv_folds}, scoring='{cv_scoring}', n_jobs=-1, verbose=0, random_state=42)\nrs.fit(X_train, y_train)\nbest_model = rs.best_estimator_"})
                        else:
                            # (B) No hyperparam tuning
                            best_model = base_pipe
                            if use_cross_val and cv_scoring:
                                st.write("**Cross-Validation Results**")
                                cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring=cv_scoring, n_jobs=-1)
                                st.write(f"CV {cv_scoring}: {cv_scores}")
                                st.write(f"Average CV {cv_scoring}: {cv_scores.mean():.4f}")
                            best_model.fit(X_train, y_train)
                            st.session_state.action_log.append({"action": "train_model", "code": f"best_model = Pipeline([('preprocessor', preprocessor), ('model', {model_name}())])\nbest_model.fit(X_train, y_train)" if not use_cross_val else f"best_model = Pipeline([('preprocessor', preprocessor), ('model', {model_name}())])\ncv_scores = cross_val_score(best_model, X_train, y_train, cv={cv_folds}, scoring='{cv_scoring}', n_jobs=-1)\nbest_model.fit(X_train, y_train)"})

                        st.session_state.final_models[model_name] = best_model
                        # Save parameter grid for export
                        if adv_mode and model_name in param_grids:
                            if "param_grids" not in st.session_state:
                                st.session_state.param_grids = {}
                            st.session_state.param_grids[model_name] = param_grids[model_name]


                except Exception as e:
                    st.error(f"Training {model_name} failed: {e}")

                progress_bar.progress((i + 1) / total_models)

            st.success("Training completed!")

# Model Evaluation & Interpretability
if st.session_state.final_models:
    st.header("Model Evaluation & Interpretability")
    st.info("Evaluate models on the test set and explore predictions with SHAP or LIME.")
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    ptype = st.session_state.problem_type

    results = {}
    for m_name, model in st.session_state.final_models.items():
        try:
            y_pred = model.predict(X_test)
            if ptype == "Classification":
                acc = accuracy_score(y_test, y_pred)
                f1_sc = f1_score(y_test, y_pred, average="weighted")
                auc_ = (roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                        if hasattr(model, "predict_proba") else None)
                c_report = classification_report(y_test, y_pred, output_dict=True)
                results[m_name] = {
                    "Accuracy": acc,
                    "F1-score": f1_sc,
                    "AUC": auc_,
                    "Precision": c_report["weighted avg"]["precision"],
                    "Recall": c_report["weighted avg"]["recall"]
                }
            else:
                mse_ = mean_squared_error(y_test, y_pred)
                rmse_ = np.sqrt(mse_)
                r2_ = r2_score(y_test, y_pred)
                results[m_name] = {"MSE": mse_, "RMSE": rmse_, "R²": r2_}
        except Exception as e:
            st.error(f"Evaluation for {m_name} failed: {e}")

    st.subheader("Model Comparison")
    st.dataframe(pd.DataFrame(results).T)

    chosen_model = st.selectbox("Select model for detailed evaluation", list(st.session_state.final_models.keys()))
    pipe = st.session_state.final_models[chosen_model]
    y_pred = pipe.predict(X_test)

    if ptype == "Classification":
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        if hasattr(pipe, "predict_proba"):
            pr = pipe.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, pr)
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, pr):.2f}")
            ax2.plot([0, 1], [0, 1], "--")
            ax2.set_xlabel("FPR")
            ax2.set_ylabel("TPR")
            ax2.legend()
            st.pyplot(fig2)
    else:
        if plotly_installed:
            import plotly.express as px
            fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, title=f"{chosen_model}: Actual vs Predicted")
            fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(dash="dash"))
            st.plotly_chart(fig)
        else:
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)

    st.subheader("Model Explainability")
    # SHAP Section
    if shap_installed and st.button("Show SHAP Summary"):
        with st.spinner("Calculating SHAP values..."):
            try:
                if hasattr(pipe, 'named_steps'):
                    final_model = pipe.named_steps["model"]
                    X_train_transformed = pipe.named_steps["preprocessor"].transform(st.session_state.X_train)
                    X_test_transformed = pipe.named_steps["preprocessor"].transform(st.session_state.X_test)
                    feature_names = st.session_state.transformed_feature_names
                else:
                    final_model = pipe.model
                    X_train_transformed = st.session_state.X_train
                    X_test_transformed = st.session_state.X_test
                    feature_names = X_train_transformed.columns.tolist()

                import shap
                X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names).astype(float)
                X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names).astype(float)

                if ptype == "Classification":
                    def predict_fn(X):
                        if hasattr(final_model, 'predict_proba'):
                            return final_model.predict_proba(X)
                        else:
                            preds = final_model.predict(X)
                            return np.column_stack([1-preds, preds])
                else:
                    def predict_fn(X):
                        return final_model.predict(X)

                explainer = shap.Explainer(predict_fn, X_train_df)
                shap_values = explainer(X_test_df)

                if ptype == "Classification":
                    if getattr(shap_values, "values", None) is not None:
                        if shap_values.values.ndim == 3:
                            shap_values_to_plot = shap_values[...,1]
                        else:
                            shap_values_to_plot = shap_values.values
                    else:
                        shap_values_to_plot = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values
                else:
                    shap_values_to_plot = shap_values.values if hasattr(shap_values, 'values') else shap_values

                fig, ax = plt.subplots()
                shap.summary_plot(shap_values_to_plot, X_test_df, feature_names=feature_names, show=False)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"SHAP calculation failed: {e}")

    # LIME Section
    if lime_installed and st.button("Show LIME Explanation"):
        with st.spinner("Calculating LIME explanation..."):
            try:
                pipe = st.session_state.final_models[chosen_model]
                ptype = st.session_state.problem_type

                X_train_original = st.session_state.X_train.copy()
                X_test_original = st.session_state.X_test.copy()
                feature_names = X_train_original.columns.tolist()

                from sklearn.preprocessing import LabelEncoder
                encoders = {}
                categorical_names = {}
                for col in st.session_state.cat_cols:
                    if col in X_train_original.columns:
                        le = LabelEncoder()
                        X_train_original[col] = le.fit_transform(X_train_original[col].astype(str))
                        X_test_original[col] = le.transform(X_test_original[col].astype(str))
                        encoders[col] = le
                        col_idx = feature_names.index(col)
                        categorical_names[col_idx] = list(le.classes_)

                X_train_trans = X_train_original.values
                X_test_trans = X_test_original.values

                def predict_fn(X):
                    X_df = pd.DataFrame(X, columns=feature_names)
                    for col, le in encoders.items():
                        X_df[col] = le.inverse_transform(X_df[col].astype(int))
                    if ptype == "Classification":
                        if hasattr(pipe, 'predict_proba'):
                            return pipe.predict_proba(X_df)
                        else:
                            preds = pipe.predict(X_df)
                            return np.column_stack([1 - preds, preds])
                    else:
                        return pipe.predict(X_df)

                import lime.lime_tabular
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=X_train_trans,
                    feature_names=feature_names,
                    categorical_features=[feature_names.index(c) for c in st.session_state.cat_cols if c in feature_names],
                    categorical_names=categorical_names,
                    class_names=list(map(str, np.unique(st.session_state.y_train))) if ptype == "Classification" else None,
                    mode="classification" if ptype == "Classification" else "regression"
                )

                instance_idx = st.slider("Select instance index for LIME", 0, len(X_test_trans) - 1, 0)
                instance_array = X_test_trans[instance_idx]

                explanation = explainer.explain_instance(
                    data_row=instance_array,
                    predict_fn=predict_fn,
                    num_features=10
                )
                fig = explanation.as_pyplot_figure()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"LIME explanation failed: {e}")
                st.exception(e)

# Feature Importance
if st.session_state.final_models:
    st.header("Feature Importance")
    chosen_model = st.selectbox("Select model for feature importance", list(st.session_state.final_models.keys()))
    pipeline = st.session_state.final_models[chosen_model]
    if hasattr(pipeline, 'named_steps'):
        model_obj = pipeline.named_steps["model"]
    else:
        model_obj = pipeline.model

    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    if st.session_state.transformed_feature_names:
        if hasattr(model_obj, "feature_importances_"):
            st.subheader("Tree-based Feature Importances")
            importances = model_obj.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            top_n = min(20, len(importances))
            top_idx = sorted_idx[:top_n]

            if plotly_installed:
                import plotly.express as px
                fig = px.bar(
                    x=[st.session_state.transformed_feature_names[i] for i in top_idx][::-1],
                    y=importances[top_idx][::-1],
                    title="Top Feature Importances (Tree-based)"
                )
                st.plotly_chart(fig)
            else:
                fig, ax = plt.subplots()
                ax.barh([st.session_state.transformed_feature_names[i] for i in top_idx][::-1], importances[top_idx][::-1])
                st.pyplot(fig)

        st.subheader("Permutation Importance")
        from sklearn.inspection import permutation_importance
        perm_import = permutation_importance(pipeline, X_test, y_test, n_repeats=5, random_state=42)
        sorted_idx = perm_import.importances_mean.argsort()[::-1]
        top_n = min(20, len(perm_import.importances_mean))
        top_idx = sorted_idx[:top_n]
        feature_names = X_test.columns.tolist()

        if plotly_installed:
            import plotly.express as px
            fig = px.bar(
                x=[feature_names[i] for i in top_idx][::-1],
                y=perm_import.importances_mean[top_idx][::-1],
                title="Top Permutation Importances"
            )
            st.plotly_chart(fig)
        else:
            fig, ax = plt.subplots()
            ax.barh([feature_names[i] for i in top_idx][::-1], perm_import.importances_mean[top_idx][::-1])
            st.pyplot(fig)

# Download & Deployment
if st.session_state.final_models:
    st.header("Download & Deployment")
    st.info("Download your trained model and data, or generate API + Notebook code for deployment.")

    model_keys = list(st.session_state.final_models.keys())
    chosen_dl = st.selectbox("Select model to download", model_keys)

    if st.button("Generate Download Link"):
        final_pipe = st.session_state.final_models[chosen_dl]
        pkl_bytes = pickle.dumps(final_pipe)
        steps_script = "\n".join(
            ["# user_steps_code.py"] + 
            [f"    {log['code'] if 'code' in log else log['action']}" for log in st.session_state.action_log] + 
            [""]
        )
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as zf:
            zf.writestr("model.pkl", pkl_bytes)
            zf.writestr("cleaned_data.csv", st.session_state.data.to_csv(index=False))
            zf.writestr("user_steps_code.py", steps_script)
        st.download_button(
            label="Download Model + Data + Steps (ZIP)",
            data=buffer.getvalue(),
            file_name=f"{chosen_dl}_project.zip",
            mime="application/zip"
        )

    # FastAPI code snippet
    if st.button("Generate API Code"):
        st.code(f'''
from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
async def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {{"prediction": prediction.tolist()}}
''', language="python")
        st.write("Save as `app.py`, then run `uvicorn app:app --reload`.")

  # Export to Jupyter Notebook
if (
    st.session_state.get("final_models") and
    st.session_state.get("X_train") is not None and
    st.session_state.get("y_train") is not None and
    st.session_state.get("X_test") is not None and
    st.session_state.get("y_test") is not None and
    st.session_state.get("true_num_cols") and
    st.session_state.get("true_cat_cols")
):
    st.header("Export to Jupyter Notebook")
    st.info("Download a full notebook with all the steps you followed.")
    trained_models = list(st.session_state.final_models.keys())
    chosen_model = st.selectbox("Select model to include in notebook", trained_models, key="notebook_model")
    if st.button("Export to Jupyter Notebook"):
        

        pipeline = st.session_state.final_models[chosen_model]
        model_obj = pipeline.steps[-1][1] if hasattr(pipeline, 'steps') else pipeline.model
        model_class = model_obj.__class__
        model_name = model_class.__name__
        model_module = model_class.__module__
        model_params = model_obj.get_params()

        # 🧠 Fetch columns (safe)
        num_cols = st.session_state.get("true_num_cols", [])
        cat_cols = st.session_state.get("true_cat_cols", [])

        # Create new notebook
        nb = new_notebook()

        # --- Import code
        import_code = f"""
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from {model_module} import {model_name}
    from sklearn.metrics import accuracy_score, mean_squared_error
    """

        # --- Load + cleaning code
        load_data_code = next((log["code"] for log in st.session_state.action_log if log["action"] == "load_data"), "# Load your data here\n# df = pd.read_csv('your_data.csv')")
        cleaning_codes = [log["code"] for log in st.session_state.action_log if log["action"] in [
            "remove_duplicates", "handle_missing", "handle_missing_recommended",
            "handle_outliers", "handle_outliers_recommended"
        ]]
        cleaning_code = "\n".join(cleaning_codes) if cleaning_codes else "# No data cleaning steps applied"
        split_code = next((log["code"] for log in st.session_state.action_log if log["action"] == "train_test_split"), "# Perform train/test split")

        # --- Preprocessor code
        preprocessor_code = f"""
    # Define final numerical and categorical columns
    num_cols = {num_cols}
    cat_cols = {cat_cols}

    # Define preprocessor
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
    """

        # --- Model + grid search if exists
        model_code = ""
        param_grid = st.session_state.get("param_grids", {}).get(chosen_model, {})

        if param_grid:
            nb.cells.append(new_code_cell(f"# Parameter Grid\nparam_grid = {json.dumps(param_grid, indent=4)}"))
            model_code = f"""
        from sklearn.model_selection import GridSearchCV

        base_model = {model_name}()
        param_grid = {param_grid}

        search = GridSearchCV(
            base_model,
            param_grid,
            cv={cv_folds},
            scoring='{cv_scoring}'
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        """.strip()
        else:
            model_code = f"model = {model_name}(**{model_params})"


        # --- Pipeline and training code
        pipeline_code = "pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])"
        train_code = "pipeline.fit(X_train, y_train)"

        # --- Evaluation code
        eval_code = """
    y_pred = pipeline.predict(X_test)
    """ + ("""
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    """ if st.session_state.problem_type == "Classification" else """
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE: {mse}")
    """)

        # --- Add all cells to notebook
        nb.cells.append(new_markdown_cell("# ML Pipeline Generated by Streamlit App"))
        nb.cells.append(new_code_cell(import_code.strip()))
        nb.cells.append(new_code_cell(load_data_code))
        nb.cells.append(new_code_cell("# Data Cleaning\n" + cleaning_code))
        nb.cells.append(new_code_cell(split_code))
        nb.cells.append(new_code_cell(preprocessor_code))
        nb.cells.append(new_code_cell("# Define Model\n" + model_code))
        nb.cells.append(new_code_cell("# Create Pipeline and Train\n" + pipeline_code + "\n" + train_code))
        nb.cells.append(new_code_cell("# Evaluate Model\n" + eval_code))

        # --- Export notebook
        nb_bytes = nbformat.writes(nb).encode('utf-8')
        st.download_button(
            label="Download Notebook",
            data=nb_bytes,
            file_name="ml_pipeline.ipynb",
            mime="application/x-ipynb+json"
        )




st.markdown("---")
st.markdown("**Enhanced ML App** | **Created by Ahmed Nmer** | 2025 ©")
