"""
Data Science Tools - Analysis, visualization, and ML capabilities
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, mean_squared_error, r2_score,
                             confusion_matrix, classification_report,
                             mean_absolute_error)
from .data_cleaner import DataCleaner

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class DataScienceTools:
    """Main class for data science operations"""
    
    def __init__(self):
        self.df = None

    def load_data(self, file_path: str):
        """Load CSV data"""
        self.df = pd.read_csv(file_path)
        return f"✅ Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns."

    def get_dataframe(self):
        """Return the DataFrame"""
        return self.df

    def explore_data(self):
        """Explore dataset - show columns, types, and sample rows"""
        if self.df is None:
            return "⚠️ No data loaded."
        info = f"Columns: {list(self.df.columns)}\n"
        info += f"Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns\n"
        info += f"Data types:\n{self.df.dtypes.to_string()}\n\n"
        info += f"First 5 rows:\n{self.df.head().to_string()}"
        return info
    
    def summarize_data(self):
        """Generate comprehensive data summary"""
        if self.df is None:
            return "⚠️ No data loaded."
        
        summary = "📊 DATA SUMMARY\n" + "="*50 + "\n\n"
        summary += f"Dataset Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns\n\n"
        
        # Column information
        summary += "Columns:\n"
        for col in self.df.columns:
            summary += f"  - {col} ({self.df[col].dtype})\n"
        
        # Missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            summary += f"\n⚠️ Missing Values: {missing.sum()} total\n"
            for col in missing[missing > 0].index:
                summary += f"  - {col}: {missing[col]} ({missing[col]/len(self.df)*100:.1f}%)\n"
        else:
            summary += "\n✅ No missing values\n"
        
        # Numeric summary - Show ALL numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary += f"\n📈 Numeric Columns Summary ({len(numeric_cols)} columns):\n"
            for col in numeric_cols:
                col_stats = self.df[col].describe()
                summary += f"\n{col}:\n"
                summary += f"  Mean: {col_stats['mean']:.6f}\n"
                summary += f"  Std:  {col_stats['std']:.6f}\n"
                summary += f"  Min:  {col_stats['min']:.6f}\n"
                summary += f"  25%:  {col_stats['25%']:.6f}\n"
                summary += f"  50%:  {col_stats['50%']:.6f}\n"
                summary += f"  75%:  {col_stats['75%']:.6f}\n"
                summary += f"  Max:  {col_stats['max']:.6f}\n"
        
        # Categorical summary - Show ALL categorical columns
        cat_cols = self.df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            summary += f"\n\n📋 Categorical Columns ({len(cat_cols)} columns):\n"
            for col in cat_cols:
                summary += f"  - {col}: {self.df[col].nunique()} unique values\n"
        
        # Duplicates
        duplicates = self.df.duplicated().sum()
        summary += f"\n🔄 Duplicate Rows: {duplicates}\n"
        
        return summary
    
    def get_column_info(self, column_name: str):
        """Get detailed information about a specific column"""
        if self.df is None:
            return "⚠️ No data loaded."
        if column_name not in self.df.columns:
            return f"⚠️ Column '{column_name}' not found. Available columns: {list(self.df.columns)}"
        
        col = self.df[column_name]
        info = f"Column: {column_name}\n"
        info += f"Data type: {col.dtype}\n"
        info += f"Non-null count: {col.count()} / {len(col)}\n"
        info += f"Unique values: {col.nunique()}\n"
        
        if col.dtype in ['int64', 'float64']:
            info += f"\nStatistics:\n{col.describe().to_string()}"
        else:
            info += f"\nTop 5 values:\n{col.value_counts().head().to_string()}"
        
        return info
    
    def analyze_correlations(self):
        """Analyze correlations between numeric columns"""
        if self.df is None:
            return "⚠️ No data loaded."
        
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            return "⚠️ Need at least 2 numeric columns for correlation analysis."
        
        # Calculate correlation matrix
        corr_matrix = self.df[numeric_cols].corr()
        
        analysis = "📊 CORRELATION ANALYSIS\n" + "="*60 + "\n\n"
        analysis += f"Analyzing {len(numeric_cols)} numeric columns: {', '.join(numeric_cols)}\n\n"
        
        # Find strong correlations
        analysis += "🔥 STRONG CORRELATIONS (|r| > 0.7):\n"
        strong_corrs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    strong_corrs.append((col1, col2, corr_val))
                    direction = "positive" if corr_val > 0 else "negative"
                    analysis += f"  • {col1} ↔ {col2}: {corr_val:.3f} ({direction})\n"
        
        if not strong_corrs:
            analysis += "  None found\n"
        
        # Moderate correlations
        analysis += "\n📈 MODERATE CORRELATIONS (0.5 < |r| < 0.7):\n"
        moderate_corrs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if 0.5 < abs(corr_val) <= 0.7:
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    moderate_corrs.append((col1, col2, corr_val))
                    direction = "positive" if corr_val > 0 else "negative"
                    analysis += f"  • {col1} ↔ {col2}: {corr_val:.3f} ({direction})\n"
        
        if not moderate_corrs:
            analysis += "  None found\n"
        
        # Full correlation matrix
        analysis += "\n📋 FULL CORRELATION MATRIX:\n"
        analysis += corr_matrix.to_string()
        
        # Interpretation guide
        analysis += "\n\n💡 INTERPRETATION GUIDE:\n"
        analysis += "  • |r| > 0.7: Strong relationship\n"
        analysis += "  • 0.5 < |r| < 0.7: Moderate relationship\n"
        analysis += "  • 0.3 < |r| < 0.5: Weak relationship\n"
        analysis += "  • |r| < 0.3: Very weak/no relationship\n"
        analysis += "  • Positive r: Variables increase together\n"
        analysis += "  • Negative r: One increases as other decreases\n"
        
        return analysis

    def clean_data(self, level='standard'):
        """Basic auto-clean using DataCleaner"""
        if self.df is None:
            return "⚠️ No data loaded."
        
        cleaner = DataCleaner(self.df)
        self.df = cleaner.auto_clean(level=level)
        report = cleaner.get_cleaning_report()
        
        return report
    
    def get_advanced_cleaner(self):
        """Get DataCleaner instance for advanced cleaning"""
        if self.df is None:
            return None
        return DataCleaner(self.df)

    def suggest_model(self, target: str):
        """Suggest appropriate ML model for target column"""
        if self.df is None:
            return "⚠️ No data loaded."
        if target not in self.df.columns:
            return f"⚠️ Target '{target}' not found."

        df_clean = self.df.copy()

        # Encode categorical columns
        for col in df_clean.select_dtypes(include=["object"]).columns:
            df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))

        X = df_clean.drop(columns=[target])
        y = df_clean[target]

        # Determine task type
        if y.nunique() <= 10:
            model = RandomForestClassifier()
            task = "classification"
        else:
            model = RandomForestRegressor()
            task = "regression"

        # Split and fit
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if task == "classification":
            score = accuracy_score(y_test, preds)
            return f"🤖 Suggested Model: RandomForestClassifier (Accuracy: {score:.2f})"
        else:
            mse = mean_squared_error(y_test, preds)
            return f"🤖 Suggested Model: RandomForestRegressor (MSE: {mse:.2f})"
    
    def generate_plot(self, query: str):
        """Generate visualization from natural language query"""
        if self.df is None:
            return "⚠️ No data loaded. Please upload a dataset first."
        
        query_lower = query.lower()
        
        try:
            plt.clf()
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extract column names from query
            columns = [col for col in self.df.columns if col.lower() in query_lower]
            
            if not columns:
                return f"⚠️ Could not identify columns from query. Available columns: {list(self.df.columns)}"
            
            # Scatter plot
            if 'scatter' in query_lower and len(columns) >= 2:
                x_col, y_col = columns[0], columns[1]
                ax.scatter(self.df[x_col], self.df[y_col], alpha=0.6)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
                ax.grid(True, alpha=0.3)
                
            # Box plot
            elif 'box' in query_lower:
                if len(columns) == 1:
                    ax.boxplot(self.df[columns[0]].dropna())
                    ax.set_ylabel(columns[0])
                    ax.set_title(f'Box Plot: {columns[0]}')
                else:
                    self.df[columns].boxplot(ax=ax)
                    ax.set_title(f'Box Plot: {", ".join(columns)}')
                    
            # Histogram
            elif 'hist' in query_lower or 'distribution' in query_lower:
                for col in columns:
                    ax.hist(self.df[col].dropna(), alpha=0.6, label=col, bins=30)
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Histogram: {", ".join(columns)}')
                ax.legend()
                
            # Line plot
            elif 'line' in query_lower and len(columns) >= 2:
                x_col, y_col = columns[0], columns[1]
                ax.plot(self.df[x_col], self.df[y_col])
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f'Line Plot: {x_col} vs {y_col}')
                ax.grid(True, alpha=0.3)
                
            # Bar plot
            elif 'bar' in query_lower:
                if len(columns) == 1:
                    value_counts = self.df[columns[0]].value_counts().head(10)
                    ax.bar(range(len(value_counts)), value_counts.values)
                    ax.set_xticks(range(len(value_counts)))
                    ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                    ax.set_ylabel('Count')
                    ax.set_title(f'Bar Plot: {columns[0]}')
                else:
                    self.df[columns].plot(kind='bar', ax=ax)
                    ax.set_title(f'Bar Plot: {", ".join(columns)}')
                    
            # Correlation heatmap
            elif 'correlation' in query_lower or 'heatmap' in query_lower:
                if len(columns) >= 2:
                    corr = self.df[columns].corr()
                    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
                    ax.set_title('Correlation Heatmap')
                else:
                    numeric_cols = self.df.select_dtypes(include=['number']).columns
                    corr = self.df[numeric_cols].corr()
                    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
                    ax.set_title('Correlation Heatmap')
                    
            else:
                # Default to scatter plot if 2 columns, histogram if 1
                if len(columns) >= 2:
                    x_col, y_col = columns[0], columns[1]
                    ax.scatter(self.df[x_col], self.df[y_col], alpha=0.6)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f'{x_col} vs {y_col}')
                elif len(columns) == 1:
                    ax.hist(self.df[columns[0]].dropna(), bins=30)
                    ax.set_xlabel(columns[0])
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'Distribution of {columns[0]}')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            return f"❌ Error generating plot: {str(e)}"
    
    def train_models(self, df, target_column, model_types, test_size=0.2):
        """Train and compare multiple ML models"""
        results = {}
        
        # Prepare data
        df_clean = df.copy()
        
        # Encode categorical columns
        for col in df_clean.select_dtypes(include=["object"]).columns:
            if col != target_column:
                df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))
        
        # Separate features and target
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]
        
        # Encode target if categorical
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y.astype(str))
        
        # Determine task type
        if y.nunique() <= 10:
            task_type = "classification"
        else:
            task_type = "regression"
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train each model
        for model_name in model_types:
            start_time = time.time()
            
            if task_type == "classification":
                if model_name == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_name == "XGBoost":
                    if XGBOOST_AVAILABLE:
                        model = XGBClassifier(n_estimators=100, random_state=42)
                    else:
                        continue
                elif model_name == "Logistic Regression":
                    model = LogisticRegression(random_state=42, max_iter=1000)
                else:
                    continue
                
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = accuracy_score(y_test, preds)
                
                # Create confusion matrix plot
                fig, ax = plt.subplots(figsize=(8, 6))
                cm = confusion_matrix(y_test, preds)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f'Confusion Matrix - {model_name}')
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                
                results[model_name] = {
                    'score': score,
                    'task_type': task_type,
                    'metric': 'Accuracy',
                    'training_time': time.time() - start_time,
                    'plot': fig
                }
                
            else:  # regression
                if model_name == "Random Forest":
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif model_name == "XGBoost":
                    if XGBOOST_AVAILABLE:
                        model = XGBRegressor(n_estimators=100, random_state=42)
                    else:
                        continue
                elif model_name == "Linear Regression":
                    model = LinearRegression()
                else:
                    continue
                
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = r2_score(y_test, preds)
                mse = mean_squared_error(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                
                # Create prediction plot
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test, preds, alpha=0.6)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax.set_xlabel('True Values')
                ax.set_ylabel('Predictions')
                ax.set_title(f'Predictions vs Actual - {model_name}')
                ax.grid(True, alpha=0.3)
                
                results[model_name] = {
                    'score': score,
                    'mse': mse,
                    'mae': mae,
                    'task_type': task_type,
                    'metric': 'R²',
                    'training_time': time.time() - start_time,
                    'plot': fig
                }
        
        return results

