"""
Data Cleaner - Advanced data cleaning with multiple strategies
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """Advanced data cleaning with multiple strategies and detailed reporting"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
        self.cleaning_log = []
        self.encoders = {}
        self.scalers = {}
        
    def get_cleaning_report(self):
        """Generate comprehensive cleaning report"""
        report = "🧹 DATA CLEANING REPORT\n" + "="*70 + "\n\n"
        
        if not self.cleaning_log:
            report += "No cleaning operations performed yet.\n"
            return report
        
        report += f"Operations Performed: {len(self.cleaning_log)}\n\n"
        
        for i, log_entry in enumerate(self.cleaning_log, 1):
            report += f"{i}. {log_entry}\n"
        
        # Compare before/after
        report += "\n" + "="*70 + "\n"
        report += "📊 BEFORE vs AFTER:\n\n"
        report += f"Shape: {self.original_df.shape} → {self.df.shape}\n"
        report += f"Missing Values: {self.original_df.isnull().sum().sum()} → {self.df.isnull().sum().sum()}\n"
        report += f"Duplicates: {self.original_df.duplicated().sum()} → {self.df.duplicated().sum()}\n"
        report += f"Memory: {self.original_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB → "
        report += f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n"
        
        return report
    
    def handle_missing_values(self, strategy='auto', numeric_method='mean', categorical_method='mode'):
        """Handle missing values with multiple strategies"""
        if self.df.isnull().sum().sum() == 0:
            self.cleaning_log.append("✅ No missing values found")
            return self.df
        
        missing_before = self.df.isnull().sum().sum()
        
        if strategy == 'drop':
            self.df = self.df.dropna()
            self.cleaning_log.append(f"🗑️ Dropped rows with missing values ({missing_before} values removed)")
            
        elif strategy == 'auto' or strategy == 'impute':
            # Numeric columns
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if self.df[col].isnull().any():
                    if numeric_method == 'mean':
                        value = self.df[col].mean()
                        self.df[col].fillna(value, inplace=True)
                        self.cleaning_log.append(f"🔢 Filled '{col}' missing with mean ({value:.2f})")
                    elif numeric_method == 'median':
                        value = self.df[col].median()
                        self.df[col].fillna(value, inplace=True)
                        self.cleaning_log.append(f"🔢 Filled '{col}' missing with median ({value:.2f})")
                    elif numeric_method == 'mode':
                        value = self.df[col].mode()[0] if not self.df[col].mode().empty else 0
                        self.df[col].fillna(value, inplace=True)
                        self.cleaning_log.append(f"🔢 Filled '{col}' missing with mode ({value:.2f})")
                    elif numeric_method == 'knn':
                        imputer = KNNImputer(n_neighbors=5)
                        self.df[col] = imputer.fit_transform(self.df[[col]])
                        self.cleaning_log.append(f"🔢 Filled '{col}' missing with KNN imputation")
            
            # Categorical columns
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.df[col].isnull().any():
                    if categorical_method == 'mode':
                        value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                        self.df[col].fillna(value, inplace=True)
                        self.cleaning_log.append(f"📝 Filled '{col}' missing with mode ('{value}')")
                    elif categorical_method == 'constant':
                        self.df[col].fillna('Unknown', inplace=True)
                        self.cleaning_log.append(f"📝 Filled '{col}' missing with 'Unknown'")
                    elif categorical_method == 'missing':
                        self.df[col].fillna('Missing', inplace=True)
                        self.cleaning_log.append(f"📝 Filled '{col}' missing with 'Missing'")
        
        elif strategy == 'fill_zero':
            self.df.fillna(0, inplace=True)
            self.cleaning_log.append(f"0️⃣ Filled all missing values with 0")
            
        elif strategy == 'fill_forward':
            self.df.fillna(method='ffill', inplace=True)
            self.cleaning_log.append(f"➡️ Forward filled missing values")
            
        elif strategy == 'fill_backward':
            self.df.fillna(method='bfill', inplace=True)
            self.cleaning_log.append(f"⬅️ Backward filled missing values")
        
        return self.df
    
    def remove_duplicates(self, subset=None, keep='first'):
        """Remove duplicate rows"""
        duplicates_before = self.df.duplicated(subset=subset).sum()
        
        if duplicates_before == 0:
            self.cleaning_log.append("✅ No duplicate rows found")
            return self.df
        
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        self.cleaning_log.append(f"🗑️ Removed {duplicates_before} duplicate rows (kept={keep})")
        
        return self.df
    
    def handle_outliers(self, method='iqr', columns=None, threshold=1.5):
        """Detect and handle outliers"""
        if columns is None:
            columns = self.df.select_dtypes(include=['number']).columns.tolist()
        
        outliers_removed = 0
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                if outliers > 0:
                    self.df[col] = self.df[col].clip(lower_bound, upper_bound)
                    outliers_removed += outliers
                    self.cleaning_log.append(f"📊 Clipped {outliers} outliers in '{col}' using IQR method")
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outliers = (z_scores > threshold).sum()
                if outliers > 0:
                    mask = z_scores <= threshold
                    self.df = self.df[mask]
                    outliers_removed += outliers
                    self.cleaning_log.append(f"📊 Removed {outliers} outliers in '{col}' using Z-score (threshold={threshold})")
        
        if outliers_removed == 0:
            self.cleaning_log.append(f"✅ No outliers detected with {method} method")
        
        return self.df
    
    def encode_categorical(self, method='label', columns=None):
        """Encode categorical variables"""
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns.tolist()
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            if method == 'label':
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le
                self.cleaning_log.append(f"🔤 Label encoded '{col}' ({len(le.classes_)} categories)")
            
            elif method == 'onehot':
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df = self.df.drop(columns=[col])
                self.cleaning_log.append(f"🔤 One-hot encoded '{col}' (created {len(dummies.columns)} new columns)")
        
        return self.df
    
    def scale_features(self, method='standard', columns=None):
        """Scale numerical features"""
        if columns is None:
            columns = self.df.select_dtypes(include=['number']).columns.tolist()
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            
            self.df[col] = scaler.fit_transform(self.df[[col]])
            self.scalers[col] = scaler
            self.cleaning_log.append(f"📏 Scaled '{col}' using {method} scaling")
        
        return self.df
    
    def optimize_dtypes(self):
        """Optimize data types to reduce memory usage"""
        memory_before = self.df.memory_usage(deep=True).sum() / 1024**2
        
        # Optimize integers
        for col in self.df.select_dtypes(include=['int64']).columns:
            col_min = self.df[col].min()
            col_max = self.df[col].max()
            
            if col_min >= 0:
                if col_max < 255:
                    self.df[col] = self.df[col].astype('uint8')
                elif col_max < 65535:
                    self.df[col] = self.df[col].astype('uint16')
                elif col_max < 4294967295:
                    self.df[col] = self.df[col].astype('uint32')
            else:
                if col_min > -128 and col_max < 127:
                    self.df[col] = self.df[col].astype('int8')
                elif col_min > -32768 and col_max < 32767:
                    self.df[col] = self.df[col].astype('int16')
                elif col_min > -2147483648 and col_max < 2147483647:
                    self.df[col] = self.df[col].astype('int32')
        
        # Optimize floats
        for col in self.df.select_dtypes(include=['float64']).columns:
            self.df[col] = self.df[col].astype('float32')
        
        memory_after = self.df.memory_usage(deep=True).sum() / 1024**2
        memory_saved = memory_before - memory_after
        
        if memory_saved > 0:
            self.cleaning_log.append(f"💾 Optimized dtypes: Saved {memory_saved:.2f} MB ({memory_saved/memory_before*100:.1f}%)")
        
        return self.df
    
    def remove_constant_columns(self):
        """Remove columns with constant values"""
        constant_cols = [col for col in self.df.columns if self.df[col].nunique() == 1]
        
        if constant_cols:
            self.df = self.df.drop(columns=constant_cols)
            self.cleaning_log.append(f"🗑️ Removed {len(constant_cols)} constant columns: {constant_cols}")
        else:
            self.cleaning_log.append("✅ No constant columns found")
        
        return self.df
    
    def remove_high_cardinality_columns(self, threshold=0.95):
        """Remove columns with too many unique values (potential IDs)"""
        high_card_cols = []
        
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                uniqueness = self.df[col].nunique() / len(self.df)
                if uniqueness > threshold:
                    high_card_cols.append(col)
        
        if high_card_cols:
            self.df = self.df.drop(columns=high_card_cols)
            self.cleaning_log.append(f"🗑️ Removed {len(high_card_cols)} high cardinality columns (>{threshold*100}% unique): {high_card_cols}")
        else:
            self.cleaning_log.append(f"✅ No high cardinality columns found (threshold={threshold*100}%)")
        
        return self.df
    
    def auto_clean(self, level='standard'):
        """Automatic data cleaning with preset configurations"""
        self.cleaning_log.append(f"🤖 Starting auto-clean with level='{level}'")
        
        if level in ['light', 'standard', 'aggressive']:
            # Always do these
            self.remove_duplicates()
            self.handle_missing_values(strategy='auto')
        
        if level in ['standard', 'aggressive']:
            self.handle_outliers(method='iqr', threshold=1.5)
            self.encode_categorical(method='label')
            self.optimize_dtypes()
        
        if level == 'aggressive':
            self.remove_constant_columns()
            self.remove_high_cardinality_columns(threshold=0.95)
        
        self.cleaning_log.append(f"✅ Auto-clean completed ({level} level)")
        
        return self.df
    
    def get_cleaned_dataframe(self):
        """Return the cleaned DataFrame"""
        return self.df
    
    def reset(self):
        """Reset to original DataFrame"""
        self.df = self.original_df.copy()
        self.cleaning_log = []
        self.encoders = {}
        self.scalers = {}
        self.cleaning_log.append("🔄 Reset to original data")
        return self.df

