import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

class DataScienceTools:
    def __init__(self):
        self.df = None

    def load_data(self, file_path: str):
        self.df = pd.read_csv(file_path)
        return f"✅ Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns."

    def get_dataframe(self):
        return self.df

    def explore_data(self):
        if self.df is None:
            return "⚠️ No data loaded."
        return f"Columns: {list(self.df.columns)}\n\nHead:\n{self.df.head().to_string()}"

    def clean_data(self):
        if self.df is None:
            return "⚠️ No data loaded."
        df_clean = self.df.copy()
        # Fill numeric missing values
        df_clean.fillna(df_clean.mean(numeric_only=True), inplace=True)
        # Encode categorical columns
        for col in df_clean.select_dtypes(include=["object"]).columns:
            df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))
        self.df = df_clean
        return "✅ Data cleaned (missing filled, categorical encoded)."

    def suggest_model(self, target: str):
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
