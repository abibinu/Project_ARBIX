# Create a new file called ml_predictor.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class MLPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, df):
        """
        Prepare features for the ML model
        """
        df = df.copy()
        
        # Technical indicator features
        features = []
        
        # 1. EMA-based features
        df['ema_ratio'] = df['EMA_20'] / df['EMA_50']
        df['ema_diff'] = df['EMA_20'] - df['EMA_50']
        df['ema_diff_pct'] = df['ema_diff'] / df['close'] * 100
        
        # 2. RSI features
        df['rsi_diff'] = df['RSI_14'] - df['RSI_14'].shift(1)
        df['rsi_ma'] = df['RSI_14'].rolling(5).mean()
        df['rsi_trend'] = df['RSI_14'] - df['rsi_ma']
        
        # 3. Price action features
        df['price_to_ema_ratio'] = df['close'] / df['EMA_200']
        df['range_pct'] = (df['high'] - df['low']) / df['close'] * 100
        df['body_pct'] = abs(df['close'] - df['open']) / (df['high'] - df['low']) * 100
        
        # 4. Volume features
        df['volume_ema'] = df['volume'].ewm(span=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ema']
        
        # 5. Trend features
        df['up_streak'] = 0
        df['down_streak'] = 0
        
        streak = 0
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                streak = max(0, streak) + 1
                df.loc[i, 'up_streak'] = streak
                df.loc[i, 'down_streak'] = 0
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                streak = min(0, streak) - 1
                df.loc[i, 'down_streak'] = abs(streak)
                df.loc[i, 'up_streak'] = 0
            else:
                df.loc[i, 'up_streak'] = df.loc[i-1, 'up_streak']
                df.loc[i, 'down_streak'] = df.loc[i-1, 'down_streak']
        
        # List of features to use
        feature_columns = [
            'ema_ratio', 'ema_diff_pct', 'rsi_diff', 'rsi_trend',
            'price_to_ema_ratio', 'range_pct', 'body_pct',
            'volume_ratio', 'up_streak', 'down_streak', 'RSI_14'
        ]
        
        return df[feature_columns].copy()
    
    def prepare_labels(self, df, forward_periods=10, threshold=2.0):
        """
        Create labels for supervised learning
        
        Label = 1 if price increases by threshold% within forward_periods
        Label = 0 otherwise
        """
        df = df.copy()
        
        # Calculate forward returns
        df['forward_return'] = df['close'].shift(-forward_periods) / df['close'] - 1
        
        # Create binary labels
        df['label'] = (df['forward_return'] > threshold/100).astype(int)
        
        return df['label']
    
    def train(self, df, test_size=0.3):
        """
        Train the machine learning model
        """
        # Prepare features and labels
        X = self.prepare_features(df)
        y = self.prepare_labels(df)
        
        # Drop rows with NaN
        # Fix warning by aligning indices and ensuring boolean masks are compatible
        y_na = y.isna()
        y_na = y_na.reindex(X.index, fill_value=False)
        valid_idx = ~(X.isna().any(axis=1) | y_na)
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return accuracy
    
    def predict(self, df):
        """
        Make predictions on new data
        """
        if not self.is_trained:
            print("Model not trained yet!")
            return None
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        probabilities = self.model.predict_proba(X_scaled)
        
        # Get probability of positive class (favorable entry)
        buy_probabilities = probabilities[:, 1]
        
        # Return as pandas Series aligned with prepared features index
        return pd.Series(buy_probabilities, index=X.index)
