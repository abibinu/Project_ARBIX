# Create a new file called ml_predictor.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight
import config  # Use base config instead of live_config
import indicators  # Import at the top level

class MLPredictor:
    def __init__(self):
        # Use balanced subsample to handle class imbalance
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced_subsample',
            min_samples_leaf=5  # Ensure minimum samples per leaf
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, df):
        """
        Prepare features for the ML model
        """
        df = df.copy()
        
        # Add technical indicators first
        df = indicators.add_indicators(df)
        
        # Technical indicator features
        features = []
        
        # 1. EMA-based features
        df['ema_ratio'] = df[config.COL_EMA_SHORT] / df[config.COL_EMA_LONG]
        df['ema_diff'] = df[config.COL_EMA_SHORT] - df[config.COL_EMA_LONG]
        df['ema_diff_pct'] = df['ema_diff'] / df['close'] * 100
        
        # 2. RSI features
        df['rsi_diff'] = df[config.COL_RSI] - df[config.COL_RSI].shift(1)
        df['rsi_ma'] = df[config.COL_RSI].rolling(5).mean()
        df['rsi_trend'] = df[config.COL_RSI] - df['rsi_ma']
        
        # 3. Price action features
        df['price_to_ema_ratio'] = df['close'] / df[config.COL_EMA_LONGTERM]
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
                df.iloc[i, df.columns.get_loc('up_streak')] = streak
                df.iloc[i, df.columns.get_loc('down_streak')] = 0
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                streak = min(0, streak) - 1
                df.iloc[i, df.columns.get_loc('down_streak')] = abs(streak)
                df.iloc[i, df.columns.get_loc('up_streak')] = 0
            else:
                df.iloc[i, df.columns.get_loc('up_streak')] = df.iloc[i-1, df.columns.get_loc('up_streak')]
                df.iloc[i, df.columns.get_loc('down_streak')] = df.iloc[i-1, df.columns.get_loc('down_streak')]
        
        # List of features to use
        feature_columns = [
            'ema_ratio', 'ema_diff_pct', 'rsi_diff', 'rsi_trend',
            'price_to_ema_ratio', 'range_pct', 'body_pct',
            'volume_ratio', 'up_streak', 'down_streak', config.COL_RSI
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
        """Train the machine learning model with improved class handling"""
        df = df.copy()
        
        # Check if we have enough data
        if len(df) < config.ML_TRAINING_WINDOW:
            print(f"Warning: Insufficient data for ML training. Got {len(df)} candles, need {config.ML_TRAINING_WINDOW}")
            print("Will proceed with available data but predictions may be less reliable")
        
        # Prepare features and labels
        X = self.prepare_features(df)
        y = self.prepare_labels(df)
        
        # Ensure X and y have the same index
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        # Drop rows with NaN
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Check minimum required samples
        if len(X) < 100:
            print("Warning: Not enough valid data points for training")
            return 0.0
            
        # Check class balance
        class_counts = np.bincount(y)
        if len(class_counts) > 1:
            minority_pct = (class_counts.min() / len(y)) * 100
            if minority_pct < 10:
                print(f"Warning: Severe class imbalance detected. Minority class is only {minority_pct:.1f}%")
                print("Consider adjusting label thresholds or collecting more data")
            
        # Calculate class weights
        weights = dict(zip(
            np.unique(y),
            class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        ))
        
        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model with class weights
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate with zero_division parameter set
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Print class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("\nClass Distribution:")
        for label, count in zip(unique, counts):
            print(f"Class {label}: {count} samples ({count/len(y)*100:.1f}%)")
        
        self.is_trained = True
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        # Return model metrics
        metrics = {
            'accuracy': accuracy,
            'data_points': len(X),
            'class_balance': minority_pct if len(class_counts) > 1 else 50.0
        }
        
        return metrics
    
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
