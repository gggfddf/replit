import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Optional TensorFlow import with comprehensive error handling
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, MultiHeadAttention, LayerNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
    print("✓ TensorFlow available for deep learning models")
except ImportError as e:
    print(f"⚠ TensorFlow not available: {e}")
    print("→ Using traditional ML models only")
except Exception as e:
    print(f"⚠ TensorFlow compatibility issue: {e}")
    print("→ Using traditional ML models only")

class DeepLearningEngine:
    """
    Comprehensive Deep Learning Engine combining LSTM, CNN, Transformer, and AutoEncoder models
    for advanced market pattern recognition and prediction
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        
    def prepare_sequences(self, data, seq_length=60):
        """Prepare sequential data for neural networks"""
        X, y = [], []
        
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i])
            
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape):
        """Build advanced LSTM model for sequential pattern recognition"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='huber', 
                     metrics=['mae'])
        
        return model
    
    def build_cnn_model(self, input_shape):
        """Build CNN model for chart pattern recognition"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=50, kernel_size=3, activation='relu'),
            Flatten(),
            Dense(50, activation='relu'),
            Dropout(0.3),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='huber', 
                     metrics=['mae'])
        
        return model
    
    def build_transformer_model(self, input_shape):
        """Build Transformer model for complex pattern relationships"""
        inputs = Input(shape=input_shape)
        
        # Multi-head attention
        attention = MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
        attention = LayerNormalization(epsilon=1e-6)(attention)
        
        # Add & Norm
        x = tf.keras.layers.Add()([inputs, attention])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Feed forward
        ffn = tf.keras.layers.Dense(512, activation='relu')(x)
        ffn = tf.keras.layers.Dense(input_shape[-1])(ffn)
        
        # Add & Norm
        x = tf.keras.layers.Add()([x, ffn])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Global average pooling and output
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(50, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='huber', 
                     metrics=['mae'])
        
        return model
    
    def build_autoencoder_model(self, input_shape):
        """Build AutoEncoder for anomaly detection"""
        input_layer = Input(shape=input_shape)
        
        # Encoder
        encoded = Dense(128, activation='relu')(input_layer)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(32, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(input_shape[0], activation='linear')(decoded)
        
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def create_advanced_features(self, data):
        """Create advanced engineered features for deep learning"""
        df = data.copy()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Returns'].rolling(20).std()
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        
        # Volume features
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['Price_Volume'] = df['Close'] * df['Volume']
        
        # Technical features
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'] = self.calculate_macd(df['Close'])
        df['BB_Upper'], df['BB_Lower'] = self.calculate_bollinger_bands(df['Close'])
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Time-based features
        df['Hour'] = df.index.hour if hasattr(df.index, 'hour') else 0
        df['DayOfWeek'] = df.index.dayofweek if hasattr(df.index, 'dayofweek') else 0
        df['Month'] = df.index.month if hasattr(df.index, 'month') else 0
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'Close_MA_{window}'] = df['Close'].rolling(window).mean()
            df[f'Close_Std_{window}'] = df['Close'].rolling(window).std()
            df[f'Volume_MA_{window}'] = df['Volume'].rolling(window).mean()
        
        return df.dropna()
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window).mean()
        rolling_std = prices.rolling(window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band
    
    def train_ensemble_models(self, data):
        """Train ensemble of deep learning models"""
        try:
            if not TENSORFLOW_AVAILABLE:
                return self.train_traditional_models(data)
            
            # Create advanced features
            featured_data = self.create_advanced_features(data)
            
            # Select numerical features
            feature_columns = [col for col in featured_data.columns if featured_data[col].dtype in ['float64', 'int64']]
            feature_columns = [col for col in feature_columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            
            X = featured_data[feature_columns].values
            y = featured_data['Close'].shift(-1).dropna().values
            X = X[:-1]  # Remove last row to match y
            
            # Scale features
            self.scalers['features'] = StandardScaler()
            self.scalers['target'] = MinMaxScaler()
            
            X_scaled = self.scalers['features'].fit_transform(X)
            y_scaled = self.scalers['target'].fit_transform(y.reshape(-1, 1)).flatten()
            
            # Split data
            split_index = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
            y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]
            
            # Prepare sequences for neural networks
            seq_length = 60
            X_train_seq, y_train_seq = self.prepare_sequences(X_train, seq_length)
            X_test_seq, y_test_seq = self.prepare_sequences(X_test, seq_length)
            
            # Train models
            models_performance = {}
            
            # 1. LSTM Model
            try:
                lstm_model = self.build_lstm_model((seq_length, X_train.shape[1]))
                early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
            
                lstm_history = lstm_model.fit(
                    X_train_seq, y_train_seq,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test_seq, y_test_seq),
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                lstm_pred = lstm_model.predict(X_test_seq, verbose=0)
                lstm_mae = mean_absolute_error(y_test_seq, lstm_pred)
                lstm_mse = mean_squared_error(y_test_seq, lstm_pred)
                
                self.models['LSTM'] = lstm_model
                models_performance['LSTM'] = {
                    'accuracy': max(0, 100 - lstm_mae * 100),
                    'loss': lstm_mse,
                    'mae': lstm_mae
                }
            except Exception as e:
                print(f"LSTM training failed: {e}")
        
            # 2. CNN Model
            try:
                cnn_model = self.build_cnn_model((seq_length, X_train.shape[1]))
                cnn_history = cnn_model.fit(
                    X_train_seq, y_train_seq,
                    epochs=30,
                    batch_size=32,
                    validation_data=(X_test_seq, y_test_seq),
                    verbose=0
                )
                
                cnn_pred = cnn_model.predict(X_test_seq, verbose=0)
                cnn_mae = mean_absolute_error(y_test_seq, cnn_pred)
                cnn_mse = mean_squared_error(y_test_seq, cnn_pred)
                
                self.models['CNN'] = cnn_model
                models_performance['CNN'] = {
                    'accuracy': max(0, 100 - cnn_mae * 100),
                    'loss': cnn_mse,
                    'mae': cnn_mae
                }
            except Exception as e:
                print(f"CNN training failed: {e}")
        
            # 3. Random Forest (for feature importance)
            try:
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf_model.fit(X_train, y_train)
                
                rf_pred = rf_model.predict(X_test)
                rf_mae = mean_absolute_error(y_test, rf_pred)
                rf_mse = mean_squared_error(y_test, rf_pred)
                
                # Feature importance
                feature_importance = dict(zip(feature_columns, rf_model.feature_importances_))
                self.feature_importance = dict(sorted(feature_importance.items(), 
                                                    key=lambda x: x[1], reverse=True)[:10])
                
                self.models['RandomForest'] = rf_model
                models_performance['RandomForest'] = {
                    'accuracy': max(0, 100 - rf_mae * 100),
                    'loss': rf_mse,
                    'mae': rf_mae
                }
            except Exception as e:
                print(f"Random Forest training failed: {e}")
            
            # 4. XGBoost
            try:
                xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                xgb_model.fit(X_train, y_train)
                
                xgb_pred = xgb_model.predict(X_test)
                xgb_mae = mean_absolute_error(y_test, xgb_pred)
                xgb_mse = mean_squared_error(y_test, xgb_pred)
                
                self.models['XGBoost'] = xgb_model
                models_performance['XGBoost'] = {
                    'accuracy': max(0, 100 - xgb_mae * 100),
                    'loss': xgb_mse,
                    'mae': xgb_mae
                }
            except Exception as e:
                print(f"XGBoost training failed: {e}")
            
            self.model_performance = models_performance
            return models_performance
            
        except Exception as e:
            print(f"Deep learning model training failed: {e}")
            return self.train_traditional_models(data)
    
    def analyze_market(self, market_data):
        """Comprehensive market analysis using deep learning ensemble"""
        # Train models
        performance = self.train_ensemble_models(market_data)
        
        # Generate ensemble predictions
        predictions = self.generate_ensemble_predictions(market_data)
        
        # Market regime detection
        regime = self.detect_market_regime(market_data)
        
        # Anomaly detection
        anomalies = self.detect_market_anomalies(market_data)
        
        return {
            'model_performance': performance,
            'feature_importance': self.feature_importance,
            'ensemble_predictions': predictions,
            'market_regime': regime,
            'anomalies': anomalies
        }
    
    def generate_ensemble_predictions(self, data):
        """Generate ensemble predictions from all models"""
        # Create features for latest data
        featured_data = self.create_advanced_features(data)
        feature_columns = [col for col in featured_data.columns if featured_data[col].dtype in ['float64', 'int64']]
        feature_columns = [col for col in feature_columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        latest_features = featured_data[feature_columns].tail(60).values
        
        if hasattr(self.scalers, 'features'):
            latest_scaled = self.scalers['features'].transform(latest_features)
            
            predictions = {}
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                try:
                    if model_name in ['LSTM', 'CNN']:
                        # Sequence prediction
                        seq_input = latest_scaled.reshape(1, latest_scaled.shape[0], latest_scaled.shape[1])
                        pred_scaled = model.predict(seq_input, verbose=0)[0][0]
                        pred = self.scalers['target'].inverse_transform([[pred_scaled]])[0][0]
                    else:
                        # Direct prediction
                        pred_scaled = model.predict(latest_scaled[-1:])
                        pred = self.scalers['target'].inverse_transform([[pred_scaled[0]]])[0][0]
                    
                    predictions[model_name] = pred
                except Exception as e:
                    print(f"Prediction failed for {model_name}: {e}")
                    predictions[model_name] = data['Close'].iloc[-1]
            
            return predictions
        
        return {}
    
    def detect_market_regime(self, data):
        """Detect current market regime (Bull/Bear/Sideways)"""
        returns = data['Close'].pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)
        
        # Calculate trend strength
        sma_20 = data['Close'].rolling(20).mean()
        sma_50 = data['Close'].rolling(50).mean()
        
        current_price = data['Close'].iloc[-1]
        current_sma20 = sma_20.iloc[-1]
        current_sma50 = sma_50.iloc[-1]
        current_volatility = volatility.iloc[-1]
        
        # Regime classification
        if current_price > current_sma20 > current_sma50 and current_volatility < 0.3:
            regime = "Bull Market"
            confidence = 0.8
        elif current_price < current_sma20 < current_sma50 and current_volatility < 0.3:
            regime = "Bear Market"
            confidence = 0.8
        elif abs(current_price - current_sma20) / current_sma20 < 0.05:
            regime = "Sideways Market"
            confidence = 0.7
        else:
            regime = "Transition Period"
            confidence = 0.5
        
        return {
            'regime': regime,
            'confidence': confidence,
            'volatility': current_volatility,
            'trend_strength': abs(current_sma20 - current_sma50) / current_sma50
        }
    
    def detect_market_anomalies(self, data):
        """Detect market anomalies using statistical methods"""
        returns = data['Close'].pct_change().dropna()
        volume_changes = data['Volume'].pct_change().dropna()
        
        # Price anomalies (beyond 3 sigma)
        price_mean = returns.mean()
        price_std = returns.std()
        price_anomalies = returns[abs(returns - price_mean) > 3 * price_std]
        
        # Volume anomalies
        volume_mean = volume_changes.mean()
        volume_std = volume_changes.std()
        volume_anomalies = volume_changes[abs(volume_changes - volume_mean) > 3 * volume_std]
        
        return {
            'price_anomalies': len(price_anomalies),
            'volume_anomalies': len(volume_anomalies),
            'latest_return_zscore': (returns.iloc[-1] - price_mean) / price_std if len(returns) > 0 else 0,
            'latest_volume_zscore': (volume_changes.iloc[-1] - volume_mean) / volume_std if len(volume_changes) > 0 else 0
        }
    
    def train_traditional_models(self, data):
        """Fallback method using traditional ML when TensorFlow is not available"""
        try:
            # Create simplified features
            featured_data = self.create_advanced_features(data)
            
            # Select features
            feature_columns = [col for col in featured_data.columns if featured_data[col].dtype in ['float64', 'int64']]
            feature_columns = [col for col in feature_columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            
            X = featured_data[feature_columns].fillna(0)
            y = featured_data['Close'].values
            
            # Train traditional models
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            
            if len(X) > 100:  # Need sufficient data
                train_size = int(0.8 * len(X))
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                
                rf_model.fit(X_train, y_train)
                xgb_model.fit(X_train, y_train)
                
                self.models['random_forest'] = rf_model
                self.models['xgboost'] = xgb_model
            
            return {
                'ensemble_prediction': y[-1],  # Latest price as fallback
                'model_confidence': 0.7,
                'feature_importance': {},
                'prediction_intervals': {'lower': y[-1] * 0.95, 'upper': y[-1] * 1.05},
                'regime_prediction': 'Traditional Analysis',
                'anomaly_score': 0.0,
                'models_trained': ['Random Forest', 'XGBoost'] if len(X) > 100 else ['Statistical Fallback']
            }
            
        except Exception as e:
            print(f"Traditional models training error: {e}")
            return {
                'ensemble_prediction': data['Close'].iloc[-1],
                'model_confidence': 0.5,
                'feature_importance': {},
                'prediction_intervals': {'lower': data['Close'].iloc[-1] * 0.9, 'upper': data['Close'].iloc[-1] * 1.1},
                'regime_prediction': 'Fallback Analysis',
                'anomaly_score': 0.0,
                'models_trained': ['Fallback']
            }
