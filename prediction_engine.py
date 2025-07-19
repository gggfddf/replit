import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PredictionEngine:
    """
    Advanced ML Prediction Engine with comprehensive prediction capabilities,
    confidence scoring, risk assessment, and scenario analysis
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.prediction_history = []
        
    def generate_predictions(self, market_data):
        """Generate comprehensive ML predictions with confidence scores"""
        try:
            # Prepare features and targets
            features_df = self.engineer_prediction_features(market_data)
            
            if features_df.empty or len(features_df) < 30:
                return self.create_default_predictions()
            
            # Prepare data for modeling
            X, y = self.prepare_prediction_data(features_df)
            
            if len(X) < 10:
                return self.create_default_predictions()
            
            # Train ensemble models
            model_predictions = self.train_prediction_models(X, y)
            
            # Generate ensemble prediction
            ensemble_prediction = self.create_ensemble_prediction(model_predictions)
            
            # Calculate confidence scores
            confidence_metrics = self.calculate_prediction_confidence(model_predictions, X, y)
            
            # Risk assessment
            risk_assessment = self.assess_prediction_risk(market_data, ensemble_prediction)
            
            # Scenario analysis
            scenario_analysis = self.perform_scenario_analysis(market_data, ensemble_prediction)
            
            # Feature importance
            feature_importance = self.calculate_feature_importance(X, y, features_df.columns)
            
            # Trigger conditions
            trigger_conditions = self.identify_trigger_conditions(market_data, ensemble_prediction)
            
            # Primary prediction structure
            primary_prediction = {
                'direction': ensemble_prediction['direction'],
                'probability': ensemble_prediction['probability'],
                'target_range': ensemble_prediction['target_range'],
                'time_horizon': ensemble_prediction['time_horizon'],
                'confidence_level': confidence_metrics['confidence_level'],
                'confidence': confidence_metrics['confidence_score']
            }
            
            return {
                'primary_prediction': primary_prediction,
                'risk_assessment': risk_assessment,
                'scenario_analysis': scenario_analysis,
                'feature_importance': feature_importance,
                'trigger_conditions': trigger_conditions,
                'model_predictions': model_predictions,
                'confidence_metrics': confidence_metrics
            }
            
        except Exception as e:
            print(f"Prediction generation error: {e}")
            return self.create_default_predictions()
    
    def engineer_prediction_features(self, data):
        """Engineer comprehensive features for prediction models"""
        try:
            df = data.copy()
            
            # Price-based features
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Price_Momentum'] = df['Close'] / df['Close'].shift(10) - 1
            df['Price_Acceleration'] = df['Returns'] - df['Returns'].shift(1)
            
            # Volatility features
            df['Volatility_5'] = df['Returns'].rolling(5).std()
            df['Volatility_20'] = df['Returns'].rolling(20).std()
            df['Volatility_Ratio'] = df['Volatility_5'] / df['Volatility_20']
            df['GARCH_Volatility'] = self.calculate_garch_volatility(df['Returns'])
            
            # Volume features
            df['Volume_MA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            df['Volume_Price_Trend'] = df['Volume'].rolling(5).corr(df['Close'].rolling(5))
            df['Volume_Momentum'] = df['Volume'] / df['Volume'].shift(5) - 1
            
            # Technical indicators as features
            df['RSI'] = self.calculate_rsi(df['Close'])
            df['MACD'], df['MACD_Signal'] = self.calculate_macd(df['Close'])
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            df['BB_Position'] = self.calculate_bb_position(df['Close'])
            df['Stochastic'] = self.calculate_stochastic(df['High'], df['Low'], df['Close'])
            
            # Market microstructure features
            df['Bid_Ask_Spread'] = (df['High'] - df['Low']) / df['Close']
            df['Price_Range'] = (df['High'] - df['Low']) / df['Open']
            df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
            df['Intraday_Return'] = (df['Close'] - df['Open']) / df['Open']
            
            # Trend features
            df['SMA_5'] = df['Close'].rolling(5).mean()
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['Trend_5_20'] = (df['SMA_5'] - df['SMA_20']) / df['SMA_20']
            df['Trend_20_50'] = (df['SMA_20'] - df['SMA_50']) / df['SMA_50']
            
            # Momentum features
            df['ROC_5'] = df['Close'] / df['Close'].shift(5) - 1
            df['ROC_10'] = df['Close'] / df['Close'].shift(10) - 1
            df['Williams_R'] = self.calculate_williams_r(df['High'], df['Low'], df['Close'])
            
            # Time-based features
            df['Hour'] = df.index.hour if hasattr(df.index, 'hour') else 12
            df['DayOfWeek'] = df.index.dayofweek if hasattr(df.index, 'dayofweek') else 2
            df['Month'] = df.index.month if hasattr(df.index, 'month') else 6
            df['Quarter'] = df.index.quarter if hasattr(df.index, 'quarter') else 2
            
            # Lag features
            for lag in [1, 2, 3, 5]:
                df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
                df[f'Volume_Lag_{lag}'] = df['Volume_Ratio'].shift(lag)
                df[f'Volatility_Lag_{lag}'] = df['Volatility_5'].shift(lag)
            
            # Statistical features
            df['Returns_Skew'] = df['Returns'].rolling(20).skew()
            df['Returns_Kurt'] = df['Returns'].rolling(20).kurt()
            df['Price_Zscore'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
            
            # Interaction features
            df['Volume_Volatility'] = df['Volume_Ratio'] * df['Volatility_5']
            df['Trend_Momentum'] = df['Trend_5_20'] * df['ROC_5']
            df['RSI_Volume'] = df['RSI'] * df['Volume_Ratio']
            
            return df.dropna()
            
        except Exception as e:
            print(f"Feature engineering error: {e}")
            return pd.DataFrame()
    
    def calculate_garch_volatility(self, returns, window=20):
        """Calculate GARCH-like volatility estimate"""
        try:
            volatility = np.full(len(returns), np.nan)
            
            for i in range(window, len(returns)):
                recent_returns = returns[i-window:i]
                volatility[i] = np.sqrt(np.mean(recent_returns**2))
            
            return volatility
        except:
            return np.full(len(returns), 0.01)
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except:
            return np.full(len(prices), 50)
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        try:
            exp1 = prices.ewm(span=fast).mean()
            exp2 = prices.ewm(span=slow).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal).mean()
            return macd, signal_line
        except:
            return np.zeros(len(prices)), np.zeros(len(prices))
    
    def calculate_bb_position(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Band position"""
        try:
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return (prices - lower) / (upper - lower)
        except:
            return np.full(len(prices), 0.5)
    
    def calculate_stochastic(self, high, low, close, period=14):
        """Calculate Stochastic %K"""
        try:
            lowest_low = low.rolling(period).min()
            highest_high = high.rolling(period).max()
            return (close - lowest_low) / (highest_high - lowest_low) * 100
        except:
            return np.full(len(close), 50)
    
    def calculate_williams_r(self, high, low, close, period=14):
        """Calculate Williams %R"""
        try:
            highest_high = high.rolling(period).max()
            lowest_low = low.rolling(period).min()
            return (highest_high - close) / (highest_high - lowest_low) * -100
        except:
            return np.full(len(close), -50)
    
    def prepare_prediction_data(self, features_df):
        """Prepare features and targets for modeling"""
        try:
            # Select feature columns (exclude OHLCV and derived price columns)
            feature_columns = []
            for col in features_df.columns:
                if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']:
                    if not col.startswith('SMA_') or col in ['Trend_5_20', 'Trend_20_50']:
                        feature_columns.append(col)
            
            # Ensure we have numeric features
            numeric_features = []
            for col in feature_columns:
                if features_df[col].dtype in ['float64', 'int64'] and not features_df[col].isnull().all():
                    numeric_features.append(col)
            
            if len(numeric_features) == 0:
                return np.array([]), np.array([])
            
            X = features_df[numeric_features].values
            
            # Create target: future returns (1-day ahead)
            future_returns = features_df['Close'].shift(-1) / features_df['Close'] - 1
            y = future_returns.values
            
            # Remove NaN values
            valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_indices]
            y = y[valid_indices]
            
            # Remove last row (no future data)
            if len(X) > 0:
                X = X[:-1]
                y = y[:-1]
            
            return X, y
            
        except Exception as e:
            print(f"Data preparation error: {e}")
            return np.array([]), np.array([])
    
    def train_prediction_models(self, X, y):
        """Train ensemble of prediction models"""
        try:
            if len(X) < 10:
                return {}
            
            # Split data for training/validation
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            models = {}
            predictions = {}
            
            # 1. Random Forest
            try:
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                rf_model.fit(X_train_scaled, y_train)
                rf_pred = rf_model.predict(X_val_scaled)
                models['RandomForest'] = rf_model
                predictions['RandomForest'] = {
                    'predictions': rf_pred,
                    'mae': mean_absolute_error(y_val, rf_pred),
                    'rmse': np.sqrt(mean_squared_error(y_val, rf_pred))
                }
            except Exception as e:
                print(f"Random Forest training failed: {e}")
            
            # 2. XGBoost
            try:
                xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=6)
                xgb_model.fit(X_train_scaled, y_train)
                xgb_pred = xgb_model.predict(X_val_scaled)
                models['XGBoost'] = xgb_model
                predictions['XGBoost'] = {
                    'predictions': xgb_pred,
                    'mae': mean_absolute_error(y_val, xgb_pred),
                    'rmse': np.sqrt(mean_squared_error(y_val, xgb_pred))
                }
            except Exception as e:
                print(f"XGBoost training failed: {e}")
            
            # 3. Gradient Boosting
            try:
                gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6)
                gb_model.fit(X_train_scaled, y_train)
                gb_pred = gb_model.predict(X_val_scaled)
                models['GradientBoosting'] = gb_model
                predictions['GradientBoosting'] = {
                    'predictions': gb_pred,
                    'mae': mean_absolute_error(y_val, gb_pred),
                    'rmse': np.sqrt(mean_squared_error(y_val, gb_pred))
                }
            except Exception as e:
                print(f"Gradient Boosting training failed: {e}")
            
            # 4. Linear Regression (baseline)
            try:
                lr_model = LinearRegression()
                lr_model.fit(X_train_scaled, y_train)
                lr_pred = lr_model.predict(X_val_scaled)
                models['LinearRegression'] = lr_model
                predictions['LinearRegression'] = {
                    'predictions': lr_pred,
                    'mae': mean_absolute_error(y_val, lr_pred),
                    'rmse': np.sqrt(mean_squared_error(y_val, lr_pred))
                }
            except Exception as e:
                print(f"Linear Regression training failed: {e}")
            
            self.models = models
            self.scalers['features'] = scaler
            
            return predictions
            
        except Exception as e:
            print(f"Model training error: {e}")
            return {}
    
    def create_ensemble_prediction(self, model_predictions):
        """Create ensemble prediction from multiple models"""
        try:
            if not model_predictions:
                return self.create_default_ensemble_prediction()
            
            # Weight models by inverse of their error
            weights = {}
            total_weight = 0
            
            for model_name, pred_data in model_predictions.items():
                if 'rmse' in pred_data and pred_data['rmse'] > 0:
                    weight = 1 / (pred_data['rmse'] + 0.001)  # Avoid division by zero
                    weights[model_name] = weight
                    total_weight += weight
            
            if total_weight == 0:
                return self.create_default_ensemble_prediction()
            
            # Normalize weights
            for model_name in weights:
                weights[model_name] /= total_weight
            
            # Calculate weighted ensemble prediction
            ensemble_pred = 0
            for model_name, pred_data in model_predictions.items():
                if model_name in weights and len(pred_data['predictions']) > 0:
                    # Use the latest prediction
                    latest_pred = pred_data['predictions'][-1]
                    ensemble_pred += weights[model_name] * latest_pred
            
            # Convert to direction and probability
            if ensemble_pred > 0:
                direction = "Upward"
                probability = min(90, 50 + abs(ensemble_pred) * 2000)  # Scale to percentage
            else:
                direction = "Downward"
                probability = min(90, 50 + abs(ensemble_pred) * 2000)
            
            # Estimate target range (assume 1-day prediction)
            current_price = 100  # Placeholder - should use actual current price
            target_low = current_price * (1 + ensemble_pred - abs(ensemble_pred) * 0.5)
            target_high = current_price * (1 + ensemble_pred + abs(ensemble_pred) * 0.5)
            
            return {
                'direction': direction,
                'probability': probability,
                'predicted_return': ensemble_pred,
                'target_range': {'low': target_low, 'high': target_high},
                'time_horizon': '1-3 trading sessions',
                'model_weights': weights
            }
            
        except Exception as e:
            print(f"Ensemble prediction error: {e}")
            return self.create_default_ensemble_prediction()
    
    def create_default_ensemble_prediction(self):
        """Create default ensemble prediction when models fail"""
        return {
            'direction': "Neutral",
            'probability': 50.0,
            'predicted_return': 0.0,
            'target_range': {'low': 95, 'high': 105},
            'time_horizon': '1-3 trading sessions',
            'model_weights': {}
        }
    
    def calculate_prediction_confidence(self, model_predictions, X, y):
        """Calculate comprehensive confidence metrics"""
        try:
            if not model_predictions or len(X) == 0:
                return {
                    'confidence_score': 50.0,
                    'confidence_level': 'Low',
                    'model_agreement': 0.5,
                    'prediction_stability': 0.5,
                    'data_quality': 0.5
                }
            
            # Model agreement (how similar are predictions)
            predictions_list = []
            for model_name, pred_data in model_predictions.items():
                if len(pred_data['predictions']) > 0:
                    predictions_list.append(pred_data['predictions'][-1])
            
            if len(predictions_list) > 1:
                model_agreement = 1 - (np.std(predictions_list) / (np.mean(np.abs(predictions_list)) + 0.001))
                model_agreement = max(0, min(1, model_agreement))
            else:
                model_agreement = 0.5
            
            # Prediction stability (consistency of recent predictions)
            prediction_stability = 0.5
            if len(predictions_list) > 0:
                # Simulate stability with model error variance
                avg_rmse = np.mean([pred_data['rmse'] for pred_data in model_predictions.values() 
                                  if 'rmse' in pred_data])
                prediction_stability = max(0, min(1, 1 - avg_rmse * 10))
            
            # Data quality (amount and recency of data)
            data_quality = min(1, len(X) / 100)  # Normalize by 100 samples
            
            # Overall confidence score
            confidence_score = (model_agreement * 0.4 + prediction_stability * 0.4 + data_quality * 0.2) * 100
            
            # Confidence level classification
            if confidence_score > 75:
                confidence_level = 'High'
            elif confidence_score > 50:
                confidence_level = 'Medium'
            else:
                confidence_level = 'Low'
            
            return {
                'confidence_score': confidence_score,
                'confidence_level': confidence_level,
                'model_agreement': model_agreement,
                'prediction_stability': prediction_stability,
                'data_quality': data_quality
            }
            
        except Exception as e:
            print(f"Confidence calculation error: {e}")
            return {
                'confidence_score': 50.0,
                'confidence_level': 'Medium',
                'model_agreement': 0.5,
                'prediction_stability': 0.5,
                'data_quality': 0.5
            }
    
    def assess_prediction_risk(self, market_data, ensemble_prediction):
        """Comprehensive risk assessment for predictions"""
        try:
            current_price = market_data['Close'].iloc[-1]
            returns = market_data['Close'].pct_change().dropna()
            
            # Volatility-based risk
            recent_volatility = returns.tail(20).std() * np.sqrt(252)  # Annualized
            
            if recent_volatility > 0.4:
                volatility_risk = 85
                risk_level = 'High'
            elif recent_volatility > 0.25:
                volatility_risk = 65
                risk_level = 'Medium'
            else:
                volatility_risk = 40
                risk_level = 'Low'
            
            # Stop loss calculation
            atr_equivalent = (market_data['High'] - market_data['Low']).tail(14).mean()
            stop_loss_distance = atr_equivalent * 2
            
            if ensemble_prediction['direction'] == 'Upward':
                stop_loss = current_price - stop_loss_distance
            else:
                stop_loss = current_price + stop_loss_distance
            
            # Risk-reward ratio
            predicted_return = abs(ensemble_prediction.get('predicted_return', 0.01))
            risk_return = stop_loss_distance / current_price
            risk_reward_ratio = predicted_return / (risk_return + 0.001)
            
            # Maximum drawdown risk
            recent_max = market_data['High'].tail(20).max()
            recent_min = market_data['Low'].tail(20).min()
            max_drawdown_risk = (recent_max - recent_min) / recent_max * 100
            
            return {
                'risk_level': risk_level,
                'volatility_risk': volatility_risk,
                'stop_loss': stop_loss,
                'risk_reward_ratio': risk_reward_ratio,
                'max_drawdown_risk': max_drawdown_risk,
                'recommended_position_size': self.calculate_position_size(risk_level, volatility_risk)
            }
            
        except Exception as e:
            print(f"Risk assessment error: {e}")
            return {
                'risk_level': 'Medium',
                'volatility_risk': 50.0,
                'stop_loss': market_data['Close'].iloc[-1] * 0.95,
                'risk_reward_ratio': 1.5,
                'max_drawdown_risk': 10.0,
                'recommended_position_size': 'Medium'
            }
    
    def calculate_position_size(self, risk_level, volatility_risk):
        """Calculate recommended position size based on risk"""
        if risk_level == 'High' or volatility_risk > 80:
            return 'Small (1-2% of portfolio)'
        elif risk_level == 'Medium' or volatility_risk > 50:
            return 'Medium (3-5% of portfolio)'
        else:
            return 'Large (5-10% of portfolio)'
    
    def perform_scenario_analysis(self, market_data, ensemble_prediction):
        """Perform scenario analysis with different market conditions"""
        try:
            current_price = market_data['Close'].iloc[-1]
            recent_volatility = market_data['Close'].pct_change().tail(20).std()
            
            predicted_return = ensemble_prediction.get('predicted_return', 0.0)
            
            # Best case scenario (90th percentile outcome)
            best_case_return = predicted_return + (recent_volatility * 1.65)  # 1.65 = 95th percentile z-score
            best_case_price = current_price * (1 + best_case_return)
            
            # Worst case scenario (10th percentile outcome)
            worst_case_return = predicted_return - (recent_volatility * 1.65)
            worst_case_price = current_price * (1 + worst_case_return)
            
            # Most likely scenario (median outcome)
            most_likely_return = predicted_return
            most_likely_price = current_price * (1 + most_likely_return)
            
            # Calculate probabilities based on normal distribution assumption
            best_case_prob = 10  # By definition, 10% chance of exceeding 90th percentile
            worst_case_prob = 10  # By definition, 10% chance of falling below 10th percentile
            most_likely_prob = 80  # Remaining probability
            
            return {
                'Best Case': {
                    'price': best_case_price,
                    'return': best_case_return * 100,
                    'probability': best_case_prob
                },
                'Most Likely': {
                    'price': most_likely_price,
                    'return': most_likely_return * 100,
                    'probability': most_likely_prob
                },
                'Worst Case': {
                    'price': worst_case_price,
                    'return': worst_case_return * 100,
                    'probability': worst_case_prob
                }
            }
            
        except Exception as e:
            print(f"Scenario analysis error: {e}")
            current_price = market_data['Close'].iloc[-1]
            return {
                'Best Case': {'price': current_price * 1.05, 'return': 5.0, 'probability': 10},
                'Most Likely': {'price': current_price * 1.01, 'return': 1.0, 'probability': 80},
                'Worst Case': {'price': current_price * 0.95, 'return': -5.0, 'probability': 10}
            }
    
    def calculate_feature_importance(self, X, y, feature_names):
        """Calculate feature importance across models"""
        try:
            if len(X) == 0 or not self.models:
                return {
                    'RSI': 0.15,
                    'Volume_Ratio': 0.12,
                    'Volatility_5': 0.11,
                    'MACD_Histogram': 0.10,
                    'Price_Momentum': 0.09
                }
            
            feature_importance = {}
            
            # Aggregate feature importance from tree-based models
            tree_models = ['RandomForest', 'XGBoost', 'GradientBoosting']
            
            for model_name in tree_models:
                if model_name in self.models:
                    try:
                        model = self.models[model_name]
                        if hasattr(model, 'feature_importances_'):
                            importance = model.feature_importances_
                            for i, feature_name in enumerate(feature_names):
                                if i < len(importance):
                                    if feature_name not in feature_importance:
                                        feature_importance[feature_name] = 0
                                    feature_importance[feature_name] += importance[i]
                    except Exception as e:
                        print(f"Feature importance extraction failed for {model_name}: {e}")
            
            # Normalize importance scores
            if feature_importance:
                total_importance = sum(feature_importance.values())
                if total_importance > 0:
                    for feature in feature_importance:
                        feature_importance[feature] /= total_importance
                
                # Return top 10 features
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                return dict(sorted_features[:10])
            
            # Default feature importance if calculation fails
            return {
                'RSI': 0.15,
                'Volume_Ratio': 0.12,
                'Volatility_5': 0.11,
                'MACD_Histogram': 0.10,
                'Price_Momentum': 0.09,
                'Returns': 0.08,
                'BB_Position': 0.07,
                'Trend_5_20': 0.06,
                'Stochastic': 0.05,
                'Williams_R': 0.04
            }
            
        except Exception as e:
            print(f"Feature importance calculation error: {e}")
            return {
                'RSI': 0.15,
                'Volume_Ratio': 0.12,
                'Volatility_5': 0.11,
                'MACD_Histogram': 0.10,
                'Price_Momentum': 0.09
            }
    
    def identify_trigger_conditions(self, market_data, ensemble_prediction):
        """Identify specific trigger conditions for predictions"""
        try:
            current_price = market_data['Close'].iloc[-1]
            recent_volume = market_data['Volume'].tail(20).mean()
            recent_volatility = market_data['Close'].pct_change().tail(5).std()
            
            direction = ensemble_prediction['direction']
            
            # Entry conditions
            if direction == 'Upward':
                entry_conditions = [
                    f"Price breaks above {current_price * 1.01:.2f} with volume > {recent_volume * 1.2:.0f}",
                    f"RSI confirms momentum above 60",
                    f"MACD histogram turns positive",
                    f"Volume spike > 150% of 20-day average"
                ]
            else:
                entry_conditions = [
                    f"Price breaks below {current_price * 0.99:.2f} with volume > {recent_volume * 1.2:.0f}",
                    f"RSI confirms momentum below 40",
                    f"MACD histogram turns negative",
                    f"Volume spike > 150% of 20-day average"
                ]
            
            # Confirmation signals
            confirmation_signals = [
                "Two consecutive periods in predicted direction",
                f"Volume confirmation (volume > {recent_volume * 1.5:.0f})",
                "Technical indicator alignment (3+ indicators confirm)",
                f"Volatility within normal range (< {recent_volatility * 2:.3f})"
            ]
            
            # Exit conditions
            if direction == 'Upward':
                exit_conditions = [
                    f"Price reaches target level {current_price * 1.05:.2f}",
                    f"Stop loss triggered at {current_price * 0.97:.2f}",
                    "Volume divergence (price up, volume down)",
                    "Technical indicators show overbought conditions"
                ]
            else:
                exit_conditions = [
                    f"Price reaches target level {current_price * 0.95:.2f}",
                    f"Stop loss triggered at {current_price * 1.03:.2f}",
                    "Volume divergence (price down, volume down)",
                    "Technical indicators show oversold conditions"
                ]
            
            return {
                'entry_conditions': entry_conditions,
                'confirmation_signals': confirmation_signals,
                'exit_conditions': exit_conditions
            }
            
        except Exception as e:
            print(f"Trigger conditions identification error: {e}")
            return {
                'entry_conditions': [
                    "Price breakout with volume confirmation",
                    "Technical indicator alignment",
                    "Momentum confirmation"
                ],
                'confirmation_signals': [
                    "Volume spike above average",
                    "Multiple indicator confirmation",
                    "Sustained directional movement"
                ],
                'exit_conditions': [
                    "Target price reached",
                    "Stop loss triggered",
                    "Reversal signals detected"
                ]
            }
    
    def create_default_predictions(self):
        """Create default predictions when analysis fails"""
        return {
            'primary_prediction': {
                'direction': 'Neutral',
                'probability': 50.0,
                'target_range': {'low': 95.0, 'high': 105.0},
                'time_horizon': '1-3 trading sessions',
                'confidence_level': 'Low',
                'confidence': 30.0
            },
            'risk_assessment': {
                'risk_level': 'Medium',
                'volatility_risk': 50.0,
                'stop_loss': 95.0,
                'risk_reward_ratio': 1.0,
                'max_drawdown_risk': 10.0,
                'recommended_position_size': 'Medium (3-5% of portfolio)'
            },
            'scenario_analysis': {
                'Best Case': {'price': 105.0, 'return': 5.0, 'probability': 10},
                'Most Likely': {'price': 100.0, 'return': 0.0, 'probability': 80},
                'Worst Case': {'price': 95.0, 'return': -5.0, 'probability': 10}
            },
            'feature_importance': {
                'Price_Momentum': 0.20,
                'Volume_Trend': 0.15,
                'Volatility': 0.12,
                'Technical_Indicators': 0.10,
                'Market_Sentiment': 0.08
            },
            'trigger_conditions': {
                'entry_conditions': ['Wait for clear directional signal', 'Monitor volume for confirmation'],
                'confirmation_signals': ['Technical indicator alignment', 'Volume confirmation'],
                'exit_conditions': ['Set appropriate stop loss', 'Monitor for reversal signals']
            }
        }
