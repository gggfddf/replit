import pandas as pd
import numpy as np
# Use finta for technical indicators (working alternative)
try:
    from finta import TA as finta_ta
    FINTA_AVAILABLE = True
except ImportError:
    FINTA_AVAILABLE = False

# Optional talib import 
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class TechnicalAnalyzer:
    """
    Comprehensive Technical Analysis Engine that generates detailed technical intelligence
    with multi-timeframe analysis, composite signals, and ML-backed insights
    """
    
    def __init__(self):
        self.analysis_results = {}
        self.composite_signals = {}
        
    def generate_comprehensive_analysis(self, market_data):
        """
        Generate comprehensive technical analysis report
        """
        try:
            # Ensure we have the required data
            if market_data.empty or len(market_data) < 50:
                return self.create_default_analysis()
            
            # Multi-timeframe trend analysis
            trend_analysis = self.analyze_multi_timeframe_trends(market_data)
            
            # Support and resistance levels
            support_resistance = self.identify_support_resistance_levels(market_data)
            
            # Momentum analysis
            momentum_analysis = self.analyze_momentum_indicators(market_data)
            
            # Volume analysis
            volume_analysis = self.analyze_volume_patterns(market_data)
            
            # Volatility analysis
            volatility_analysis = self.analyze_volatility_patterns(market_data)
            
            # Composite technical signals
            composite_signals = self.generate_composite_signals(market_data)
            
            # Technical predictions
            technical_predictions = self.generate_technical_predictions(market_data)
            
            # Risk assessment from technical perspective
            technical_risk = self.assess_technical_risk(market_data)
            
            return {
                'trend_strength': trend_analysis['strength'],
                'trend_direction': trend_analysis['direction'],
                'trend_confidence': trend_analysis['confidence'],
                'support_resistance': support_resistance,
                'momentum_signals': momentum_analysis,
                'volume_signals': volume_analysis,
                'volatility_signals': volatility_analysis,
                'composite_signals': composite_signals,
                'technical_predictions': technical_predictions,
                'technical_risk': technical_risk,
                'multi_timeframe_alignment': self.check_timeframe_alignment(market_data)
            }
            
        except Exception as e:
            print(f"Technical analysis error: {e}")
            return self.create_default_analysis()
    
    def analyze_multi_timeframe_trends(self, data):
        """
        Analyze trends across multiple timeframes
        """
        try:
            current_price = data['Close'].iloc[-1]
            
            # Short-term trend (5-day SMA)
            sma_5 = data['Close'].rolling(5).mean().iloc[-1]
            short_trend = "Bullish" if current_price > sma_5 else "Bearish"
            short_strength = abs(current_price - sma_5) / sma_5 * 100
            
            # Medium-term trend (20-day SMA)
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            medium_trend = "Bullish" if current_price > sma_20 else "Bearish"
            medium_strength = abs(current_price - sma_20) / sma_20 * 100
            
            # Long-term trend (50-day SMA)
            sma_50 = data['Close'].rolling(50).mean().iloc[-1]
            long_trend = "Bullish" if current_price > sma_50 else "Bearish"
            long_strength = abs(current_price - sma_50) / sma_50 * 100
            
            # Overall trend assessment
            bullish_count = sum([short_trend == "Bullish", medium_trend == "Bullish", long_trend == "Bullish"])
            
            if bullish_count >= 2:
                overall_direction = "Bullish"
            elif bullish_count <= 1:
                overall_direction = "Bearish"
            else:
                overall_direction = "Neutral"
            
            # Trend strength (1-10 scale)
            avg_strength = (short_strength + medium_strength + long_strength) / 3
            strength_score = min(10, max(1, avg_strength * 2))
            
            # Trend confidence
            alignment_score = bullish_count / 3 * 100
            strength_confidence = min(100, avg_strength * 10 + 50)
            confidence = (alignment_score + strength_confidence) / 2
            
            return {
                'direction': overall_direction,
                'strength': strength_score,
                'confidence': confidence,
                'short_term': short_trend,
                'medium_term': medium_trend,
                'long_term': long_trend,
                'alignment_score': alignment_score
            }
            
        except Exception as e:
            print(f"Trend analysis error: {e}")
            return {
                'direction': 'Neutral',
                'strength': 5,
                'confidence': 50,
                'short_term': 'Neutral',
                'medium_term': 'Neutral',
                'long_term': 'Neutral',
                'alignment_score': 50
            }
    
    def identify_support_resistance_levels(self, data):
        """
        Identify key support and resistance levels with strength scoring
        """
        try:
            # Find pivot points using local maxima/minima
            highs = data['High'].values
            lows = data['Low'].values
            
            # Find resistance levels (local maxima)
            resistance_indices = find_peaks(highs, distance=5, prominence=np.std(highs)*0.5)[0]
            
            # Find support levels (local minima)
            support_indices = find_peaks(-lows, distance=5, prominence=np.std(lows)*0.5)[0]
            
            # Calculate level strength based on touches and volume
            resistance_levels = []
            for idx in resistance_indices[-10:]:  # Last 10 resistance levels
                level = highs[idx]
                strength = self.calculate_level_strength(data, level, 'resistance')
                resistance_levels.append({
                    'level': level,
                    'strength': strength,
                    'index': idx,
                    'date': data.index[idx] if idx < len(data.index) else data.index[-1]
                })
            
            support_levels = []
            for idx in support_indices[-10:]:  # Last 10 support levels
                level = lows[idx]
                strength = self.calculate_level_strength(data, level, 'support')
                support_levels.append({
                    'level': level,
                    'strength': strength,
                    'index': idx,
                    'date': data.index[idx] if idx < len(data.index) else data.index[-1]
                })
            
            # Sort by strength
            resistance_levels.sort(key=lambda x: x['strength'], reverse=True)
            support_levels.sort(key=lambda x: x['strength'], reverse=True)
            
            return {
                'resistance': [r['level'] for r in resistance_levels[:3]],
                'support': [s['level'] for s in support_levels[:3]],
                'resistance_detailed': resistance_levels[:3],
                'support_detailed': support_levels[:3]
            }
            
        except Exception as e:
            print(f"Support/Resistance analysis error: {e}")
            current_price = data['Close'].iloc[-1]
            return {
                'resistance': [current_price * 1.05, current_price * 1.10],
                'support': [current_price * 0.95, current_price * 0.90],
                'resistance_detailed': [],
                'support_detailed': []
            }
    
    def calculate_level_strength(self, data, level, level_type, tolerance=0.02):
        """
        Calculate the strength of a support/resistance level
        """
        try:
            if level_type == 'resistance':
                touches = np.sum(np.abs(data['High'] - level) / level < tolerance)
                # Consider volume at touches
                volume_weight = data.loc[np.abs(data['High'] - level) / level < tolerance, 'Volume'].mean()
            else:
                touches = np.sum(np.abs(data['Low'] - level) / level < tolerance)
                volume_weight = data.loc[np.abs(data['Low'] - level) / level < tolerance, 'Volume'].mean()
            
            # Normalize volume weight
            avg_volume = data['Volume'].mean()
            volume_factor = volume_weight / avg_volume if avg_volume > 0 else 1
            
            # Strength score
            strength = touches * volume_factor
            
            return min(100, max(1, strength * 10))
            
        except Exception as e:
            return 50  # Default strength
    
    def analyze_momentum_indicators(self, data):
        """
        Analyze momentum indicators with pattern detection
        """
        try:
            momentum_signals = {}
            
            # RSI Analysis
            rsi = talib.RSI(data['Close'].values, timeperiod=14)
            current_rsi = rsi[-1] if len(rsi) > 0 else 50
            
            if current_rsi > 80:
                rsi_signal = 'Strong Sell'
            elif current_rsi > 70:
                rsi_signal = 'Sell'
            elif current_rsi < 20:
                rsi_signal = 'Strong Buy'
            elif current_rsi < 30:
                rsi_signal = 'Buy'
            else:
                rsi_signal = 'Neutral'
            
            momentum_signals['RSI'] = {
                'signal': rsi_signal,
                'value': current_rsi
            }
            
            # MACD Analysis
            macd, signal_line, histogram = talib.MACD(data['Close'].values)
            if len(macd) > 1:
                current_macd = macd[-1]
                current_signal = signal_line[-1]
                
                if current_macd > current_signal:
                    if macd[-2] <= signal_line[-2]:  # Bullish crossover
                        macd_signal = 'Strong Buy'
                    else:
                        macd_signal = 'Buy'
                else:
                    if macd[-2] >= signal_line[-2]:  # Bearish crossover
                        macd_signal = 'Strong Sell'
                    else:
                        macd_signal = 'Sell'
            else:
                macd_signal = 'Neutral'
                current_macd = 0
            
            momentum_signals['MACD'] = {
                'signal': macd_signal,
                'value': current_macd
            }
            
            # Stochastic Analysis
            slowk, slowd = talib.STOCH(data['High'].values, data['Low'].values, data['Close'].values)
            if len(slowk) > 0:
                current_stoch = slowk[-1]
                
                if current_stoch > 80:
                    stoch_signal = 'Sell'
                elif current_stoch < 20:
                    stoch_signal = 'Buy'
                else:
                    stoch_signal = 'Neutral'
            else:
                stoch_signal = 'Neutral'
                current_stoch = 50
            
            momentum_signals['Stochastic'] = {
                'signal': stoch_signal,
                'value': current_stoch
            }
            
            # Williams %R Analysis
            willr = talib.WILLR(data['High'].values, data['Low'].values, data['Close'].values)
            if len(willr) > 0:
                current_willr = willr[-1]
                
                if current_willr > -20:
                    willr_signal = 'Sell'
                elif current_willr < -80:
                    willr_signal = 'Buy'
                else:
                    willr_signal = 'Neutral'
            else:
                willr_signal = 'Neutral'
                current_willr = -50
            
            momentum_signals['Williams_R'] = {
                'signal': willr_signal,
                'value': current_willr
            }
            
            return momentum_signals
            
        except Exception as e:
            print(f"Momentum analysis error: {e}")
            return {
                'RSI': {'signal': 'Neutral', 'value': 50},
                'MACD': {'signal': 'Neutral', 'value': 0},
                'Stochastic': {'signal': 'Neutral', 'value': 50},
                'Williams_R': {'signal': 'Neutral', 'value': -50}
            }
    
    def analyze_volume_patterns(self, data):
        """
        Analyze volume patterns and signals
        """
        try:
            volume_signals = {}
            
            # On Balance Volume (OBV)
            obv = talib.OBV(data['Close'].values, data['Volume'].values)
            if len(obv) >= 20:
                obv_slope = np.polyfit(range(20), obv[-20:], 1)[0]
                obv_trend = 'Rising' if obv_slope > 0 else 'Falling'
            else:
                obv_trend = 'Neutral'
            
            volume_signals['OBV'] = {
                'signal': 'Buy' if obv_trend == 'Rising' else 'Sell' if obv_trend == 'Falling' else 'Neutral',
                'trend': obv_trend
            }
            
            # Volume Rate of Change
            volume_roc = data['Volume'].pct_change(periods=10).iloc[-1] * 100
            
            if volume_roc > 50:
                vol_roc_signal = 'High Activity'
            elif volume_roc < -30:
                vol_roc_signal = 'Low Activity'
            else:
                vol_roc_signal = 'Normal Activity'
            
            volume_signals['Volume_ROC'] = {
                'signal': vol_roc_signal,
                'value': volume_roc
            }
            
            # Volume Moving Average Convergence
            vol_ma_short = data['Volume'].rolling(10).mean().iloc[-1]
            vol_ma_long = data['Volume'].rolling(30).mean().iloc[-1]
            
            if vol_ma_short > vol_ma_long * 1.2:
                vol_ma_signal = 'Increasing Volume'
            elif vol_ma_short < vol_ma_long * 0.8:
                vol_ma_signal = 'Decreasing Volume'
            else:
                vol_ma_signal = 'Stable Volume'
            
            volume_signals['Volume_MA'] = {
                'signal': vol_ma_signal,
                'short_ma': vol_ma_short,
                'long_ma': vol_ma_long
            }
            
            return volume_signals
            
        except Exception as e:
            print(f"Volume analysis error: {e}")
            return {
                'OBV': {'signal': 'Neutral', 'trend': 'Neutral'},
                'Volume_ROC': {'signal': 'Normal Activity', 'value': 0},
                'Volume_MA': {'signal': 'Stable Volume', 'short_ma': 0, 'long_ma': 0}
            }
    
    def analyze_volatility_patterns(self, data):
        """
        Analyze volatility patterns and signals
        """
        try:
            volatility_signals = {}
            
            # Average True Range (ATR)
            atr = talib.ATR(data['High'].values, data['Low'].values, data['Close'].values)
            if len(atr) >= 20:
                current_atr = atr[-1]
                avg_atr = np.mean(atr[-20:])
                atr_ratio = current_atr / avg_atr
                
                if atr_ratio > 1.5:
                    atr_signal = 'High Volatility'
                elif atr_ratio < 0.7:
                    atr_signal = 'Low Volatility'
                else:
                    atr_signal = 'Normal Volatility'
            else:
                atr_signal = 'Normal Volatility'
                atr_ratio = 1
            
            volatility_signals['ATR'] = {
                'signal': atr_signal,
                'ratio': atr_ratio
            }
            
            # Bollinger Band Width
            upper, middle, lower = talib.BBANDS(data['Close'].values)
            if len(upper) > 0:
                bb_width = (upper[-1] - lower[-1]) / middle[-1]
                bb_position = (data['Close'].iloc[-1] - lower[-1]) / (upper[-1] - lower[-1])
                
                if bb_width < 0.1:
                    bb_signal = 'Squeeze - Breakout Expected'
                elif bb_position > 0.8:
                    bb_signal = 'Near Upper Band - Overbought'
                elif bb_position < 0.2:
                    bb_signal = 'Near Lower Band - Oversold'
                else:
                    bb_signal = 'Normal Range'
            else:
                bb_signal = 'Normal Range'
                bb_position = 0.5
            
            volatility_signals['Bollinger_Bands'] = {
                'signal': bb_signal,
                'position': bb_position
            }
            
            return volatility_signals
            
        except Exception as e:
            print(f"Volatility analysis error: {e}")
            return {
                'ATR': {'signal': 'Normal Volatility', 'ratio': 1},
                'Bollinger_Bands': {'signal': 'Normal Range', 'position': 0.5}
            }
    
    def generate_composite_signals(self, data):
        """
        Generate composite technical signals combining multiple indicators
        """
        try:
            # Get all individual signals
            trend_analysis = self.analyze_multi_timeframe_trends(data)
            momentum_signals = self.analyze_momentum_indicators(data)
            volume_signals = self.analyze_volume_patterns(data)
            
            # Scoring system
            bullish_score = 0
            bearish_score = 0
            total_signals = 0
            
            # Trend signals (weight: 3)
            if trend_analysis['direction'] == 'Bullish':
                bullish_score += 3
            elif trend_analysis['direction'] == 'Bearish':
                bearish_score += 3
            total_signals += 3
            
            # Momentum signals (weight: 2 each)
            for indicator, signal_data in momentum_signals.items():
                signal = signal_data['signal']
                if 'Buy' in signal:
                    bullish_score += 2 if 'Strong' in signal else 1
                elif 'Sell' in signal:
                    bearish_score += 2 if 'Strong' in signal else 1
                total_signals += 2
            
            # Volume signals (weight: 1 each)
            for indicator, signal_data in volume_signals.items():
                signal = signal_data['signal']
                if any(keyword in signal for keyword in ['Buy', 'Rising', 'Increasing', 'High Activity']):
                    bullish_score += 1
                elif any(keyword in signal for keyword in ['Sell', 'Falling', 'Decreasing']):
                    bearish_score += 1
                total_signals += 1
            
            # Calculate composite signal
            if total_signals > 0:
                bullish_percentage = (bullish_score / total_signals) * 100
                bearish_percentage = (bearish_score / total_signals) * 100
                
                if bullish_percentage > 60:
                    composite_signal = 'Strong Buy'
                elif bullish_percentage > 40:
                    composite_signal = 'Buy'
                elif bearish_percentage > 60:
                    composite_signal = 'Strong Sell'
                elif bearish_percentage > 40:
                    composite_signal = 'Sell'
                else:
                    composite_signal = 'Neutral'
                
                confidence = max(bullish_percentage, bearish_percentage)
            else:
                composite_signal = 'Neutral'
                confidence = 50
                bullish_percentage = 50
                bearish_percentage = 50
            
            return {
                'overall_signal': composite_signal,
                'confidence': confidence,
                'bullish_score': bullish_score,
                'bearish_score': bearish_score,
                'bullish_percentage': bullish_percentage,
                'bearish_percentage': bearish_percentage,
                'signal_strength': abs(bullish_percentage - bearish_percentage)
            }
            
        except Exception as e:
            print(f"Composite signals error: {e}")
            return {
                'overall_signal': 'Neutral',
                'confidence': 50,
                'bullish_score': 0,
                'bearish_score': 0,
                'bullish_percentage': 50,
                'bearish_percentage': 50,
                'signal_strength': 0
            }
    
    def generate_technical_predictions(self, data):
        """
        Generate ML-backed technical predictions
        """
        try:
            # Simple technical prediction based on indicators
            current_price = data['Close'].iloc[-1]
            
            # Moving average prediction
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            price_to_sma = current_price / sma_20
            
            # Momentum prediction
            rsi = talib.RSI(data['Close'].values, timeperiod=14)[-1]
            
            # Volatility prediction
            volatility = data['Returns'].rolling(20).std().iloc[-1] if 'Returns' in data.columns else 0.02
            
            # Combine factors for prediction
            if price_to_sma > 1.02 and rsi < 70:
                direction = 'Upward'
                target_price = current_price * (1 + volatility * 2)
                probability = min(85, 60 + (price_to_sma - 1) * 500)
            elif price_to_sma < 0.98 and rsi > 30:
                direction = 'Downward'
                target_price = current_price * (1 - volatility * 2)
                probability = min(85, 60 + (1 - price_to_sma) * 500)
            else:
                direction = 'Sideways'
                target_price = current_price
                probability = 50
            
            return {
                'direction': direction,
                'target_price': target_price,
                'probability': probability,
                'time_horizon': '5-10 trading days',
                'confidence_level': 'Medium' if probability > 60 else 'Low'
            }
            
        except Exception as e:
            print(f"Technical predictions error: {e}")
            current_price = data['Close'].iloc[-1] if not data.empty else 100
            return {
                'direction': 'Neutral',
                'target_price': current_price,
                'probability': 50,
                'time_horizon': '5-10 trading days',
                'confidence_level': 'Low'
            }
    
    def assess_technical_risk(self, data):
        """
        Assess risk from technical perspective
        """
        try:
            # Volatility risk
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            if volatility > 0.4:
                volatility_risk = 'High'
            elif volatility > 0.25:
                volatility_risk = 'Medium'
            else:
                volatility_risk = 'Low'
            
            # Support/resistance risk
            current_price = data['Close'].iloc[-1]
            support_resistance = self.identify_support_resistance_levels(data)
            
            # Distance to nearest support/resistance
            nearest_resistance = min([r for r in support_resistance['resistance'] if r > current_price], 
                                   default=current_price * 1.1)
            nearest_support = max([s for s in support_resistance['support'] if s < current_price], 
                                default=current_price * 0.9)
            
            upside_risk = (nearest_resistance - current_price) / current_price * 100
            downside_risk = (current_price - nearest_support) / current_price * 100
            
            # Technical risk score
            if volatility_risk == 'High' or min(upside_risk, downside_risk) < 3:
                risk_level = 'High'
            elif volatility_risk == 'Medium' or min(upside_risk, downside_risk) < 7:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            return {
                'risk_level': risk_level,
                'volatility_risk': volatility_risk,
                'volatility_value': volatility,
                'upside_resistance': upside_risk,
                'downside_support': downside_risk,
                'recommended_stop_loss': nearest_support
            }
            
        except Exception as e:
            print(f"Technical risk assessment error: {e}")
            current_price = data['Close'].iloc[-1] if not data.empty else 100
            return {
                'risk_level': 'Medium',
                'volatility_risk': 'Medium',
                'volatility_value': 0.25,
                'upside_resistance': 5,
                'downside_support': 5,
                'recommended_stop_loss': current_price * 0.95
            }
    
    def check_timeframe_alignment(self, data):
        """
        Check alignment across different timeframes
        """
        try:
            current_price = data['Close'].iloc[-1]
            
            # Multiple moving averages
            sma_5 = data['Close'].rolling(5).mean().iloc[-1]
            sma_10 = data['Close'].rolling(10).mean().iloc[-1]
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(50).mean().iloc[-1]
            
            # Check alignment
            bullish_alignment = (current_price > sma_5 > sma_10 > sma_20 > sma_50)
            bearish_alignment = (current_price < sma_5 < sma_10 < sma_20 < sma_50)
            
            if bullish_alignment:
                alignment = 'Strong Bullish Alignment'
                alignment_score = 100
            elif bearish_alignment:
                alignment = 'Strong Bearish Alignment'
                alignment_score = 0
            else:
                # Partial alignment
                above_count = sum([current_price > sma_5, current_price > sma_10, 
                                 current_price > sma_20, current_price > sma_50])
                alignment_score = above_count / 4 * 100
                
                if alignment_score > 75:
                    alignment = 'Bullish Alignment'
                elif alignment_score < 25:
                    alignment = 'Bearish Alignment'
                else:
                    alignment = 'Mixed Alignment'
            
            return {
                'alignment': alignment,
                'alignment_score': alignment_score,
                'timeframes_bullish': above_count if 'above_count' in locals() else (4 if bullish_alignment else 0),
                'timeframes_bearish': 4 - above_count if 'above_count' in locals() else (4 if bearish_alignment else 0)
            }
            
        except Exception as e:
            print(f"Timeframe alignment error: {e}")
            return {
                'alignment': 'Mixed Alignment',
                'alignment_score': 50,
                'timeframes_bullish': 2,
                'timeframes_bearish': 2
            }
    
    def create_default_analysis(self):
        """
        Create default analysis when calculations fail
        """
        return {
            'trend_strength': 5,
            'trend_direction': 'Neutral',
            'trend_confidence': 50,
            'support_resistance': {
                'resistance': [105, 110],
                'support': [95, 90],
                'resistance_detailed': [],
                'support_detailed': []
            },
            'momentum_signals': {
                'RSI': {'signal': 'Neutral', 'value': 50},
                'MACD': {'signal': 'Neutral', 'value': 0},
                'Stochastic': {'signal': 'Neutral', 'value': 50}
            },
            'volume_signals': {
                'OBV': {'signal': 'Neutral', 'trend': 'Neutral'}
            },
            'volatility_signals': {
                'ATR': {'signal': 'Normal Volatility', 'ratio': 1}
            },
            'composite_signals': {
                'overall_signal': 'Neutral',
                'confidence': 50
            },
            'technical_predictions': {
                'direction': 'Neutral',
                'target_price': 100,
                'probability': 50
            },
            'technical_risk': {
                'risk_level': 'Medium',
                'volatility_risk': 'Medium'
            },
            'multi_timeframe_alignment': {
                'alignment': 'Mixed Alignment',
                'alignment_score': 50
            }
        }
