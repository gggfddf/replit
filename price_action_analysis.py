import pandas as pd
import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class PriceActionAnalyzer:
    """
    Comprehensive Price Action Analysis Engine focusing on pure price movements,
    volume-price relationships, and market structure analysis
    """
    
    def __init__(self):
        self.support_levels = []
        self.resistance_levels = []
        self.pattern_history = []
        
    def analyze_price_action(self, market_data):
        """
        Generate comprehensive price action analysis
        """
        try:
            if market_data.empty or len(market_data) < 30:
                return self.create_default_price_action_analysis()
            
            # Market structure analysis
            market_structure = self.analyze_market_structure(market_data)
            
            # Volume-price relationship analysis
            volume_price_analysis = self.analyze_volume_price_relationship(market_data)
            
            # Support and resistance analysis
            support_resistance = self.analyze_dynamic_support_resistance(market_data)
            
            # Price exhaustion analysis
            exhaustion_signals = self.detect_price_exhaustion_signals(market_data)
            
            # Breakout analysis
            breakout_analysis = self.analyze_breakout_potential(market_data)
            
            # Price action predictions
            price_predictions = self.generate_price_action_predictions(market_data)
            
            # Entry and exit points
            entry_exit_points = self.identify_entry_exit_points(market_data)
            
            return {
                'market_structure': market_structure,
                'volume_price_analysis': volume_price_analysis,
                'support_resistance': support_resistance,
                'exhaustion_signals': exhaustion_signals,
                'breakout_analysis': breakout_analysis,
                'price_predictions': price_predictions,
                'entry_exit_points': entry_exit_points,
                'price_action_strength': self.calculate_price_action_strength(market_data)
            }
            
        except Exception as e:
            print(f"Price action analysis error: {e}")
            return self.create_default_price_action_analysis()
    
    def analyze_market_structure(self, data):
        """
        Analyze market structure including higher highs, lower lows, etc.
        """
        try:
            highs = data['High'].values
            lows = data['Low'].values
            
            # Find significant peaks and troughs
            peak_indices = find_peaks(highs, distance=10, prominence=np.std(highs)*0.5)[0]
            trough_indices = find_peaks(-lows, distance=10, prominence=np.std(lows)*0.5)[0]
            
            # Analyze recent structure (last 10 peaks/troughs)
            recent_peaks = peak_indices[-5:] if len(peak_indices) >= 5 else peak_indices
            recent_troughs = trough_indices[-5:] if len(trough_indices) >= 5 else trough_indices
            
            # Determine trend structure
            if len(recent_peaks) >= 2 and len(recent_troughs) >= 2:
                # Higher highs and higher lows = uptrend
                higher_highs = all(highs[recent_peaks[i]] > highs[recent_peaks[i-1]] 
                                 for i in range(1, len(recent_peaks)))
                higher_lows = all(lows[recent_troughs[i]] > lows[recent_troughs[i-1]] 
                                for i in range(1, len(recent_troughs)))
                
                # Lower highs and lower lows = downtrend
                lower_highs = all(highs[recent_peaks[i]] < highs[recent_peaks[i-1]] 
                                for i in range(1, len(recent_peaks)))
                lower_lows = all(lows[recent_troughs[i]] < lows[recent_troughs[i-1]] 
                               for i in range(1, len(recent_troughs)))
                
                if higher_highs and higher_lows:
                    structure = "Strong Uptrend Structure"
                    trend_strength = 8
                elif lower_highs and lower_lows:
                    structure = "Strong Downtrend Structure"
                    trend_strength = 8
                elif higher_highs or higher_lows:
                    structure = "Weak Uptrend Structure"
                    trend_strength = 6
                elif lower_highs or lower_lows:
                    structure = "Weak Downtrend Structure"
                    trend_strength = 6
                else:
                    structure = "Sideways/Consolidation Structure"
                    trend_strength = 4
            else:
                structure = "Insufficient Data for Structure Analysis"
                trend_strength = 5
            
            # Swing analysis
            swing_analysis = self.analyze_swing_points(data, recent_peaks, recent_troughs)
            
            return {
                'structure_type': structure,
                'trend_strength': trend_strength,
                'recent_peaks': len(recent_peaks),
                'recent_troughs': len(recent_troughs),
                'swing_analysis': swing_analysis,
                'structure_confidence': min(100, len(recent_peaks) * len(recent_troughs) * 10)
            }
            
        except Exception as e:
            print(f"Market structure analysis error: {e}")
            return {
                'structure_type': 'Neutral Structure',
                'trend_strength': 5,
                'recent_peaks': 0,
                'recent_troughs': 0,
                'swing_analysis': {},
                'structure_confidence': 50
            }
    
    def analyze_swing_points(self, data, peaks, troughs):
        """
        Analyze swing points for trend continuation/reversal signals
        """
        try:
            if len(peaks) < 2 or len(troughs) < 2:
                return {'swing_signal': 'Insufficient Data', 'confidence': 0}
            
            current_price = data['Close'].iloc[-1]
            
            # Get last swing high and low
            last_swing_high = data['High'].iloc[peaks[-1]] if len(peaks) > 0 else current_price
            last_swing_low = data['Low'].iloc[troughs[-1]] if len(troughs) > 0 else current_price
            
            # Calculate swing range
            swing_range = abs(last_swing_high - last_swing_low)
            price_position = (current_price - last_swing_low) / swing_range if swing_range > 0 else 0.5
            
            # Swing signals
            if price_position > 0.8:
                swing_signal = "Near Swing High - Potential Reversal"
            elif price_position < 0.2:
                swing_signal = "Near Swing Low - Potential Reversal"
            elif 0.4 <= price_position <= 0.6:
                swing_signal = "Middle of Swing Range - Neutral"
            else:
                swing_signal = "Active Swing Movement"
            
            return {
                'swing_signal': swing_signal,
                'price_position_in_range': price_position,
                'swing_range': swing_range,
                'last_swing_high': last_swing_high,
                'last_swing_low': last_swing_low,
                'confidence': min(100, swing_range / current_price * 1000)
            }
            
        except Exception as e:
            print(f"Swing analysis error: {e}")
            return {'swing_signal': 'Analysis Error', 'confidence': 0}
    
    def analyze_volume_price_relationship(self, data):
        """
        Analyze the relationship between volume and price movements
        """
        try:
            # Calculate price and volume changes
            price_change = data['Close'].pct_change()
            volume_change = data['Volume'].pct_change()
            
            # Price-volume correlation
            correlation = price_change.corr(volume_change)
            
            # Volume trend analysis
            volume_ma_short = data['Volume'].rolling(5).mean()
            volume_ma_long = data['Volume'].rolling(20).mean()
            volume_trend = "Rising" if volume_ma_short.iloc[-1] > volume_ma_long.iloc[-1] else "Falling"
            
            # Accumulation/Distribution analysis
            # Simplified A/D calculation
            money_flow_multiplier = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
            money_flow_multiplier = money_flow_multiplier.fillna(0)
            money_flow_volume = money_flow_multiplier * data['Volume']
            ad_line = money_flow_volume.cumsum()
            
            # A/D trend
            ad_trend = "Accumulation" if ad_line.iloc[-1] > ad_line.iloc[-20] else "Distribution"
            
            # Volume breakouts
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            volume_breakouts = 0
            for i in range(-10, 0):  # Last 10 periods
                if data['Volume'].iloc[i] > avg_volume * 1.5:
                    volume_breakouts += 1
            
            # Volume confirmation analysis
            recent_price_trend = "Up" if data['Close'].iloc[-1] > data['Close'].iloc[-5] else "Down"
            recent_volume_trend = "Up" if volume_ma_short.iloc[-1] > volume_ma_short.iloc[-5] else "Down"
            
            if recent_price_trend == recent_volume_trend:
                volume_confirmation = "Confirmed"
            else:
                volume_confirmation = "Divergence"
            
            # Buying vs Selling pressure
            up_volume = data.loc[price_change > 0, 'Volume'].sum()
            down_volume = data.loc[price_change < 0, 'Volume'].sum()
            total_volume = up_volume + down_volume
            
            buying_pressure = (up_volume / total_volume * 100) if total_volume > 0 else 50
            selling_pressure = (down_volume / total_volume * 100) if total_volume > 0 else 50
            
            return {
                'correlation': correlation,
                'volume_trend': volume_trend,
                'accumulation_distribution': ad_trend,
                'volume_breakouts': volume_breakouts,
                'avg_volume_20d': avg_volume,
                'volume_ratio': volume_ratio,
                'volume_confirmation': volume_confirmation,
                'buying_pressure': buying_pressure,
                'selling_pressure': selling_pressure
            }
            
        except Exception as e:
            print(f"Volume-price analysis error: {e}")
            return {
                'correlation': 0,
                'volume_trend': 'Neutral',
                'accumulation_distribution': 'Neutral',
                'volume_breakouts': 0,
                'avg_volume_20d': 0,
                'volume_ratio': 1,
                'volume_confirmation': 'Neutral',
                'buying_pressure': 50,
                'selling_pressure': 50
            }
    
    def analyze_dynamic_support_resistance(self, data):
        """
        Analyze dynamic support and resistance levels based on price action
        """
        try:
            # Dynamic support/resistance using price clusters
            price_levels = np.concatenate([data['High'].values, data['Low'].values, data['Close'].values])
            
            # Use clustering to find key levels
            price_levels = price_levels.reshape(-1, 1)
            
            # Remove outliers
            q1, q3 = np.percentile(price_levels, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            price_levels_clean = price_levels[(price_levels >= lower_bound) & (price_levels <= upper_bound)]
            
            if len(price_levels_clean) > 10:
                # Cluster key levels
                kmeans = KMeans(n_clusters=min(8, len(price_levels_clean) // 20), random_state=42)
                clusters = kmeans.fit_predict(price_levels_clean)
                key_levels = kmeans.cluster_centers_.flatten()
                
                current_price = data['Close'].iloc[-1]
                
                # Separate support and resistance
                resistance_levels = sorted([level for level in key_levels if level > current_price])
                support_levels = sorted([level for level in key_levels if level < current_price], reverse=True)
                
                # Calculate level strength
                resistance_strength = []
                for level in resistance_levels[:3]:  # Top 3 resistance levels
                    touches = np.sum(np.abs(data['High'].values - level) / level < 0.02)
                    strength = min(100, touches * 20)
                    resistance_strength.append(strength)
                
                support_strength = []
                for level in support_levels[:3]:  # Top 3 support levels
                    touches = np.sum(np.abs(data['Low'].values - level) / level < 0.02)
                    strength = min(100, touches * 20)
                    support_strength.append(strength)
            else:
                current_price = data['Close'].iloc[-1]
                resistance_levels = [current_price * 1.05, current_price * 1.10]
                support_levels = [current_price * 0.95, current_price * 0.90]
                resistance_strength = [50, 40]
                support_strength = [50, 40]
            
            return {
                'resistance_levels': resistance_levels[:3],
                'support_levels': support_levels[:3],
                'resistance_strength': resistance_strength,
                'support_strength': support_strength,
                'nearest_resistance': resistance_levels[0] if resistance_levels else current_price * 1.05,
                'nearest_support': support_levels[0] if support_levels else current_price * 0.95
            }
            
        except Exception as e:
            print(f"Support/Resistance analysis error: {e}")
            current_price = data['Close'].iloc[-1] if not data.empty else 100
            return {
                'resistance_levels': [current_price * 1.05],
                'support_levels': [current_price * 0.95],
                'resistance_strength': [50],
                'support_strength': [50],
                'nearest_resistance': current_price * 1.05,
                'nearest_support': current_price * 0.95
            }
    
    def detect_price_exhaustion_signals(self, data):
        """
        Detect price exhaustion and potential reversal signals
        """
        try:
            exhaustion_signals = {}
            
            # Volume exhaustion
            price_change = data['Close'].pct_change()
            volume_change = data['Volume'].pct_change()
            
            # Check for divergence (price up, volume down or vice versa)
            recent_price_trend = price_change.tail(5).mean()
            recent_volume_trend = volume_change.tail(5).mean()
            
            if recent_price_trend > 0.01 and recent_volume_trend < -0.1:
                volume_exhaustion = "Bullish Exhaustion - Volume Declining"
            elif recent_price_trend < -0.01 and recent_volume_trend < -0.1:
                volume_exhaustion = "Bearish Exhaustion - Volume Declining"
            else:
                volume_exhaustion = "No Volume Exhaustion"
            
            exhaustion_signals['volume_exhaustion'] = volume_exhaustion
            
            # Range exhaustion (narrowing ranges)
            daily_ranges = data['High'] - data['Low']
            avg_range = daily_ranges.rolling(20).mean()
            recent_range = daily_ranges.tail(5).mean()
            range_ratio = recent_range / avg_range.iloc[-1] if avg_range.iloc[-1] > 0 else 1
            
            if range_ratio < 0.6:
                range_exhaustion = "Range Contraction - Breakout Pending"
            elif range_ratio > 1.5:
                range_exhaustion = "Range Expansion - High Volatility"
            else:
                range_exhaustion = "Normal Range Activity"
            
            exhaustion_signals['range_exhaustion'] = range_exhaustion
            
            # Momentum exhaustion
            momentum_3d = (data['Close'].iloc[-1] - data['Close'].iloc[-4]) / data['Close'].iloc[-4]
            momentum_10d = (data['Close'].iloc[-1] - data['Close'].iloc[-11]) / data['Close'].iloc[-11]
            
            if abs(momentum_3d) < abs(momentum_10d) * 0.3:
                momentum_exhaustion = "Momentum Slowing"
            elif abs(momentum_3d) > abs(momentum_10d) * 2:
                momentum_exhaustion = "Momentum Accelerating"
            else:
                momentum_exhaustion = "Normal Momentum"
            
            exhaustion_signals['momentum_exhaustion'] = momentum_exhaustion
            
            # Gap exhaustion
            gaps = data['Open'] - data['Close'].shift(1)
            recent_gaps = gaps.tail(10)
            gap_count = np.sum(np.abs(recent_gaps) > data['Close'].iloc[-1] * 0.02)  # Gaps > 2%
            
            if gap_count > 3:
                gap_exhaustion = "Multiple Gaps - Potential Exhaustion"
            else:
                gap_exhaustion = "Normal Gap Activity"
            
            exhaustion_signals['gap_exhaustion'] = gap_exhaustion
            
            return exhaustion_signals
            
        except Exception as e:
            print(f"Exhaustion signals error: {e}")
            return {
                'volume_exhaustion': 'No Volume Exhaustion',
                'range_exhaustion': 'Normal Range Activity',
                'momentum_exhaustion': 'Normal Momentum',
                'gap_exhaustion': 'Normal Gap Activity'
            }
    
    def analyze_breakout_potential(self, data):
        """
        Analyze potential for price breakouts
        """
        try:
            current_price = data['Close'].iloc[-1]
            
            # Consolidation detection
            high_20 = data['High'].rolling(20).max().iloc[-1]
            low_20 = data['Low'].rolling(20).min().iloc[-1]
            consolidation_range = (high_20 - low_20) / current_price * 100
            
            # Volume buildup during consolidation
            recent_volume = data['Volume'].tail(10).mean()
            older_volume = data['Volume'].iloc[-30:-20].mean()
            volume_buildup = recent_volume / older_volume if older_volume > 0 else 1
            
            # Bollinger Band squeeze (proxy)
            bb_width = consolidation_range  # Simplified
            
            # Breakout probability calculation
            if consolidation_range < 5 and volume_buildup > 1.2:
                breakout_probability = 0.8
                breakout_signal = "High Probability Breakout Setup"
            elif consolidation_range < 8 and volume_buildup > 1.1:
                breakout_probability = 0.6
                breakout_signal = "Moderate Breakout Potential"
            elif consolidation_range > 15:
                breakout_probability = 0.2
                breakout_signal = "Low Breakout Potential - High Volatility"
            else:
                breakout_probability = 0.4
                breakout_signal = "Normal Market Conditions"
            
            # Direction prediction
            if current_price > (high_20 + low_20) / 2:
                breakout_direction = "Upward Breakout More Likely"
            else:
                breakout_direction = "Downward Breakout More Likely"
            
            # Target calculation
            breakout_target = current_price + (high_20 - low_20) if "Upward" in breakout_direction else current_price - (high_20 - low_20)
            
            return {
                'breakout_probability': breakout_probability,
                'breakout_signal': breakout_signal,
                'breakout_direction': breakout_direction,
                'breakout_target': breakout_target,
                'consolidation_range': consolidation_range,
                'volume_buildup': volume_buildup,
                'key_resistance': high_20,
                'key_support': low_20
            }
            
        except Exception as e:
            print(f"Breakout analysis error: {e}")
            current_price = data['Close'].iloc[-1] if not data.empty else 100
            return {
                'breakout_probability': 0.5,
                'breakout_signal': 'Normal Market Conditions',
                'breakout_direction': 'Neutral',
                'breakout_target': current_price,
                'consolidation_range': 10,
                'volume_buildup': 1,
                'key_resistance': current_price * 1.05,
                'key_support': current_price * 0.95
            }
    
    def generate_price_action_predictions(self, data):
        """
        Generate predictions based on pure price action
        """
        try:
            current_price = data['Close'].iloc[-1]
            
            # Market structure bias
            market_structure = self.analyze_market_structure(data)
            structure_bias = market_structure['trend_strength'] / 10
            
            # Support/resistance levels
            sr_levels = self.analyze_dynamic_support_resistance(data)
            
            # Price position relative to key levels
            nearest_resistance = sr_levels['nearest_resistance']
            nearest_support = sr_levels['nearest_support']
            
            price_position = (current_price - nearest_support) / (nearest_resistance - nearest_support) if nearest_resistance != nearest_support else 0.5
            
            # Volume confirmation
            volume_analysis = self.analyze_volume_price_relationship(data)
            volume_factor = 1.2 if volume_analysis['volume_confirmation'] == 'Confirmed' else 0.8
            
            # Prediction logic
            if price_position > 0.8 and "Uptrend" in market_structure['structure_type']:
                direction = "Upward"
                target_price = nearest_resistance * 1.02
                probability = min(85, 60 + structure_bias * 10) * volume_factor
            elif price_position < 0.2 and "Downtrend" in market_structure['structure_type']:
                direction = "Downward"
                target_price = nearest_support * 0.98
                probability = min(85, 60 + structure_bias * 10) * volume_factor
            elif 0.3 <= price_position <= 0.7:
                direction = "Sideways"
                target_price = current_price
                probability = 50
            else:
                direction = "Uncertain"
                target_price = current_price
                probability = 40
            
            confidence_level = "High" if probability > 70 else "Medium" if probability > 50 else "Low"
            
            return {
                'direction': direction,
                'target_price': target_price,
                'probability': probability,
                'confidence_level': confidence_level,
                'time_horizon': '3-7 trading days',
                'key_level_break': nearest_resistance if direction == "Upward" else nearest_support,
                'invalidation_level': nearest_support if direction == "Upward" else nearest_resistance
            }
            
        except Exception as e:
            print(f"Price action predictions error: {e}")
            current_price = data['Close'].iloc[-1] if not data.empty else 100
            return {
                'direction': 'Neutral',
                'target_price': current_price,
                'probability': 50,
                'confidence_level': 'Low',
                'time_horizon': '3-7 trading days',
                'key_level_break': current_price * 1.02,
                'invalidation_level': current_price * 0.98
            }
    
    def identify_entry_exit_points(self, data):
        """
        Identify optimal entry and exit points based on price action
        """
        try:
            current_price = data['Close'].iloc[-1]
            
            # Support/resistance levels for entries
            sr_levels = self.analyze_dynamic_support_resistance(data)
            
            # Market structure for bias
            market_structure = self.analyze_market_structure(data)
            
            # Entry points
            if "Uptrend" in market_structure['structure_type']:
                entry_points = {
                    'long_entry': sr_levels['nearest_support'],
                    'long_entry_aggressive': current_price * 1.005,  # Market buy
                    'short_entry': sr_levels['nearest_resistance'],
                    'entry_bias': 'Long Bias'
                }
            elif "Downtrend" in market_structure['structure_type']:
                entry_points = {
                    'long_entry': sr_levels['nearest_support'],
                    'short_entry': sr_levels['nearest_resistance'],
                    'short_entry_aggressive': current_price * 0.995,  # Market sell
                    'entry_bias': 'Short Bias'
                }
            else:
                entry_points = {
                    'long_entry': sr_levels['nearest_support'],
                    'short_entry': sr_levels['nearest_resistance'],
                    'entry_bias': 'Range Trading'
                }
            
            # Exit points
            atr = (data['High'] - data['Low']).tail(14).mean()  # 14-day ATR approximation
            
            exit_points = {
                'profit_target_long': sr_levels['nearest_resistance'],
                'profit_target_short': sr_levels['nearest_support'],
                'stop_loss_long': current_price - atr * 2,
                'stop_loss_short': current_price + atr * 2,
                'trailing_stop_distance': atr * 1.5
            }
            
            # Risk-reward ratios
            if 'long_entry' in entry_points:
                long_risk = abs(entry_points['long_entry'] - exit_points['stop_loss_long'])
                long_reward = abs(exit_points['profit_target_long'] - entry_points['long_entry'])
                long_rr = long_reward / long_risk if long_risk > 0 else 1
            else:
                long_rr = 1
            
            if 'short_entry' in entry_points:
                short_risk = abs(exit_points['stop_loss_short'] - entry_points['short_entry'])
                short_reward = abs(entry_points['short_entry'] - exit_points['profit_target_short'])
                short_rr = short_reward / short_risk if short_risk > 0 else 1
            else:
                short_rr = 1
            
            return {
                'entry_points': entry_points,
                'exit_points': exit_points,
                'risk_reward_ratios': {
                    'long_rr': long_rr,
                    'short_rr': short_rr
                },
                'recommended_position_size': self.calculate_position_size(atr, current_price),
                'market_condition': market_structure['structure_type']
            }
            
        except Exception as e:
            print(f"Entry/Exit points error: {e}")
            current_price = data['Close'].iloc[-1] if not data.empty else 100
            return {
                'entry_points': {
                    'long_entry': current_price * 0.99,
                    'short_entry': current_price * 1.01,
                    'entry_bias': 'Neutral'
                },
                'exit_points': {
                    'profit_target_long': current_price * 1.05,
                    'profit_target_short': current_price * 0.95,
                    'stop_loss_long': current_price * 0.97,
                    'stop_loss_short': current_price * 1.03
                },
                'risk_reward_ratios': {'long_rr': 1.5, 'short_rr': 1.5},
                'recommended_position_size': 'Medium',
                'market_condition': 'Neutral'
            }
    
    def calculate_position_size(self, atr, current_price):
        """
        Calculate recommended position size based on volatility
        """
        try:
            volatility_ratio = atr / current_price
            
            if volatility_ratio > 0.05:  # High volatility
                return "Small (1-2% of portfolio)"
            elif volatility_ratio > 0.03:  # Medium volatility
                return "Medium (3-5% of portfolio)"
            else:  # Low volatility
                return "Large (5-8% of portfolio)"
                
        except:
            return "Medium (3-5% of portfolio)"
    
    def calculate_price_action_strength(self, data):
        """
        Calculate overall price action strength score
        """
        try:
            # Volume strength
            volume_analysis = self.analyze_volume_price_relationship(data)
            volume_score = 1 if volume_analysis['volume_confirmation'] == 'Confirmed' else 0.5
            
            # Structure strength
            market_structure = self.analyze_market_structure(data)
            structure_score = market_structure['trend_strength'] / 10
            
            # Breakout potential
            breakout_analysis = self.analyze_breakout_potential(data)
            breakout_score = breakout_analysis['breakout_probability']
            
            # Combined strength
            overall_strength = (volume_score + structure_score + breakout_score) / 3 * 100
            
            if overall_strength > 75:
                strength_level = "Very Strong"
            elif overall_strength > 60:
                strength_level = "Strong"
            elif overall_strength > 40:
                strength_level = "Moderate"
            else:
                strength_level = "Weak"
            
            return {
                'overall_score': overall_strength,
                'strength_level': strength_level,
                'volume_component': volume_score * 100,
                'structure_component': structure_score * 100,
                'breakout_component': breakout_score * 100
            }
            
        except Exception as e:
            print(f"Price action strength calculation error: {e}")
            return {
                'overall_score': 50,
                'strength_level': 'Moderate',
                'volume_component': 50,
                'structure_component': 50,
                'breakout_component': 50
            }
    
    def create_default_price_action_analysis(self):
        """
        Create default analysis when calculations fail
        """
        return {
            'market_structure': {
                'structure_type': 'Neutral Structure',
                'trend_strength': 5,
                'structure_confidence': 50
            },
            'volume_price_analysis': {
                'correlation': 0,
                'volume_trend': 'Neutral',
                'accumulation_distribution': 'Neutral',
                'volume_confirmation': 'Neutral',
                'buying_pressure': 50,
                'selling_pressure': 50
            },
            'support_resistance': {
                'resistance_levels': [105],
                'support_levels': [95],
                'nearest_resistance': 105,
                'nearest_support': 95
            },
            'exhaustion_signals': {
                'volume_exhaustion': 'No Volume Exhaustion',
                'range_exhaustion': 'Normal Range Activity',
                'momentum_exhaustion': 'Normal Momentum'
            },
            'breakout_analysis': {
                'breakout_probability': 0.5,
                'breakout_signal': 'Normal Market Conditions',
                'breakout_direction': 'Neutral'
            },
            'price_predictions': {
                'direction': 'Neutral',
                'target_price': 100,
                'probability': 50,
                'confidence_level': 'Low'
            },
            'entry_exit_points': {
                'entry_points': {'entry_bias': 'Neutral'},
                'exit_points': {},
                'risk_reward_ratios': {'long_rr': 1.5, 'short_rr': 1.5}
            },
            'price_action_strength': {
                'overall_score': 50,
                'strength_level': 'Moderate'
            }
        }
