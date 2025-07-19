import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Use finta for technical indicators (best compatibility)
try:
    from finta import TA as finta_ta
    FINTA_AVAILABLE = True
    print("✓ Using finta for technical indicators - 80+ indicators available")
except ImportError:
    FINTA_AVAILABLE = False
    print("finta not available, using manual implementations")

# Use ta library as secondary option
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

# Optional talib import for fallback
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

class TechnicalIndicatorEngine:
    """
    Advanced Technical Indicator Engine with 40+ indicators and pattern analysis
    Focuses on pattern detection within indicators, not just values
    """
    
    def __init__(self):
        self.indicators = {}
        self.patterns = {}
        self.signals = {}
        
    def calculate_all_indicators(self, data):
        """Calculate all 40+ technical indicators with pattern analysis"""
        df = data.copy()
        results = {}
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Extract OHLCV data
        open_price = df['Open'].values
        high_price = df['High'].values
        low_price = df['Low'].values
        close_price = df['Close'].values
        volume = df['Volume'].values
        
        # 1. BOLLINGER BANDS with advanced analysis
        results['Bollinger_Bands'] = self.analyze_bollinger_bands(close_price)
        
        # 2. VWAP with reversion analysis
        results['VWAP'] = self.analyze_vwap(high_price, low_price, close_price, volume)
        
        # 3. RSI with divergence analysis
        results['RSI'] = self.analyze_rsi(close_price)
        
        # 4. MACD with histogram pattern analysis
        results['MACD'] = self.analyze_macd(close_price)
        
        # 5. STOCHASTIC with %K/%D relationships
        results['Stochastic'] = self.analyze_stochastic(high_price, low_price, close_price)
        
        # 6. VOLUME INDICATORS
        results['Volume_OBV'] = self.analyze_obv(close_price, volume)
        results['Volume_AD'] = self.analyze_accumulation_distribution(high_price, low_price, close_price, volume)
        results['Volume_CMF'] = self.analyze_cmf(high_price, low_price, close_price, volume)
        
        # 7. MOVING AVERAGES with dynamic analysis
        results['SMA_Analysis'] = self.analyze_moving_averages(close_price, 'SMA')
        results['EMA_Analysis'] = self.analyze_moving_averages(close_price, 'EMA')
        results['WMA_Analysis'] = self.analyze_moving_averages(close_price, 'WMA')
        
        # 8. VOLATILITY INDICATORS
        results['ATR'] = self.analyze_atr(high_price, low_price, close_price)
        results['Volatility_Bands'] = self.analyze_volatility_bands(close_price)
        
        # 9. MOMENTUM INDICATORS
        results['Williams_R'] = self.analyze_williams_r(high_price, low_price, close_price)
        results['CCI'] = self.analyze_cci(high_price, low_price, close_price)
        results['ROC'] = self.analyze_roc(close_price)
        results['MFI'] = self.analyze_mfi(high_price, low_price, close_price, volume)
        
        # 10. TREND INDICATORS
        results['ADX'] = self.analyze_adx(high_price, low_price, close_price)
        results['Parabolic_SAR'] = self.analyze_parabolic_sar(high_price, low_price)
        results['Aroon'] = self.analyze_aroon(high_price, low_price)
        
        # 11. ICHIMOKU CLOUD
        results['Ichimoku'] = self.analyze_ichimoku(high_price, low_price, close_price)
        
        # 12. FIBONACCI LEVELS
        results['Fibonacci'] = self.analyze_fibonacci_levels(high_price, low_price)
        
        # 13. PIVOT POINTS
        results['Pivot_Points'] = self.analyze_pivot_points(high_price, low_price, close_price)
        
        # 14. CUSTOM COMPOSITE INDICATORS
        results['Composite_Momentum'] = self.analyze_composite_momentum(close_price, volume)
        results['Composite_Trend'] = self.analyze_composite_trend(high_price, low_price, close_price)
        results['Composite_Volatility'] = self.analyze_composite_volatility(high_price, low_price, close_price)
        
        # 15. ADDITIONAL ADVANCED INDICATORS
        results['Donchian_Channels'] = self.analyze_donchian_channels(high_price, low_price)
        results['Keltner_Channels'] = self.analyze_keltner_channels(high_price, low_price, close_price)
        results['TRIX'] = self.analyze_trix(close_price)
        results['Elder_Ray'] = self.analyze_elder_ray(high_price, low_price, close_price)
        results['Chande_Momentum'] = self.analyze_chande_momentum(close_price)
        results['Ultimate_Oscillator'] = self.analyze_ultimate_oscillator(high_price, low_price, close_price)
        results['Detrended_Price'] = self.analyze_detrended_price_oscillator(close_price)
        
        return results
    
    def analyze_bollinger_bands(self, close_price, period=20, std_dev=2):
        """Advanced Bollinger Bands analysis with squeeze and expansion detection"""
        if len(close_price) < period:
            return self.create_default_indicator_result("Bollinger Bands", 0, "Insufficient Data", "None", 0)
        
        try:
            # Calculate Bollinger Bands with finta (preferred) or fallback
            if FINTA_AVAILABLE:
                df_temp = pd.DataFrame({'open': close_price, 'high': close_price, 'low': close_price, 'close': close_price})
                bb_result = finta_ta.BBANDS(df_temp, period=period, std_multiplier=std_dev)
                upper = bb_result['BB_UPPER'].values
                middle = bb_result['BB_MIDDLE'].values
                lower = bb_result['BB_LOWER'].values
            elif TA_AVAILABLE:
                df_temp = pd.DataFrame({'close': close_price})
                upper = ta.volatility.bollinger_hband(df_temp['close'], window=period, window_dev=std_dev).values
                middle = ta.volatility.bollinger_mavg(df_temp['close'], window=period).values
                lower = ta.volatility.bollinger_lband(df_temp['close'], window=period, window_dev=std_dev).values
            elif TALIB_AVAILABLE:
                upper, middle, lower = talib.BBANDS(close_price, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
            else:
                # Manual calculation
                middle = self.sma(close_price, period)
                std = self.rolling_std(close_price, period)
                upper = middle + (std * std_dev)
                lower = middle - (std * std_dev)
            
            # Current values
            current_price = close_price[-1]
            current_upper = upper[-1]
            current_middle = middle[-1]
            current_lower = lower[-1]
            
            # Band width analysis
            band_width = (current_upper - current_lower) / current_middle
            bb_position = (current_price - current_lower) / (current_upper - current_lower)
            
            # Squeeze detection (narrow bands)
            band_widths = (upper - lower) / middle
            avg_band_width = np.nanmean(band_widths[-50:]) if len(band_widths) >= 50 else np.nanmean(band_widths)
            squeeze_threshold = avg_band_width * 0.7
            is_squeeze = band_width < squeeze_threshold
            
            # Band walking analysis
            band_walk_periods = 0
            for i in range(min(10, len(close_price) - 1), 0, -1):
                if close_price[-i] > upper[-i] * 0.98:  # Near upper band
                    band_walk_periods += 1
                else:
                    break
            
            # Pattern detection
            if is_squeeze:
                pattern = "Bollinger Squeeze - Breakout Imminent"
                signal = "Wait for Breakout"
            elif bb_position > 0.8:
                pattern = "Upper Band Walking - Strong Uptrend"
                signal = "Strong Buy" if band_walk_periods >= 3 else "Buy"
            elif bb_position < 0.2:
                pattern = "Lower Band Walking - Strong Downtrend" 
                signal = "Strong Sell" if band_walk_periods >= 3 else "Sell"
            elif 0.4 <= bb_position <= 0.6:
                pattern = "Middle Band Mean Reversion"
                signal = "Neutral"
            else:
                pattern = "Band Expansion Phase"
                signal = "Trending"
            
            confidence = min(95, max(50, 70 + abs(bb_position - 0.5) * 50))
            
            return {
                'current_value': bb_position,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence,
                'squeeze_detected': is_squeeze,
                'band_walk_periods': band_walk_periods,
                'band_width': band_width,
                'levels': {'upper': current_upper, 'middle': current_middle, 'lower': current_lower}
            }
            
        except Exception as e:
            return self.create_default_indicator_result("Bollinger Bands", 0, "Calculation Error", "None", 0)
    
    def analyze_vwap(self, high_price, low_price, close_price, volume):
        """Advanced VWAP analysis with flat significance and reversion patterns"""
        try:
            # Calculate VWAP
            typical_price = (high_price + low_price + close_price) / 3
            vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
            
            current_price = close_price[-1]
            current_vwap = vwap[-1]
            
            # VWAP deviation analysis
            vwap_deviation = (current_price - current_vwap) / current_vwap * 100
            
            # Flat VWAP detection (low slope)
            if len(vwap) >= 20:
                recent_vwap_slope = (vwap[-1] - vwap[-20]) / vwap[-20] * 100
                is_flat_vwap = abs(recent_vwap_slope) < 0.5
            else:
                recent_vwap_slope = 0
                is_flat_vwap = False
            
            # Reversion pattern analysis
            above_vwap_periods = 0
            below_vwap_periods = 0
            
            for i in range(min(10, len(close_price)), 0, -1):
                if close_price[-i] > vwap[-i]:
                    above_vwap_periods += 1
                elif close_price[-i] < vwap[-i]:
                    below_vwap_periods += 1
            
            # Pattern detection
            if is_flat_vwap:
                if abs(vwap_deviation) > 2:
                    pattern = "Flat VWAP Reversion Setup"
                    signal = "Sell" if vwap_deviation > 0 else "Buy"
                else:
                    pattern = "Flat VWAP Consolidation"
                    signal = "Neutral"
            elif vwap_deviation > 3:
                pattern = "Extended Above VWAP"
                signal = "Take Profit" if above_vwap_periods > 5 else "Hold"
            elif vwap_deviation < -3:
                pattern = "Extended Below VWAP" 
                signal = "Oversold Bounce" if below_vwap_periods > 5 else "Hold"
            else:
                pattern = "Normal VWAP Range"
                signal = "Buy" if vwap_deviation < 0 else "Sell"
            
            confidence = min(90, max(60, 70 + abs(vwap_deviation) * 5))
            
            return {
                'current_value': vwap_deviation,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence,
                'flat_vwap': is_flat_vwap,
                'slope': recent_vwap_slope,
                'above_periods': above_vwap_periods,
                'below_periods': below_vwap_periods
            }
            
        except Exception as e:
            return self.create_default_indicator_result("VWAP", 0, "Calculation Error", "None", 0)
    
    def analyze_rsi(self, close_price, period=14):
        """Advanced RSI analysis with divergence detection and trend analysis"""
        if len(close_price) < period + 1:
            return self.create_default_indicator_result("RSI", 50, "Insufficient Data", "None", 0)
        
        try:
            # Calculate RSI
            rsi = talib.RSI(close_price, timeperiod=period)
            current_rsi = rsi[-1]
            
            # RSI trend analysis
            rsi_trend = "Neutral"
            if len(rsi) >= 5:
                recent_rsi_slope = np.polyfit(range(5), rsi[-5:], 1)[0]
                if recent_rsi_slope > 2:
                    rsi_trend = "Rising"
                elif recent_rsi_slope < -2:
                    rsi_trend = "Falling"
            
            # Divergence analysis
            divergence = self.detect_rsi_divergence(close_price, rsi)
            
            # RSI cluster formation (consolidation)
            if len(rsi) >= 10:
                recent_rsi = rsi[-10:]
                rsi_volatility = np.std(recent_rsi)
                is_clustering = rsi_volatility < 5
            else:
                is_clustering = False
            
            # Pattern detection
            if current_rsi > 80:
                if divergence == "Bearish":
                    pattern = "RSI Bearish Divergence - Overbought"
                    signal = "Strong Sell"
                else:
                    pattern = "RSI Overbought"
                    signal = "Sell"
            elif current_rsi < 20:
                if divergence == "Bullish":
                    pattern = "RSI Bullish Divergence - Oversold"
                    signal = "Strong Buy"
                else:
                    pattern = "RSI Oversold"
                    signal = "Buy"
            elif 40 <= current_rsi <= 60 and is_clustering:
                pattern = "RSI Cluster Formation - Breakout Pending"
                signal = "Wait"
            elif current_rsi > 70:
                pattern = f"RSI {rsi_trend} in Overbought Zone"
                signal = "Caution - Sell"
            elif current_rsi < 30:
                pattern = f"RSI {rsi_trend} in Oversold Zone"
                signal = "Caution - Buy"
            else:
                pattern = f"RSI {rsi_trend} in Normal Range"
                signal = "Neutral"
            
            confidence = min(90, max(50, abs(current_rsi - 50) * 1.5 + 50))
            
            return {
                'current_value': current_rsi,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence,
                'trend': rsi_trend,
                'divergence': divergence,
                'clustering': is_clustering
            }
            
        except Exception as e:
            return self.create_default_indicator_result("RSI", 50, "Calculation Error", "None", 0)
    
    def detect_rsi_divergence(self, price, rsi, lookback=20):
        """Detect bullish/bearish divergence in RSI"""
        if len(price) < lookback or len(rsi) < lookback:
            return "None"
        
        try:
            # Find recent peaks and troughs
            price_peaks = find_peaks(price[-lookback:], distance=5)[0]
            price_troughs = find_peaks(-price[-lookback:], distance=5)[0]
            rsi_peaks = find_peaks(rsi[-lookback:], distance=5)[0]
            rsi_troughs = find_peaks(-rsi[-lookback:], distance=5)[0]
            
            # Check for bearish divergence (price higher highs, RSI lower highs)
            if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
                if (price[-lookback:][price_peaks[-1]] > price[-lookback:][price_peaks[-2]] and
                    rsi[-lookback:][rsi_peaks[-1]] < rsi[-lookback:][rsi_peaks[-2]]):
                    return "Bearish"
            
            # Check for bullish divergence (price lower lows, RSI higher lows)
            if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
                if (price[-lookback:][price_troughs[-1]] < price[-lookback:][price_troughs[-2]] and
                    rsi[-lookback:][rsi_troughs[-1]] > rsi[-lookback:][rsi_troughs[-2]]):
                    return "Bullish"
            
            return "None"
            
        except Exception:
            return "None"
    
    def analyze_macd(self, close_price, fast=12, slow=26, signal=9):
        """Advanced MACD analysis with histogram pattern detection"""
        if len(close_price) < slow + signal:
            return self.create_default_indicator_result("MACD", 0, "Insufficient Data", "None", 0)
        
        try:
            # Calculate MACD
            macd, signal_line, histogram = talib.MACD(close_price, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            
            current_macd = macd[-1]
            current_signal = signal_line[-1]
            current_histogram = histogram[-1]
            
            # Histogram pattern analysis
            histogram_trend = "Neutral"
            if len(histogram) >= 5:
                recent_hist_slope = np.polyfit(range(5), histogram[-5:], 1)[0]
                if recent_hist_slope > 0.001:
                    histogram_trend = "Expanding"
                elif recent_hist_slope < -0.001:
                    histogram_trend = "Contracting"
            
            # Momentum shift detection
            momentum_shift = "None"
            if len(histogram) >= 3:
                if histogram[-3] < 0 and histogram[-2] < 0 and histogram[-1] > 0:
                    momentum_shift = "Bullish Momentum Shift"
                elif histogram[-3] > 0 and histogram[-2] > 0 and histogram[-1] < 0:
                    momentum_shift = "Bearish Momentum Shift"
            
            # Signal crossover analysis
            crossover = "None"
            if len(macd) >= 2 and len(signal_line) >= 2:
                if macd[-2] <= signal_line[-2] and macd[-1] > signal_line[-1]:
                    crossover = "Bullish Crossover"
                elif macd[-2] >= signal_line[-2] and macd[-1] < signal_line[-1]:
                    crossover = "Bearish Crossover"
            
            # Pattern detection
            if crossover == "Bullish Crossover":
                pattern = f"MACD Bullish Crossover - Histogram {histogram_trend}"
                signal_result = "Buy"
            elif crossover == "Bearish Crossover":
                pattern = f"MACD Bearish Crossover - Histogram {histogram_trend}"
                signal_result = "Sell"
            elif momentum_shift != "None":
                pattern = f"MACD {momentum_shift}"
                signal_result = "Buy" if "Bullish" in momentum_shift else "Sell"
            elif current_macd > current_signal and histogram_trend == "Expanding":
                pattern = "MACD Above Signal - Expanding Histogram"
                signal_result = "Strong Buy"
            elif current_macd < current_signal and histogram_trend == "Expanding":
                pattern = "MACD Below Signal - Expanding Histogram" 
                signal_result = "Strong Sell"
            else:
                pattern = f"MACD {histogram_trend} Phase"
                signal_result = "Neutral"
            
            confidence = min(85, max(60, abs(current_histogram) * 1000 + 60))
            
            return {
                'current_value': current_macd - current_signal,
                'signal': signal_result,
                'pattern_detected': pattern,
                'confidence': confidence,
                'histogram_trend': histogram_trend,
                'momentum_shift': momentum_shift,
                'crossover': crossover
            }
            
        except Exception as e:
            return self.create_default_indicator_result("MACD", 0, "Calculation Error", "None", 0)
    
    def analyze_stochastic(self, high_price, low_price, close_price, k_period=14, d_period=3):
        """Advanced Stochastic analysis with %K/%D relationships"""
        if len(close_price) < k_period + d_period:
            return self.create_default_indicator_result("Stochastic", 50, "Insufficient Data", "None", 0)
        
        try:
            # Calculate Stochastic
            slowk, slowd = talib.STOCH(high_price, low_price, close_price, 
                                     fastk_period=k_period, slowk_period=3, slowd_period=d_period)
            
            current_k = slowk[-1]
            current_d = slowd[-1]
            
            # %K/%D relationship analysis
            kd_difference = current_k - current_d
            
            # Divergence analysis
            divergence = "None"
            if len(slowk) >= 10 and len(close_price) >= 10:
                # Simple divergence check
                if (close_price[-1] > close_price[-5] and slowk[-1] < slowk[-5] and current_k > 80):
                    divergence = "Bearish"
                elif (close_price[-1] < close_price[-5] and slowk[-1] > slowk[-5] and current_k < 20):
                    divergence = "Bullish"
            
            # Crossover detection
            crossover = "None"
            if len(slowk) >= 2 and len(slowd) >= 2:
                if slowk[-2] <= slowd[-2] and slowk[-1] > slowd[-1]:
                    crossover = "Bullish K/D Crossover"
                elif slowk[-2] >= slowd[-2] and slowk[-1] < slowd[-1]:
                    crossover = "Bearish K/D Crossover"
            
            # Pattern detection
            if current_k > 80 and current_d > 80:
                if divergence == "Bearish":
                    pattern = "Stochastic Bearish Divergence - Overbought"
                    signal = "Strong Sell"
                else:
                    pattern = "Stochastic Overbought Zone"
                    signal = "Sell"
            elif current_k < 20 and current_d < 20:
                if divergence == "Bullish":
                    pattern = "Stochastic Bullish Divergence - Oversold"
                    signal = "Strong Buy"
                else:
                    pattern = "Stochastic Oversold Zone"
                    signal = "Buy"
            elif crossover != "None":
                pattern = f"Stochastic {crossover}"
                signal = "Buy" if "Bullish" in crossover else "Sell"
            elif abs(kd_difference) < 2:
                pattern = "Stochastic %K/%D Convergence"
                signal = "Wait"
            else:
                pattern = "Stochastic Normal Range"
                signal = "Neutral"
            
            confidence = min(80, max(55, abs(current_k - 50) * 1.2 + 55))
            
            return {
                'current_value': current_k,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence,
                'k_value': current_k,
                'd_value': current_d,
                'kd_difference': kd_difference,
                'divergence': divergence,
                'crossover': crossover
            }
            
        except Exception as e:
            return self.create_default_indicator_result("Stochastic", 50, "Calculation Error", "None", 0)
    
    def analyze_obv(self, close_price, volume):
        """On Balance Volume analysis with trend patterns"""
        try:
            # Calculate OBV
            obv = talib.OBV(close_price, volume)
            
            # OBV trend analysis
            if len(obv) >= 20:
                obv_slope = np.polyfit(range(20), obv[-20:], 1)[0]
                price_slope = np.polyfit(range(20), close_price[-20:], 1)[0]
                
                # Normalize slopes for comparison
                obv_slope_norm = obv_slope / np.mean(obv[-20:]) if np.mean(obv[-20:]) != 0 else 0
                price_slope_norm = price_slope / np.mean(close_price[-20:])
                
                # Divergence detection
                if price_slope_norm > 0 and obv_slope_norm < 0:
                    pattern = "OBV Bearish Divergence"
                    signal = "Sell"
                elif price_slope_norm < 0 and obv_slope_norm > 0:
                    pattern = "OBV Bullish Divergence"
                    signal = "Buy"
                elif obv_slope_norm > 0.001:
                    pattern = "OBV Uptrend - Accumulation"
                    signal = "Buy"
                elif obv_slope_norm < -0.001:
                    pattern = "OBV Downtrend - Distribution"
                    signal = "Sell"
                else:
                    pattern = "OBV Sideways"
                    signal = "Neutral"
            else:
                pattern = "OBV Insufficient Data"
                signal = "Neutral"
                obv_slope_norm = 0
            
            confidence = min(75, max(50, abs(obv_slope_norm) * 10000 + 50))
            
            return {
                'current_value': obv[-1],
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence,
                'trend_slope': obv_slope_norm
            }
            
        except Exception as e:
            return self.create_default_indicator_result("OBV", 0, "Calculation Error", "None", 0)
    
    def analyze_accumulation_distribution(self, high_price, low_price, close_price, volume):
        """Accumulation/Distribution Line analysis"""
        try:
            # Calculate A/D Line
            ad_line = talib.AD(high_price, low_price, close_price, volume)
            
            # Trend analysis
            if len(ad_line) >= 20:
                ad_slope = np.polyfit(range(20), ad_line[-20:], 1)[0]
                price_slope = np.polyfit(range(20), close_price[-20:], 1)[0]
                
                # Normalize slopes
                ad_slope_norm = ad_slope / np.mean(abs(ad_line[-20:])) if np.mean(abs(ad_line[-20:])) != 0 else 0
                price_slope_norm = price_slope / np.mean(close_price[-20:])
                
                if ad_slope_norm > 0.001:
                    pattern = "Accumulation Phase"
                    signal = "Buy"
                elif ad_slope_norm < -0.001:
                    pattern = "Distribution Phase"
                    signal = "Sell"
                else:
                    pattern = "Neutral A/D"
                    signal = "Neutral"
            else:
                pattern = "A/D Insufficient Data"
                signal = "Neutral"
                ad_slope_norm = 0
            
            confidence = min(70, max(50, abs(ad_slope_norm) * 5000 + 50))
            
            return {
                'current_value': ad_line[-1],
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence
            }
            
        except Exception as e:
            return self.create_default_indicator_result("A/D Line", 0, "Calculation Error", "None", 0)
    
    def analyze_cmf(self, high_price, low_price, close_price, volume, period=20):
        """Chaikin Money Flow analysis"""
        if len(close_price) < period:
            return self.create_default_indicator_result("CMF", 0, "Insufficient Data", "None", 0)
        
        try:
            # Calculate CMF manually
            mfm = ((close_price - low_price) - (high_price - close_price)) / (high_price - low_price)
            mfm = np.where(high_price == low_price, 0, mfm)  # Handle division by zero
            mfv = mfm * volume
            
            cmf = []
            for i in range(period - 1, len(mfv)):
                cmf_value = np.sum(mfv[i-period+1:i+1]) / np.sum(volume[i-period+1:i+1])
                cmf.append(cmf_value)
            
            current_cmf = cmf[-1] if cmf else 0
            
            # Pattern analysis
            if current_cmf > 0.1:
                pattern = "Strong Money Flow Inflow"
                signal = "Buy"
            elif current_cmf < -0.1:
                pattern = "Strong Money Flow Outflow"
                signal = "Sell"
            elif current_cmf > 0:
                pattern = "Weak Money Flow Inflow"
                signal = "Weak Buy"
            elif current_cmf < 0:
                pattern = "Weak Money Flow Outflow"
                signal = "Weak Sell"
            else:
                pattern = "Neutral Money Flow"
                signal = "Neutral"
            
            confidence = min(75, max(50, abs(current_cmf) * 500 + 50))
            
            return {
                'current_value': current_cmf,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence
            }
            
        except Exception as e:
            return self.create_default_indicator_result("CMF", 0, "Calculation Error", "None", 0)
    
    def analyze_moving_averages(self, close_price, ma_type='SMA'):
        """Advanced moving average analysis with dynamic support/resistance"""
        try:
            # Calculate multiple timeframe MAs
            # Calculate moving averages manually since talib is not available
            close_series = pd.Series(close_price)
            if ma_type == 'SMA':
                ma_5 = close_series.rolling(window=5).mean().values
                ma_20 = close_series.rolling(window=20).mean().values
                ma_50 = close_series.rolling(window=50).mean().values
                ma_200 = close_series.rolling(window=200).mean().values
            elif ma_type == 'EMA':
                ma_5 = close_series.ewm(span=5).mean().values
                ma_20 = close_series.ewm(span=20).mean().values
                ma_50 = close_series.ewm(span=50).mean().values
                ma_200 = close_series.ewm(span=200).mean().values
            else:  # WMA - use exponential as approximation
                ma_5 = close_series.ewm(span=5).mean().values
                ma_20 = close_series.ewm(span=20).mean().values
                ma_50 = close_series.ewm(span=50).mean().values
                ma_200 = close_series.ewm(span=200).mean().values
            
            current_price = close_price[-1]
            
            # Golden/Death Cross detection
            cross_pattern = "None"
            if len(ma_20) >= 2 and len(ma_50) >= 2:
                if ma_20[-2] <= ma_50[-2] and ma_20[-1] > ma_50[-1]:
                    cross_pattern = "Golden Cross (20/50)"
                elif ma_20[-2] >= ma_50[-2] and ma_20[-1] < ma_50[-1]:
                    cross_pattern = "Death Cross (20/50)"
            
            # MA Cloud analysis (all MAs alignment)
            mas_current = [ma_5[-1], ma_20[-1], ma_50[-1], ma_200[-1]]
            mas_current = [ma for ma in mas_current if not np.isnan(ma)]
            
            if len(mas_current) >= 3:
                if all(mas_current[i] > mas_current[i+1] for i in range(len(mas_current)-1)):
                    cloud_pattern = "Bullish MA Cloud"
                    signal = "Strong Buy"
                elif all(mas_current[i] < mas_current[i+1] for i in range(len(mas_current)-1)):
                    cloud_pattern = "Bearish MA Cloud"
                    signal = "Strong Sell"
                else:
                    cloud_pattern = "Mixed MA Cloud"
                    signal = "Neutral"
            else:
                cloud_pattern = "Insufficient MA Data"
                signal = "Neutral"
            
            # Dynamic support/resistance levels
            support_level = min([ma for ma in mas_current if ma < current_price], default=current_price)
            resistance_level = max([ma for ma in mas_current if ma > current_price], default=current_price)
            
            # Overall pattern
            if cross_pattern != "None":
                pattern = f"{ma_type} {cross_pattern} - {cloud_pattern}"
            else:
                pattern = f"{ma_type} {cloud_pattern}"
            
            confidence = min(80, max(60, len(mas_current) * 15 + 45))
            
            return {
                'current_value': (current_price - ma_20[-1]) / ma_20[-1] * 100 if not np.isnan(ma_20[-1]) else 0,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence,
                'cross_pattern': cross_pattern,
                'cloud_pattern': cloud_pattern,
                'support_level': support_level,
                'resistance_level': resistance_level
            }
            
        except Exception as e:
            return self.create_default_indicator_result(f"{ma_type}", 0, "Calculation Error", "None", 0)
    
    def analyze_atr(self, high_price, low_price, close_price, period=14):
        """ATR Volatility analysis with expansion/contraction patterns"""
        if len(close_price) < period + 1:
            return self.create_default_indicator_result("ATR", 0, "Insufficient Data", "None", 0)
        
        try:
            # Calculate ATR manually since talib is not available
            high_low = pd.Series(high_price) - pd.Series(low_price)
            high_close = abs(pd.Series(high_price) - pd.Series(close_price).shift(1))
            low_close = abs(pd.Series(low_price) - pd.Series(close_price).shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean().values
            current_atr = atr[-1] if len(atr) > 0 else 0
            
            # Volatility expansion/contraction analysis
            if len(atr) >= 20:
                avg_atr = np.mean(atr[-20:])
                atr_ratio = current_atr / avg_atr
                
                # Trend in volatility
                atr_slope = np.polyfit(range(10), atr[-10:], 1)[0] if len(atr) >= 10 else 0
                
                if atr_ratio > 1.5:
                    pattern = "Volatility Expansion - High ATR"
                    signal = "High Volatility"
                elif atr_ratio < 0.7:
                    pattern = "Volatility Contraction - Low ATR"
                    signal = "Low Volatility"
                elif atr_slope > 0:
                    pattern = "Rising Volatility"
                    signal = "Increasing Volatility"
                elif atr_slope < 0:
                    pattern = "Falling Volatility"
                    signal = "Decreasing Volatility"
                else:
                    pattern = "Stable Volatility"
                    signal = "Normal Volatility"
            else:
                pattern = "ATR Baseline"
                signal = "Normal Volatility"
                atr_ratio = 1
            
            confidence = min(75, max(55, abs(atr_ratio - 1) * 30 + 55))
            
            return {
                'current_value': current_atr,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence,
                'volatility_ratio': atr_ratio if 'atr_ratio' in locals() else 1
            }
            
        except Exception as e:
            return self.create_default_indicator_result("ATR", 0, "Calculation Error", "None", 0)
    
    def analyze_volatility_bands(self, close_price, period=20, factor=2):
        """Custom volatility bands analysis"""
        try:
            # Calculate volatility bands (similar to Bollinger but using ATR)
            sma = talib.SMA(close_price, timeperiod=period)
            
            # Calculate rolling standard deviation manually for volatility
            rolling_std = pd.Series(close_price).rolling(period).std().values
            
            upper_band = sma + (rolling_std * factor)
            lower_band = sma - (rolling_std * factor)
            
            current_price = close_price[-1]
            band_position = (current_price - lower_band[-1]) / (upper_band[-1] - lower_band[-1])
            
            if band_position > 0.8:
                pattern = "Near Upper Volatility Band"
                signal = "Overbought"
            elif band_position < 0.2:
                pattern = "Near Lower Volatility Band"
                signal = "Oversold"
            else:
                pattern = "Within Volatility Bands"
                signal = "Normal"
            
            confidence = min(70, max(50, abs(band_position - 0.5) * 100 + 50))
            
            return {
                'current_value': band_position,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence
            }
            
        except Exception as e:
            return self.create_default_indicator_result("Volatility Bands", 0.5, "Calculation Error", "None", 0)
    
    def analyze_williams_r(self, high_price, low_price, close_price, period=14):
        """Williams %R momentum analysis"""
        if len(close_price) < period:
            return self.create_default_indicator_result("Williams %R", -50, "Insufficient Data", "None", 0)
        
        try:
            williams_r = talib.WILLR(high_price, low_price, close_price, timeperiod=period)
            current_wr = williams_r[-1]
            
            # Pattern analysis
            if current_wr > -20:
                pattern = "Williams %R Overbought"
                signal = "Sell"
            elif current_wr < -80:
                pattern = "Williams %R Oversold"
                signal = "Buy"
            elif -50 <= current_wr <= -40:
                pattern = "Williams %R Neutral Zone"
                signal = "Neutral"
            else:
                pattern = "Williams %R Normal Range"
                signal = "Hold"
            
            confidence = min(75, max(50, abs(current_wr + 50) * 1.5 + 50))
            
            return {
                'current_value': current_wr,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence
            }
            
        except Exception as e:
            return self.create_default_indicator_result("Williams %R", -50, "Calculation Error", "None", 0)
    
    def analyze_cci(self, high_price, low_price, close_price, period=20):
        """Commodity Channel Index with cyclical pattern detection"""
        if len(close_price) < period:
            return self.create_default_indicator_result("CCI", 0, "Insufficient Data", "None", 0)
        
        try:
            cci = talib.CCI(high_price, low_price, close_price, timeperiod=period)
            current_cci = cci[-1]
            
            # Cyclical pattern detection
            if len(cci) >= 10:
                cci_peaks = find_peaks(cci[-50:] if len(cci) >= 50 else cci)[0]
                cci_troughs = find_peaks(-cci[-50:] if len(cci) >= 50 else -cci)[0]
                
                cycle_detected = len(cci_peaks) >= 2 and len(cci_troughs) >= 2
            else:
                cycle_detected = False
            
            # Pattern analysis
            if current_cci > 100:
                pattern = "CCI Overbought" + (" - Cyclical Pattern" if cycle_detected else "")
                signal = "Sell"
            elif current_cci < -100:
                pattern = "CCI Oversold" + (" - Cyclical Pattern" if cycle_detected else "")
                signal = "Buy"
            elif cycle_detected:
                pattern = "CCI Cyclical Pattern Detected"
                signal = "Watch Cycle"
            else:
                pattern = "CCI Normal Range"
                signal = "Neutral"
            
            confidence = min(80, max(55, abs(current_cci) * 0.5 + 55))
            
            return {
                'current_value': current_cci,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence,
                'cycle_detected': cycle_detected
            }
            
        except Exception as e:
            return self.create_default_indicator_result("CCI", 0, "Calculation Error", "None", 0)
    
    def analyze_roc(self, close_price, period=12):
        """Rate of Change analysis"""
        if len(close_price) < period + 1:
            return self.create_default_indicator_result("ROC", 0, "Insufficient Data", "None", 0)
        
        try:
            roc = talib.ROC(close_price, timeperiod=period)
            current_roc = roc[-1]
            
            # ROC momentum analysis
            if len(roc) >= 5:
                roc_trend = np.polyfit(range(5), roc[-5:], 1)[0]
                if roc_trend > 0.5:
                    trend_pattern = "Accelerating"
                elif roc_trend < -0.5:
                    trend_pattern = "Decelerating"
                else:
                    trend_pattern = "Stable"
            else:
                trend_pattern = "Stable"
            
            if current_roc > 5:
                pattern = f"Strong Positive ROC - {trend_pattern}"
                signal = "Strong Buy"
            elif current_roc < -5:
                pattern = f"Strong Negative ROC - {trend_pattern}"
                signal = "Strong Sell"
            elif current_roc > 0:
                pattern = f"Positive ROC - {trend_pattern}"
                signal = "Buy"
            elif current_roc < 0:
                pattern = f"Negative ROC - {trend_pattern}"
                signal = "Sell"
            else:
                pattern = "Neutral ROC"
                signal = "Neutral"
            
            confidence = min(75, max(50, abs(current_roc) * 5 + 50))
            
            return {
                'current_value': current_roc,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence
            }
            
        except Exception as e:
            return self.create_default_indicator_result("ROC", 0, "Calculation Error", "None", 0)
    
    def analyze_mfi(self, high_price, low_price, close_price, volume, period=14):
        """Money Flow Index analysis"""
        if len(close_price) < period + 1:
            return self.create_default_indicator_result("MFI", 50, "Insufficient Data", "None", 0)
        
        try:
            mfi = talib.MFI(high_price, low_price, close_price, volume, timeperiod=period)
            current_mfi = mfi[-1]
            
            # MFI divergence analysis (similar to RSI)
            divergence = "None"
            if len(mfi) >= 10 and len(close_price) >= 10:
                if (close_price[-1] > close_price[-5] and mfi[-1] < mfi[-5] and current_mfi > 80):
                    divergence = "Bearish"
                elif (close_price[-1] < close_price[-5] and mfi[-1] > mfi[-5] and current_mfi < 20):
                    divergence = "Bullish"
            
            # Pattern analysis
            if current_mfi > 80:
                pattern = "MFI Overbought" + (f" - {divergence} Divergence" if divergence != "None" else "")
                signal = "Strong Sell" if divergence == "Bearish" else "Sell"
            elif current_mfi < 20:
                pattern = "MFI Oversold" + (f" - {divergence} Divergence" if divergence != "None" else "")
                signal = "Strong Buy" if divergence == "Bullish" else "Buy"
            else:
                pattern = "MFI Normal Range"
                signal = "Neutral"
            
            confidence = min(80, max(55, abs(current_mfi - 50) * 1.5 + 55))
            
            return {
                'current_value': current_mfi,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence,
                'divergence': divergence
            }
            
        except Exception as e:
            return self.create_default_indicator_result("MFI", 50, "Calculation Error", "None", 0)
    
    def analyze_adx(self, high_price, low_price, close_price, period=14):
        """ADX trend strength evolution analysis"""
        if len(close_price) < period * 2:
            return self.create_default_indicator_result("ADX", 25, "Insufficient Data", "None", 0)
        
        try:
            adx = talib.ADX(high_price, low_price, close_price, timeperiod=period)
            plus_di = talib.PLUS_DI(high_price, low_price, close_price, timeperiod=period)
            minus_di = talib.MINUS_DI(high_price, low_price, close_price, timeperiod=period)
            
            current_adx = adx[-1]
            current_plus_di = plus_di[-1]
            current_minus_di = minus_di[-1]
            
            # Trend strength evolution
            if len(adx) >= 5:
                adx_slope = np.polyfit(range(5), adx[-5:], 1)[0]
                if adx_slope > 1:
                    evolution = "Strengthening"
                elif adx_slope < -1:
                    evolution = "Weakening"
                else:
                    evolution = "Stable"
            else:
                evolution = "Stable"
            
            # Trend direction and strength
            if current_adx > 40:
                strength = "Very Strong"
            elif current_adx > 25:
                strength = "Strong"
            elif current_adx > 15:
                strength = "Moderate"
            else:
                strength = "Weak"
            
            direction = "Bullish" if current_plus_di > current_minus_di else "Bearish"
            
            pattern = f"{strength} {direction} Trend - {evolution}"
            
            if strength in ["Very Strong", "Strong"] and evolution == "Strengthening":
                signal = "Strong Buy" if direction == "Bullish" else "Strong Sell"
            elif strength == "Weak":
                signal = "Sideways"
            else:
                signal = "Buy" if direction == "Bullish" else "Sell"
            
            confidence = min(85, max(50, current_adx * 2 + 30))
            
            return {
                'current_value': current_adx,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence,
                'trend_strength': strength,
                'trend_direction': direction,
                'evolution': evolution
            }
            
        except Exception as e:
            return self.create_default_indicator_result("ADX", 25, "Calculation Error", "None", 0)
    
    def analyze_parabolic_sar(self, high_price, low_price):
        """Parabolic SAR with reversal timing prediction"""
        try:
            sar = talib.SAR(high_price, low_price)
            current_price = (high_price[-1] + low_price[-1]) / 2
            current_sar = sar[-1]
            
            # Trend and reversal analysis
            if current_price > current_sar:
                trend = "Bullish"
                signal = "Buy"
            else:
                trend = "Bearish"
                signal = "Sell"
            
            # Reversal timing prediction (distance analysis)
            sar_distance = abs(current_price - current_sar) / current_price * 100
            
            if sar_distance < 2:
                timing = "Imminent Reversal"
                pattern = f"SAR {trend} - {timing}"
            elif sar_distance < 5:
                timing = "Near Reversal"
                pattern = f"SAR {trend} - {timing}"
            else:
                timing = "Trend Continuation"
                pattern = f"SAR {trend} - {timing}"
            
            confidence = min(75, max(50, (5 - sar_distance) * 10 + 50)) if sar_distance <= 5 else 50
            
            return {
                'current_value': current_sar,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence,
                'trend': trend,
                'reversal_timing': timing,
                'distance_percent': sar_distance
            }
            
        except Exception as e:
            return self.create_default_indicator_result("Parabolic SAR", 0, "Calculation Error", "None", 0)
    
    def analyze_aroon(self, high_price, low_price, period=25):
        """Aroon indicator analysis"""
        if len(high_price) < period:
            return self.create_default_indicator_result("Aroon", 50, "Insufficient Data", "None", 0)
        
        try:
            aroon_down, aroon_up = talib.AROON(high_price, low_price, timeperiod=period)
            current_aroon_up = aroon_up[-1]
            current_aroon_down = aroon_down[-1]
            
            aroon_oscillator = current_aroon_up - current_aroon_down
            
            # Pattern analysis
            if current_aroon_up > 70 and current_aroon_down < 30:
                pattern = "Strong Bullish Aroon"
                signal = "Buy"
            elif current_aroon_down > 70 and current_aroon_up < 30:
                pattern = "Strong Bearish Aroon"
                signal = "Sell"
            elif abs(aroon_oscillator) < 20:
                pattern = "Aroon Consolidation"
                signal = "Neutral"
            else:
                pattern = "Moderate Aroon Trend"
                signal = "Buy" if aroon_oscillator > 0 else "Sell"
            
            confidence = min(80, max(50, abs(aroon_oscillator) * 0.8 + 50))
            
            return {
                'current_value': aroon_oscillator,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence,
                'aroon_up': current_aroon_up,
                'aroon_down': current_aroon_down
            }
            
        except Exception as e:
            return self.create_default_indicator_result("Aroon", 0, "Calculation Error", "None", 0)
    
    def analyze_ichimoku(self, high_price, low_price, close_price):
        """Ichimoku Cloud analysis with support/resistance"""
        try:
            # Calculate Ichimoku components
            conversion_periods = 9
            base_periods = 26
            lagging_span2_periods = 52
            displacement = 26
            
            # Conversion Line (Tenkan-sen)
            conversion_line = (talib.MAX(high_price, conversion_periods) + talib.MIN(low_price, conversion_periods)) / 2
            
            # Base Line (Kijun-sen)
            base_line = (talib.MAX(high_price, base_periods) + talib.MIN(low_price, base_periods)) / 2
            
            # Leading Span A (Senkou Span A)
            leading_span_a = (conversion_line + base_line) / 2
            
            # Leading Span B (Senkou Span B)
            leading_span_b = (talib.MAX(high_price, lagging_span2_periods) + talib.MIN(low_price, lagging_span2_periods)) / 2
            
            current_price = close_price[-1]
            current_conversion = conversion_line[-1]
            current_base = base_line[-1]
            current_span_a = leading_span_a[-displacement] if len(leading_span_a) > displacement else leading_span_a[-1]
            current_span_b = leading_span_b[-displacement] if len(leading_span_b) > displacement else leading_span_b[-1]
            
            # Cloud analysis
            cloud_top = max(current_span_a, current_span_b)
            cloud_bottom = min(current_span_a, current_span_b)
            
            if current_price > cloud_top:
                cloud_position = "Above Cloud"
                signal = "Bullish"
            elif current_price < cloud_bottom:
                cloud_position = "Below Cloud"
                signal = "Bearish"
            else:
                cloud_position = "Inside Cloud"
                signal = "Neutral"
            
            # Tenkan/Kijun cross
            tk_cross = "None"
            if len(conversion_line) >= 2 and len(base_line) >= 2:
                if conversion_line[-2] <= base_line[-2] and conversion_line[-1] > base_line[-1]:
                    tk_cross = "Golden Cross"
                elif conversion_line[-2] >= base_line[-2] and conversion_line[-1] < base_line[-1]:
                    tk_cross = "Death Cross"
            
            pattern = f"Ichimoku {cloud_position}"
            if tk_cross != "None":
                pattern += f" - {tk_cross}"
            
            confidence = 70 if cloud_position != "Inside Cloud" else 50
            
            return {
                'current_value': (current_price - cloud_bottom) / (cloud_top - cloud_bottom) if cloud_top != cloud_bottom else 0.5,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence,
                'cloud_position': cloud_position,
                'tk_cross': tk_cross,
                'support_level': cloud_bottom,
                'resistance_level': cloud_top
            }
            
        except Exception as e:
            return self.create_default_indicator_result("Ichimoku", 0.5, "Calculation Error", "None", 0)
    
    def analyze_fibonacci_levels(self, high_price, low_price, lookback=50):
        """Fibonacci retracement analysis with completion probability"""
        try:
            # Find recent high and low
            recent_high = np.max(high_price[-lookback:]) if len(high_price) >= lookback else np.max(high_price)
            recent_low = np.min(low_price[-lookback:]) if len(low_price) >= lookback else np.min(low_price)
            current_price = (high_price[-1] + low_price[-1]) / 2
            
            # Calculate Fibonacci levels
            diff = recent_high - recent_low
            fib_levels = {
                '0%': recent_high,
                '23.6%': recent_high - 0.236 * diff,
                '38.2%': recent_high - 0.382 * diff,
                '50%': recent_high - 0.5 * diff,
                '61.8%': recent_high - 0.618 * diff,
                '78.6%': recent_high - 0.786 * diff,
                '100%': recent_low
            }
            
            # Find nearest Fibonacci level
            nearest_level = min(fib_levels.items(), key=lambda x: abs(x[1] - current_price))
            distance_to_nearest = abs(current_price - nearest_level[1]) / diff * 100
            
            # Completion probability analysis
            if distance_to_nearest < 2:
                completion_probability = "High"
                pattern = f"Near Fibonacci {nearest_level[0]} Level"
                signal = "Watch Support/Resistance"
            elif distance_to_nearest < 5:
                completion_probability = "Medium"
                pattern = f"Approaching Fibonacci {nearest_level[0]} Level"
                signal = "Monitor"
            else:
                completion_probability = "Low"
                pattern = "Between Fibonacci Levels"
                signal = "Neutral"
            
            confidence = max(50, 80 - distance_to_nearest * 3)
            
            return {
                'current_value': (current_price - recent_low) / diff,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence,
                'nearest_level': nearest_level[0],
                'completion_probability': completion_probability,
                'fib_levels': fib_levels
            }
            
        except Exception as e:
            return self.create_default_indicator_result("Fibonacci", 0.5, "Calculation Error", "None", 0)
    
    def analyze_pivot_points(self, high_price, low_price, close_price):
        """Pivot Points with support/resistance cluster analysis"""
        try:
            # Calculate pivot points (using previous day's data)
            if len(high_price) < 2:
                return self.create_default_indicator_result("Pivot Points", 0, "Insufficient Data", "None", 0)
            
            prev_high = high_price[-2]
            prev_low = low_price[-2]
            prev_close = close_price[-2]
            current_price = close_price[-1]
            
            # Standard Pivot Points
            pivot = (prev_high + prev_low + prev_close) / 3
            r1 = 2 * pivot - prev_low
            r2 = pivot + (prev_high - prev_low)
            s1 = 2 * pivot - prev_high
            s2 = pivot - (prev_high - prev_low)
            
            levels = {'S2': s2, 'S1': s1, 'Pivot': pivot, 'R1': r1, 'R2': r2}
            
            # Find nearest level
            nearest_level = min(levels.items(), key=lambda x: abs(x[1] - current_price))
            distance_to_nearest = abs(current_price - nearest_level[1]) / current_price * 100
            
            # Cluster analysis
            level_distances = [abs(current_price - level) / current_price * 100 for level in levels.values()]
            nearby_levels = sum(1 for dist in level_distances if dist < 1)
            
            if nearby_levels >= 2:
                pattern = "Pivot Point Cluster - Strong Support/Resistance"
                signal = "Strong Level"
            elif distance_to_nearest < 0.5:
                pattern = f"At {nearest_level[0]} Pivot Level"
                signal = "Key Level"
            else:
                pattern = "Between Pivot Levels"
                signal = "Normal Trading"
            
            confidence = max(60, 90 - distance_to_nearest * 10)
            
            return {
                'current_value': distance_to_nearest,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence,
                'nearest_level': nearest_level[0],
                'cluster_strength': nearby_levels,
                'pivot_levels': levels
            }
            
        except Exception as e:
            return self.create_default_indicator_result("Pivot Points", 0, "Calculation Error", "None", 0)
    
    def analyze_composite_momentum(self, close_price, volume):
        """Custom composite momentum indicator"""
        try:
            # Combine multiple momentum indicators
            rsi = talib.RSI(close_price, timeperiod=14)
            roc = talib.ROC(close_price, timeperiod=12)
            
            # Volume-weighted momentum
            price_change = np.diff(close_price)
            volume_weighted_momentum = np.correlate(price_change[-20:], volume[-20:], mode='valid')[0] if len(price_change) >= 20 else 0
            
            # Normalize and combine
            current_rsi_norm = (rsi[-1] - 50) / 50 if not np.isnan(rsi[-1]) else 0
            current_roc_norm = np.tanh(roc[-1] / 10) if not np.isnan(roc[-1]) else 0
            volume_momentum_norm = np.tanh(volume_weighted_momentum / np.std(volume[-20:]) if len(volume) >= 20 and np.std(volume[-20:]) > 0 else 0)
            
            composite_momentum = (current_rsi_norm + current_roc_norm + volume_momentum_norm) / 3
            
            if composite_momentum > 0.5:
                pattern = "Strong Composite Momentum"
                signal = "Strong Buy"
            elif composite_momentum > 0.2:
                pattern = "Moderate Positive Momentum"
                signal = "Buy"
            elif composite_momentum < -0.5:
                pattern = "Strong Negative Momentum"
                signal = "Strong Sell"
            elif composite_momentum < -0.2:
                pattern = "Moderate Negative Momentum"
                signal = "Sell"
            else:
                pattern = "Neutral Momentum"
                signal = "Neutral"
            
            confidence = min(80, max(50, abs(composite_momentum) * 80 + 50))
            
            return {
                'current_value': composite_momentum,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence
            }
            
        except Exception as e:
            return self.create_default_indicator_result("Composite Momentum", 0, "Calculation Error", "None", 0)
    
    def analyze_composite_trend(self, high_price, low_price, close_price):
        """Custom composite trend indicator"""
        try:
            # Multiple trend indicators
            adx = talib.ADX(high_price, low_price, close_price, timeperiod=14)
            
            # Multiple timeframe SMA trends
            sma_20 = talib.SMA(close_price, timeperiod=20)
            sma_50 = talib.SMA(close_price, timeperiod=50)
            
            current_price = close_price[-1]
            
            # Trend strength from ADX
            trend_strength = adx[-1] / 100 if not np.isnan(adx[-1]) else 0
            
            # Price vs MA trends
            sma_trend_20 = (current_price - sma_20[-1]) / sma_20[-1] if not np.isnan(sma_20[-1]) else 0
            sma_trend_50 = (current_price - sma_50[-1]) / sma_50[-1] if not np.isnan(sma_50[-1]) else 0
            
            # Combine trends
            composite_trend = (sma_trend_20 + sma_trend_50 * 0.5) * trend_strength
            
            if composite_trend > 0.02:
                pattern = "Strong Composite Uptrend"
                signal = "Strong Buy"
            elif composite_trend > 0.005:
                pattern = "Moderate Uptrend"
                signal = "Buy"
            elif composite_trend < -0.02:
                pattern = "Strong Composite Downtrend"
                signal = "Strong Sell"
            elif composite_trend < -0.005:
                pattern = "Moderate Downtrend"
                signal = "Sell"
            else:
                pattern = "Sideways Trend"
                signal = "Neutral"
            
            confidence = min(85, max(55, abs(composite_trend) * 1000 + 55))
            
            return {
                'current_value': composite_trend,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence
            }
            
        except Exception as e:
            return self.create_default_indicator_result("Composite Trend", 0, "Calculation Error", "None", 0)
    
    def analyze_composite_volatility(self, high_price, low_price, close_price):
        """Custom composite volatility indicator"""
        try:
            # Multiple volatility measures
            atr = talib.ATR(high_price, low_price, close_price, timeperiod=14)
            
            # Price volatility
            returns = np.diff(close_price) / close_price[:-1]
            price_volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0
            
            # Range volatility
            daily_ranges = (high_price - low_price) / close_price
            range_volatility = np.std(daily_ranges[-20:]) if len(daily_ranges) >= 20 else 0
            
            # Normalize and combine
            atr_norm = atr[-1] / np.mean(atr[-50:]) if len(atr) >= 50 and not np.isnan(atr[-1]) else 1
            price_vol_norm = price_volatility / 0.02 if price_volatility > 0 else 1  # Normalize to typical 2% daily volatility
            range_vol_norm = range_volatility / 0.03 if range_volatility > 0 else 1
            
            composite_volatility = (atr_norm + price_vol_norm + range_vol_norm) / 3
            
            if composite_volatility > 2:
                pattern = "Extreme Volatility"
                signal = "High Risk"
            elif composite_volatility > 1.5:
                pattern = "High Volatility"
                signal = "Increased Risk"
            elif composite_volatility < 0.5:
                pattern = "Low Volatility"
                signal = "Low Risk"
            else:
                pattern = "Normal Volatility"
                signal = "Normal Risk"
            
            confidence = min(75, max(50, abs(composite_volatility - 1) * 30 + 50))
            
            return {
                'current_value': composite_volatility,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence
            }
            
        except Exception as e:
            return self.create_default_indicator_result("Composite Volatility", 1, "Calculation Error", "None", 0)
    
    def analyze_donchian_channels(self, high_price, low_price, period=20):
        """Donchian Channels analysis"""
        if len(high_price) < period:
            return self.create_default_indicator_result("Donchian Channels", 0.5, "Insufficient Data", "None", 0)
        
        try:
            upper_channel = talib.MAX(high_price, timeperiod=period)
            lower_channel = talib.MIN(low_price, timeperiod=period)
            
            current_price = (high_price[-1] + low_price[-1]) / 2
            channel_position = (current_price - lower_channel[-1]) / (upper_channel[-1] - lower_channel[-1])
            
            if channel_position > 0.8:
                pattern = "Near Upper Donchian Channel"
                signal = "Potential Breakout Up"
            elif channel_position < 0.2:
                pattern = "Near Lower Donchian Channel"
                signal = "Potential Breakout Down"
            else:
                pattern = "Within Donchian Channels"
                signal = "Range Trading"
            
            confidence = min(70, max(50, abs(channel_position - 0.5) * 100 + 50))
            
            return {
                'current_value': channel_position,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence
            }
            
        except Exception as e:
            return self.create_default_indicator_result("Donchian Channels", 0.5, "Calculation Error", "None", 0)
    
    def analyze_keltner_channels(self, high_price, low_price, close_price, period=20):
        """Keltner Channels analysis"""
        if len(close_price) < period:
            return self.create_default_indicator_result("Keltner Channels", 0.5, "Insufficient Data", "None", 0)
        
        try:
            # Calculate EMA and ATR
            ema = talib.EMA(close_price, timeperiod=period)
            atr = talib.ATR(high_price, low_price, close_price, timeperiod=period)
            
            # Keltner Channels
            upper_channel = ema + (2 * atr)
            lower_channel = ema - (2 * atr)
            
            current_price = close_price[-1]
            channel_position = (current_price - lower_channel[-1]) / (upper_channel[-1] - lower_channel[-1])
            
            if channel_position > 0.8:
                pattern = "Near Upper Keltner Channel"
                signal = "Overbought"
            elif channel_position < 0.2:
                pattern = "Near Lower Keltner Channel"
                signal = "Oversold"
            else:
                pattern = "Within Keltner Channels"
                signal = "Normal"
            
            confidence = min(75, max(50, abs(channel_position - 0.5) * 100 + 50))
            
            return {
                'current_value': channel_position,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence
            }
            
        except Exception as e:
            return self.create_default_indicator_result("Keltner Channels", 0.5, "Calculation Error", "None", 0)
    
    def analyze_trix(self, close_price, period=14):
        """TRIX analysis"""
        if len(close_price) < period * 3:
            return self.create_default_indicator_result("TRIX", 0, "Insufficient Data", "None", 0)
        
        try:
            # Calculate TRIX manually since talib is not available
            ema1 = pd.Series(close_price).ewm(span=period).mean()
            ema2 = ema1.ewm(span=period).mean()
            ema3 = ema2.ewm(span=period).mean()
            trix = (ema3.pct_change() * 10000).values
            current_trix = trix[-1] if len(trix) > 0 else 0
            
            # TRIX signal analysis
            if len(trix) >= 2:
                if trix[-2] <= 0 and trix[-1] > 0:
                    pattern = "TRIX Bullish Signal"
                    signal = "Buy"
                elif trix[-2] >= 0 and trix[-1] < 0:
                    pattern = "TRIX Bearish Signal"
                    signal = "Sell"
                elif current_trix > 0:
                    pattern = "TRIX Positive"
                    signal = "Bullish"
                elif current_trix < 0:
                    pattern = "TRIX Negative"
                    signal = "Bearish"
                else:
                    pattern = "TRIX Neutral"
                    signal = "Neutral"
            else:
                pattern = "TRIX Insufficient Data"
                signal = "Neutral"
            
            confidence = min(70, max(50, abs(current_trix) * 5 + 50))
            
            return {
                'current_value': current_trix,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence
            }
            
        except Exception as e:
            return self.create_default_indicator_result("TRIX", 0, "Calculation Error", "None", 0)
    
    def analyze_elder_ray(self, high_price, low_price, close_price, period=13):
        """Elder Ray analysis"""
        if len(close_price) < period:
            return self.create_default_indicator_result("Elder Ray", 0, "Insufficient Data", "None", 0)
        
        try:
            # Calculate EMA manually since talib is not available
            ema = pd.Series(close_price).ewm(span=period).mean().values
            bull_power = high_price - ema
            bear_power = low_price - ema
            
            current_bull = bull_power[-1]
            current_bear = bear_power[-1]
            
            # Elder Ray analysis
            if current_bull > 0 and current_bear > 0:
                pattern = "Strong Bull Power"
                signal = "Strong Buy"
            elif current_bull > 0 and current_bear < 0:
                pattern = "Moderate Bull Power"
                signal = "Buy"
            elif current_bull < 0 and current_bear < 0:
                pattern = "Strong Bear Power"
                signal = "Strong Sell"
            elif current_bull < 0 and current_bear > 0:
                pattern = "Moderate Bear Power"
                signal = "Sell"
            else:
                pattern = "Balanced Elder Ray"
                signal = "Neutral"
            
            power_difference = current_bull + abs(current_bear)
            confidence = min(75, max(50, abs(power_difference) / close_price[-1] * 1000 + 50))
            
            return {
                'current_value': current_bull - current_bear,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence,
                'bull_power': current_bull,
                'bear_power': current_bear
            }
            
        except Exception as e:
            return self.create_default_indicator_result("Elder Ray", 0, "Calculation Error", "None", 0)
    
    def analyze_chande_momentum(self, close_price, period=20):
        """Chande Momentum Oscillator"""
        if len(close_price) < period + 1:
            return self.create_default_indicator_result("Chande Momentum", 0, "Insufficient Data", "None", 0)
        
        try:
            cmo = talib.CMO(close_price, timeperiod=period)
            current_cmo = cmo[-1]
            
            # CMO analysis
            if current_cmo > 50:
                pattern = "Strong Chande Momentum Up"
                signal = "Strong Buy"
            elif current_cmo > 25:
                pattern = "Moderate Chande Momentum Up"
                signal = "Buy"
            elif current_cmo < -50:
                pattern = "Strong Chande Momentum Down"
                signal = "Strong Sell"
            elif current_cmo < -25:
                pattern = "Moderate Chande Momentum Down"
                signal = "Sell"
            else:
                pattern = "Neutral Chande Momentum"
                signal = "Neutral"
            
            confidence = min(80, max(50, abs(current_cmo) * 0.8 + 50))
            
            return {
                'current_value': current_cmo,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence
            }
            
        except Exception as e:
            return self.create_default_indicator_result("Chande Momentum", 0, "Calculation Error", "None", 0)
    
    def analyze_ultimate_oscillator(self, high_price, low_price, close_price):
        """Ultimate Oscillator analysis"""
        if len(close_price) < 28:
            return self.create_default_indicator_result("Ultimate Oscillator", 50, "Insufficient Data", "None", 0)
        
        try:
            uo = talib.ULTOSC(high_price, low_price, close_price)
            current_uo = uo[-1]
            
            # Ultimate Oscillator analysis
            if current_uo > 80:
                pattern = "Ultimate Oscillator Overbought"
                signal = "Sell"
            elif current_uo < 20:
                pattern = "Ultimate Oscillator Oversold"
                signal = "Buy"
            elif current_uo > 60:
                pattern = "Ultimate Oscillator Bullish"
                signal = "Bullish"
            elif current_uo < 40:
                pattern = "Ultimate Oscillator Bearish"
                signal = "Bearish"
            else:
                pattern = "Ultimate Oscillator Neutral"
                signal = "Neutral"
            
            confidence = min(75, max(50, abs(current_uo - 50) * 1.5 + 50))
            
            return {
                'current_value': current_uo,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence
            }
            
        except Exception as e:
            return self.create_default_indicator_result("Ultimate Oscillator", 50, "Calculation Error", "None", 0)
    
    def analyze_detrended_price_oscillator(self, close_price, period=20):
        """Detrended Price Oscillator"""
        if len(close_price) < period * 2:
            return self.create_default_indicator_result("DPO", 0, "Insufficient Data", "None", 0)
        
        try:
            # Calculate DPO manually
            sma = talib.SMA(close_price, timeperiod=period)
            shift = period // 2 + 1
            
            if len(close_price) > shift and len(sma) > shift:
                dpo = close_price[:-shift] - sma[:-shift]
                current_dpo = dpo[-1] if len(dpo) > 0 else 0
            else:
                current_dpo = 0
            
            # DPO analysis
            dpo_std = np.std(dpo[-20:]) if len(dpo) >= 20 else 1
            normalized_dpo = current_dpo / dpo_std if dpo_std > 0 else 0
            
            if normalized_dpo > 1:
                pattern = "DPO High Cycle"
                signal = "Cycle High"
            elif normalized_dpo < -1:
                pattern = "DPO Low Cycle"
                signal = "Cycle Low"
            else:
                pattern = "DPO Normal Cycle"
                signal = "Normal"
            
            confidence = min(70, max(50, abs(normalized_dpo) * 30 + 50))
            
            return {
                'current_value': current_dpo,
                'signal': signal,
                'pattern_detected': pattern,
                'confidence': confidence
            }
            
        except Exception as e:
            return self.create_default_indicator_result("DPO", 0, "Calculation Error", "None", 0)
    
    def create_default_indicator_result(self, name, value, signal, pattern, confidence):
        """Create default result structure for failed calculations"""
        return {
            'current_value': value,
            'signal': signal,
            'pattern_detected': pattern,
            'confidence': confidence
        }
    
    # Manual implementations of technical indicators
    def sma(self, data, period):
        """Simple Moving Average"""
        return pd.Series(data).rolling(window=period).mean().values
    
    def ema(self, data, period):
        """Exponential Moving Average"""
        return pd.Series(data).ewm(span=period).mean().values
    
    def rolling_std(self, data, period):
        """Rolling Standard Deviation"""
        return pd.Series(data).rolling(window=period).std().values
    
    def rsi_manual(self, data, period=14):
        """Manual RSI calculation"""
        delta = pd.Series(data).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).values
    
    def macd_manual(self, data, fast=12, slow=26, signal=9):
        """Manual MACD calculation"""
        ema_fast = self.ema(data, fast)
        ema_slow = self.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
