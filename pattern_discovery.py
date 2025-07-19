import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
import cv2
from skimage import feature, measure
import warnings
warnings.filterwarnings('ignore')

class PatternDiscoveryEngine:
    """
    Autonomous Pattern Discovery Engine for discovering new candlestick and chart patterns
    using unsupervised machine learning and computer vision techniques
    """
    
    def __init__(self):
        self.discovered_patterns = {}
        self.pattern_clusters = {}
        self.traditional_patterns = {}
        self.success_rates = {}
        
    def extract_candlestick_features(self, data, window=5):
        """Extract comprehensive features from candlestick data"""
        features = []
        
        for i in range(window, len(data)):
            window_data = data.iloc[i-window:i+1]
            
            # Basic OHLC features
            open_prices = window_data['Open'].values
            high_prices = window_data['High'].values
            low_prices = window_data['Low'].values
            close_prices = window_data['Close'].values
            volumes = window_data['Volume'].values
            
            # Candlestick body and shadow features
            bodies = np.abs(close_prices - open_prices)
            upper_shadows = high_prices - np.maximum(open_prices, close_prices)
            lower_shadows = np.minimum(open_prices, close_prices) - low_prices
            
            # Normalized features
            avg_close = np.mean(close_prices)
            body_ratios = bodies / avg_close
            upper_shadow_ratios = upper_shadows / avg_close
            lower_shadow_ratios = lower_shadows / avg_close
            
            # Pattern features
            feature_vector = np.concatenate([
                body_ratios,
                upper_shadow_ratios,
                lower_shadow_ratios,
                close_prices / close_prices[0] - 1,  # Normalized price changes
                volumes / np.mean(volumes),  # Volume ratios
                [np.std(close_prices) / avg_close],  # Volatility
                [np.sum(bodies) / np.sum(high_prices - low_prices)]  # Body to range ratio
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def discover_candlestick_patterns(self, data):
        """Discover new candlestick patterns using unsupervised learning"""
        # Extract features
        features = self.extract_candlestick_features(data)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=0.95)
        features_pca = pca.fit_transform(features_scaled)
        
        # Cluster patterns using multiple algorithms
        clustering_results = {}
        
        # K-Means clustering
        for n_clusters in range(5, 15):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features_pca)
            clustering_results[f'kmeans_{n_clusters}'] = clusters
        
        # DBSCAN clustering
        for eps in [0.5, 1.0, 1.5]:
            dbscan = DBSCAN(eps=eps, min_samples=5)
            clusters = dbscan.fit_predict(features_pca)
            clustering_results[f'dbscan_{eps}'] = clusters
        
        # Analyze discovered patterns
        discovered_patterns = []
        
        for method, clusters in clustering_results.items():
            unique_clusters = np.unique(clusters)
            
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise in DBSCAN
                    continue
                    
                cluster_indices = np.where(clusters == cluster_id)[0]
                
                if len(cluster_indices) > 10:  # Minimum occurrences
                    # Calculate pattern characteristics
                    cluster_features = features[cluster_indices]
                    
                    # Success rate calculation
                    future_returns = []
                    for idx in cluster_indices:
                        if idx + 5 < len(data):
                            future_return = (data['Close'].iloc[idx + 5] - data['Close'].iloc[idx]) / data['Close'].iloc[idx]
                            future_returns.append(future_return)
                    
                    avg_return = np.mean(future_returns) if future_returns else 0
                    success_rate = (np.sum(np.array(future_returns) > 0) / len(future_returns) * 100) if future_returns else 50
                    
                    # Pattern naming based on characteristics
                    avg_feature = np.mean(cluster_features, axis=0)
                    pattern_name = self.generate_pattern_name(avg_feature)
                    
                    discovered_patterns.append({
                        'name': pattern_name,
                        'method': method,
                        'cluster_id': cluster_id,
                        'occurrences': len(cluster_indices),
                        'success_rate': success_rate,
                        'avg_return': avg_return,
                        'confidence': min(0.95, len(cluster_indices) / 100),
                        'pattern_signature': avg_feature
                    })
        
        # Sort by success rate and occurrences
        discovered_patterns.sort(key=lambda x: (x['success_rate'], x['occurrences']), reverse=True)
        
        return discovered_patterns[:10]  # Return top 10 patterns
    
    def generate_pattern_name(self, feature_vector):
        """Generate descriptive names for discovered patterns"""
        # Analyze feature characteristics
        body_strength = np.mean(feature_vector[:5])  # First 5 are body ratios
        upper_shadow_strength = np.mean(feature_vector[5:10])  # Next 5 are upper shadows
        lower_shadow_strength = np.mean(feature_vector[10:15])  # Next 5 are lower shadows
        
        # Generate name based on characteristics
        name_parts = []
        
        if body_strength > 0.02:
            name_parts.append("Strong-Body")
        elif body_strength < 0.005:
            name_parts.append("Weak-Body")
        else:
            name_parts.append("Medium-Body")
        
        if upper_shadow_strength > 0.015:
            name_parts.append("Long-Upper")
        elif lower_shadow_strength > 0.015:
            name_parts.append("Long-Lower")
        
        if len(name_parts) == 1:
            name_parts.append("Formation")
        
        return "-".join(name_parts)
    
    def detect_traditional_candlestick_patterns(self, data):
        """Detect traditional candlestick patterns"""
        patterns = []
        
        for i in range(2, len(data)):
            current = data.iloc[i]
            prev1 = data.iloc[i-1]
            prev2 = data.iloc[i-2] if i >= 2 else None
            
            # Calculate pattern characteristics
            body = abs(current['Close'] - current['Open'])
            upper_shadow = current['High'] - max(current['Close'], current['Open'])
            lower_shadow = min(current['Close'], current['Open']) - current['Low']
            total_range = current['High'] - current['Low']
            
            # Doji patterns
            if body < (total_range * 0.1):
                if upper_shadow > body * 2 and lower_shadow < body:
                    patterns.append({
                        'name': 'Dragonfly Doji',
                        'type': 'Reversal',
                        'reliability': 75,
                        'timeframe': 'Short-term',
                        'index': i
                    })
                elif lower_shadow > body * 2 and upper_shadow < body:
                    patterns.append({
                        'name': 'Gravestone Doji',
                        'type': 'Reversal',
                        'reliability': 75,
                        'timeframe': 'Short-term',
                        'index': i
                    })
                else:
                    patterns.append({
                        'name': 'Standard Doji',
                        'type': 'Indecision',
                        'reliability': 60,
                        'timeframe': 'Short-term',
                        'index': i
                    })
            
            # Hammer and Hanging Man
            if lower_shadow > body * 2 and upper_shadow < body * 0.5:
                if current['Close'] < prev1['Close']:
                    patterns.append({
                        'name': 'Hammer',
                        'type': 'Bullish Reversal',
                        'reliability': 70,
                        'timeframe': 'Short-term',
                        'index': i
                    })
                else:
                    patterns.append({
                        'name': 'Hanging Man',
                        'type': 'Bearish Reversal',
                        'reliability': 65,
                        'timeframe': 'Short-term',
                        'index': i
                    })
            
            # Shooting Star and Inverted Hammer
            if upper_shadow > body * 2 and lower_shadow < body * 0.5:
                if current['Close'] > prev1['Close']:
                    patterns.append({
                        'name': 'Inverted Hammer',
                        'type': 'Bullish Reversal',
                        'reliability': 65,
                        'timeframe': 'Short-term',
                        'index': i
                    })
                else:
                    patterns.append({
                        'name': 'Shooting Star',
                        'type': 'Bearish Reversal',
                        'reliability': 70,
                        'timeframe': 'Short-term',
                        'index': i
                    })
            
            # Engulfing patterns (requires previous candle)
            if prev1 is not None:
                prev_body = abs(prev1['Close'] - prev1['Open'])
                
                if (current['Open'] < prev1['Close'] < prev1['Open'] and 
                    current['Close'] > prev1['Open'] and body > prev_body):
                    patterns.append({
                        'name': 'Bullish Engulfing',
                        'type': 'Bullish Reversal',
                        'reliability': 80,
                        'timeframe': 'Medium-term',
                        'index': i
                    })
                
                elif (current['Open'] > prev1['Close'] > prev1['Open'] and 
                      current['Close'] < prev1['Open'] and body > prev_body):
                    patterns.append({
                        'name': 'Bearish Engulfing',
                        'type': 'Bearish Reversal',
                        'reliability': 80,
                        'timeframe': 'Medium-term',
                        'index': i
                    })
            
            # Morning and Evening Star (requires 2 previous candles)
            if prev2 is not None:
                if (prev2['Close'] < prev2['Open'] and  # First candle bearish
                    abs(prev1['Close'] - prev1['Open']) < (prev2['High'] - prev2['Low']) * 0.3 and  # Second candle small
                    current['Close'] > current['Open'] and  # Third candle bullish
                    current['Close'] > (prev2['Open'] + prev2['Close']) / 2):
                    patterns.append({
                        'name': 'Morning Star',
                        'type': 'Bullish Reversal',
                        'reliability': 85,
                        'timeframe': 'Medium-term',
                        'index': i
                    })
                
                elif (prev2['Close'] > prev2['Open'] and  # First candle bullish
                      abs(prev1['Close'] - prev1['Open']) < (prev2['High'] - prev2['Low']) * 0.3 and  # Second candle small
                      current['Close'] < current['Open'] and  # Third candle bearish
                      current['Close'] < (prev2['Open'] + prev2['Close']) / 2):
                    patterns.append({
                        'name': 'Evening Star',
                        'type': 'Bearish Reversal',
                        'reliability': 85,
                        'timeframe': 'Medium-term',
                        'index': i
                    })
        
        return patterns
    
    def detect_chart_patterns(self, data):
        """Detect traditional and custom chart patterns using computer vision"""
        patterns = []
        
        # Convert price data to image-like format for pattern recognition
        price_image = self.create_price_image(data)
        
        # Detect support and resistance levels
        support_levels, resistance_levels = self.detect_support_resistance(data)
        
        # Detect various chart patterns
        patterns.extend(self.detect_head_and_shoulders(data, support_levels, resistance_levels))
        patterns.extend(self.detect_triangles(data, support_levels, resistance_levels))
        patterns.extend(self.detect_flags_pennants(data))
        patterns.extend(self.detect_double_tops_bottoms(data))
        patterns.extend(self.detect_channels(data, support_levels, resistance_levels))
        
        return patterns
    
    def create_price_image(self, data, width=200, height=100):
        """Create image representation of price data for computer vision analysis"""
        prices = data['Close'].values
        normalized_prices = (prices - prices.min()) / (prices.max() - prices.min())
        
        # Create image
        image = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(1, len(normalized_prices)):
            x1 = int((i-1) * width / len(prices))
            x2 = int(i * width / len(prices))
            y1 = int((1 - normalized_prices[i-1]) * (height - 1))
            y2 = int((1 - normalized_prices[i]) * (height - 1))
            
            cv2.line(image, (x1, y1), (x2, y2), 255, 1)
        
        return image
    
    def detect_support_resistance(self, data, window=20):
        """Detect support and resistance levels"""
        highs = data['High'].values
        lows = data['Low'].values
        
        # Find local maxima and minima
        resistance_indices = find_peaks(highs, distance=window)[0]
        support_indices = find_peaks(-lows, distance=window)[0]
        
        # Get levels with strength
        resistance_levels = []
        for idx in resistance_indices:
            level = highs[idx]
            strength = self.calculate_level_strength(data, level, 'resistance')
            resistance_levels.append({'level': level, 'strength': strength, 'index': idx})
        
        support_levels = []
        for idx in support_indices:
            level = lows[idx]
            strength = self.calculate_level_strength(data, level, 'support')
            support_levels.append({'level': level, 'strength': strength, 'index': idx})
        
        # Sort by strength
        resistance_levels.sort(key=lambda x: x['strength'], reverse=True)
        support_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return support_levels[:5], resistance_levels[:5]
    
    def calculate_level_strength(self, data, level, level_type, tolerance=0.02):
        """Calculate strength of support/resistance level"""
        if level_type == 'resistance':
            touches = np.sum(np.abs(data['High'] - level) / level < tolerance)
        else:
            touches = np.sum(np.abs(data['Low'] - level) / level < tolerance)
        
        return touches
    
    def detect_head_and_shoulders(self, data, support_levels, resistance_levels):
        """Detect Head and Shoulders patterns"""
        patterns = []
        
        if len(resistance_levels) >= 3:
            # Look for three peaks where middle is highest
            for i in range(len(resistance_levels) - 2):
                left_shoulder = resistance_levels[i]
                head = resistance_levels[i + 1]
                right_shoulder = resistance_levels[i + 2]
                
                if (head['level'] > left_shoulder['level'] and 
                    head['level'] > right_shoulder['level'] and
                    abs(left_shoulder['level'] - right_shoulder['level']) / left_shoulder['level'] < 0.05):
                    
                    # Check if pattern is recent
                    if max(left_shoulder['index'], head['index'], right_shoulder['index']) > len(data) - 50:
                        completion_probability = min(0.8, (left_shoulder['strength'] + head['strength'] + right_shoulder['strength']) / 15)
                        
                        patterns.append({
                            'pattern_type': 'Head and Shoulders',
                            'formation_stage': 'Completed' if right_shoulder['index'] > head['index'] else 'Forming',
                            'completion_probability': completion_probability,
                            'expected_breakout': 'Bearish',
                            'price_target': left_shoulder['level'] - (head['level'] - left_shoulder['level']),
                            'key_levels': [left_shoulder['level'], head['level'], right_shoulder['level']]
                        })
        
        return patterns
    
    def detect_triangles(self, data, support_levels, resistance_levels):
        """Detect triangle patterns"""
        patterns = []
        
        # Ascending triangle
        if len(resistance_levels) >= 2 and len(support_levels) >= 2:
            # Check for horizontal resistance and rising support
            recent_resistance = [r for r in resistance_levels if r['index'] > len(data) - 100]
            recent_support = [s for s in support_levels if s['index'] > len(data) - 100]
            
            if len(recent_resistance) >= 2 and len(recent_support) >= 2:
                # Horizontal resistance check
                res_levels = [r['level'] for r in recent_resistance[:2]]
                if abs(res_levels[0] - res_levels[1]) / res_levels[0] < 0.03:
                    
                    # Rising support check
                    sup_levels = sorted(recent_support[:2], key=lambda x: x['index'])
                    if sup_levels[1]['level'] > sup_levels[0]['level']:
                        
                        patterns.append({
                            'pattern_type': 'Ascending Triangle',
                            'formation_stage': 'Forming',
                            'completion_probability': 0.7,
                            'expected_breakout': 'Bullish',
                            'price_target': res_levels[0] + (res_levels[0] - sup_levels[0]['level']),
                            'key_levels': [sup_levels[0]['level'], sup_levels[1]['level'], res_levels[0]]
                        })
        
        return patterns
    
    def detect_flags_pennants(self, data):
        """Detect flag and pennant patterns"""
        patterns = []
        
        # Look for strong moves followed by consolidation
        returns = data['Close'].pct_change().abs()
        volume = data['Volume']
        
        for i in range(20, len(data) - 10):
            # Check for strong initial move
            strong_move_returns = returns.iloc[i-20:i].mean()
            consolidation_returns = returns.iloc[i:i+10].mean()
            
            if strong_move_returns > consolidation_returns * 2:
                # Check volume pattern
                strong_move_volume = volume.iloc[i-20:i].mean()
                consolidation_volume = volume.iloc[i:i+10].mean()
                
                if strong_move_volume > consolidation_volume:
                    direction = 'Bullish' if data['Close'].iloc[i] > data['Close'].iloc[i-20] else 'Bearish'
                    
                    patterns.append({
                        'pattern_type': 'Flag Pattern',
                        'formation_stage': 'Consolidation',
                        'completion_probability': 0.6,
                        'expected_breakout': direction,
                        'price_target': data['Close'].iloc[i] + (data['Close'].iloc[i] - data['Close'].iloc[i-20]),
                        'key_levels': [data['Close'].iloc[i-20], data['Close'].iloc[i]]
                    })
        
        return patterns
    
    def detect_double_tops_bottoms(self, data):
        """Detect double top and bottom patterns"""
        patterns = []
        
        highs = data['High'].values
        lows = data['Low'].values
        
        # Find peaks for double tops
        peaks = find_peaks(highs, distance=20)[0]
        
        for i in range(len(peaks) - 1):
            peak1_idx = peaks[i]
            peak2_idx = peaks[i + 1]
            
            if (abs(highs[peak1_idx] - highs[peak2_idx]) / highs[peak1_idx] < 0.03 and
                peak2_idx - peak1_idx > 20):  # At least 20 periods apart
                
                # Find valley between peaks
                valley_idx = peak1_idx + np.argmin(lows[peak1_idx:peak2_idx])
                valley_level = lows[valley_idx]
                
                patterns.append({
                    'pattern_type': 'Double Top',
                    'formation_stage': 'Completed',
                    'completion_probability': 0.75,
                    'expected_breakout': 'Bearish',
                    'price_target': valley_level - (highs[peak1_idx] - valley_level),
                    'key_levels': [highs[peak1_idx], valley_level, highs[peak2_idx]]
                })
        
        # Find troughs for double bottoms
        troughs = find_peaks(-lows, distance=20)[0]
        
        for i in range(len(troughs) - 1):
            trough1_idx = troughs[i]
            trough2_idx = troughs[i + 1]
            
            if (abs(lows[trough1_idx] - lows[trough2_idx]) / lows[trough1_idx] < 0.03 and
                trough2_idx - trough1_idx > 20):
                
                # Find peak between troughs
                peak_idx = trough1_idx + np.argmax(highs[trough1_idx:trough2_idx])
                peak_level = highs[peak_idx]
                
                patterns.append({
                    'pattern_type': 'Double Bottom',
                    'formation_stage': 'Completed',
                    'completion_probability': 0.75,
                    'expected_breakout': 'Bullish',
                    'price_target': peak_level + (peak_level - lows[trough1_idx]),
                    'key_levels': [lows[trough1_idx], peak_level, lows[trough2_idx]]
                })
        
        return patterns
    
    def detect_channels(self, data, support_levels, resistance_levels):
        """Detect price channels"""
        patterns = []
        
        if len(support_levels) >= 2 and len(resistance_levels) >= 2:
            # Check for parallel lines
            recent_support = [s for s in support_levels if s['index'] > len(data) - 100][:2]
            recent_resistance = [r for r in resistance_levels if r['index'] > len(data) - 100][:2]
            
            if len(recent_support) == 2 and len(recent_resistance) == 2:
                # Calculate slopes
                support_slope = (recent_support[1]['level'] - recent_support[0]['level']) / (recent_support[1]['index'] - recent_support[0]['index'])
                resistance_slope = (recent_resistance[1]['level'] - recent_resistance[0]['level']) / (recent_resistance[1]['index'] - recent_resistance[0]['index'])
                
                # Check if slopes are similar (parallel lines)
                if abs(support_slope - resistance_slope) < 0.01:
                    channel_type = 'Ascending' if support_slope > 0 else 'Descending' if support_slope < 0 else 'Horizontal'
                    
                    patterns.append({
                        'pattern_type': f'{channel_type} Channel',
                        'formation_stage': 'Active',
                        'completion_probability': 0.8,
                        'expected_breakout': 'Either Direction',
                        'price_target': 'Channel Width',
                        'key_levels': [recent_support[0]['level'], recent_resistance[0]['level']]
                    })
        
        return patterns
    
    def discover_patterns(self, market_data):
        """Main method to discover all types of patterns"""
        # Discover new candlestick patterns
        discovered_candlestick = self.discover_candlestick_patterns(market_data)
        
        # Detect traditional patterns
        traditional_candlestick = self.detect_traditional_candlestick_patterns(market_data)
        
        # Detect chart patterns
        chart_patterns = self.detect_chart_patterns(market_data)
        
        return {
            'discovered_patterns': discovered_candlestick,
            'traditional_candlestick': traditional_candlestick,
            'chart_patterns': chart_patterns
        }
    
    def detect_candlestick_patterns(self, market_data):
        """Wrapper method for candlestick pattern detection"""
        traditional = self.detect_traditional_candlestick_patterns(market_data)
        ml_discovered = self.discover_candlestick_patterns(market_data)
        
        return {
            'traditional_patterns': traditional,
            'ml_discovered_patterns': [
                {
                    'custom_name': pattern['name'],
                    'formation_type': 'ML-Discovered',
                    'historical_success': pattern['success_rate'],
                    'total_occurrences': pattern['occurrences']
                }
                for pattern in ml_discovered
            ]
        }
