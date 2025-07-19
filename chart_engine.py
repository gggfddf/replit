import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Optional talib import
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

# finta import for technical indicators
try:
    from finta import TA as finta_ta
    FINTA_AVAILABLE = True
except ImportError:
    FINTA_AVAILABLE = False

# ta library import for technical indicators
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

class ChartEngine:
    """
    Professional Chart Engine for generating interactive candlestick charts
    with advanced technical overlays and pattern detection visualization
    """
    
    def __init__(self):
        self.chart_colors = {
            'background': '#0e1117',
            'grid': '#262730',
            'bullish': '#00ff88',
            'bearish': '#ff4444',
            'volume': '#1f77b4',
            'text': '#fafafa',
            'support': '#00ff88',
            'resistance': '#ff4444',
            'pattern': '#ffaa00'
        }
        
    def generate_comprehensive_charts(self, market_data):
        """
        Generate comprehensive chart package with multiple timeframes and overlays
        """
        try:
            if market_data.empty:
                return self.create_default_charts()
            
            charts = {}
            
            # Main candlestick charts for different timeframes
            charts['candlestick_charts'] = {
                '5T': self.create_candlestick_chart(market_data, '5-Minute Analysis'),
                '15T': self.create_candlestick_chart(market_data, '15-Minute Analysis'),
                '1D': self.create_candlestick_chart(market_data, 'Daily Analysis'),
                '1W': self.create_weekly_chart(market_data)
            }
            
            # Technical indicator charts
            charts['indicator_charts'] = {
                'rsi_macd': self.create_rsi_macd_chart(market_data),
                'bollinger_volume': self.create_bollinger_volume_chart(market_data),
                'stochastic': self.create_stochastic_chart(market_data),
                'momentum': self.create_momentum_chart(market_data)
            }
            
            # Pattern detection charts
            charts['pattern_charts'] = {
                'detected_patterns': self.create_pattern_detection_chart(market_data),
                'support_resistance': self.create_support_resistance_chart(market_data),
                'volume_analysis': self.create_volume_analysis_chart(market_data)
            }
            
            return charts
            
        except Exception as e:
            print(f"Chart generation error: {e}")
            return self.create_default_charts()
    
    def create_candlestick_chart(self, data, title):
        """
        Create professional candlestick chart with technical overlays
        """
        try:
            # Create subplot with secondary y-axis for volume
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.2, 0.1],
                subplot_titles=('Price Action', 'Volume', 'RSI')
            )
            
            # Main candlestick chart
            candlestick = go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing=dict(fillcolor=self.chart_colors['bullish'], line=dict(color=self.chart_colors['bullish'])),
                decreasing=dict(fillcolor=self.chart_colors['bearish'], line=dict(color=self.chart_colors['bearish']))
            )
            fig.add_trace(candlestick, row=1, col=1)
            
            # Add moving averages
            sma_20 = data['Close'].rolling(20).mean()
            sma_50 = data['Close'].rolling(50).mean()
            ema_12 = data['Close'].ewm(span=12).mean()
            
            fig.add_trace(go.Scatter(x=data.index, y=sma_20, name='SMA 20', 
                                   line=dict(color='orange', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=sma_50, name='SMA 50', 
                                   line=dict(color='purple', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=ema_12, name='EMA 12', 
                                   line=dict(color='cyan', width=1, dash='dash')), row=1, col=1)
            
            # Add Bollinger Bands
            if TALIB_AVAILABLE:
                upper, middle, lower = talib.BBANDS(data['Close'].values, timeperiod=20)
            else:
                middle = data['Close'].rolling(20).mean()
                std = data['Close'].rolling(20).std()
                upper = middle + (std * 2)
                lower = middle - (std * 2)
            fig.add_trace(go.Scatter(x=data.index, y=upper, name='BB Upper', 
                                   line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=lower, name='BB Lower', 
                                   line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
            
            # Volume bars
            colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' for i in range(len(data))]
            fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', 
                               marker_color=colors, opacity=0.7), row=2, col=1)
            
            # RSI
            if TALIB_AVAILABLE:
                if TALIB_AVAILABLE:
                rsi = talib.RSI(data['Close'].values, timeperiod=14)
            else:
                rsi = self.calculate_rsi(data['Close'], 14)
            else:
                rsi = self.calculate_rsi(data['Close'], 14)
            fig.add_trace(go.Scatter(x=data.index, y=rsi, name='RSI', 
                                   line=dict(color='yellow', width=2)), row=3, col=1)
            
            # RSI overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
            
            # Add support and resistance levels
            self.add_support_resistance_levels(fig, data, row=1)
            
            # Style the chart
            fig.update_layout(
                title=dict(text=title, font=dict(size=20, color=self.chart_colors['text'])),
                template='plotly_dark',
                paper_bgcolor=self.chart_colors['background'],
                plot_bgcolor=self.chart_colors['background'],
                font=dict(color=self.chart_colors['text']),
                xaxis_rangeslider_visible=False,
                height=800,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update x and y axes
            fig.update_xaxes(gridcolor=self.chart_colors['grid'])
            fig.update_yaxes(gridcolor=self.chart_colors['grid'])
            
            return fig
            
        except Exception as e:
            print(f"Candlestick chart creation error: {e}")
            return self.create_empty_chart(title)
    
    def create_weekly_chart(self, data):
        """
        Create weekly candlestick chart
        """
        try:
            # Resample to weekly data
            weekly_data = data.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            return self.create_candlestick_chart(weekly_data, 'Weekly Analysis')
            
        except Exception as e:
            print(f"Weekly chart creation error: {e}")
            return self.create_empty_chart('Weekly Analysis')
    
    def create_rsi_macd_chart(self, data):
        """
        Create RSI and MACD indicator chart
        """
        try:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('RSI (14)', 'MACD (12,26,9)')
            )
            
            # RSI
            if FINTA_AVAILABLE:
                rsi = finta_ta.RSI(data, period=14)
            elif TA_AVAILABLE:
                rsi = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
            elif TALIB_AVAILABLE:
                rsi = talib.RSI(data['Close'].values, timeperiod=14)
            else:
                rsi = self.calculate_rsi(data['Close'], 14)
            fig.add_trace(go.Scatter(x=data.index, y=rsi, name='RSI', 
                                   line=dict(color='yellow', width=2)), row=1, col=1)
            
            # RSI levels
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="orange", row=1, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="orange", row=1, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=1, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=1, col=1)
            
            # MACD
            if TALIB_AVAILABLE:
                macd, signal, histogram = talib.MACD(data['Close'].values)
            else:
                macd, signal, histogram = self.calculate_macd(data['Close'])
            fig.add_trace(go.Scatter(x=data.index, y=macd, name='MACD', 
                                   line=dict(color='blue', width=2)), row=2, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=signal, name='Signal', 
                                   line=dict(color='red', width=2)), row=2, col=1)
            fig.add_trace(go.Bar(x=data.index, y=histogram, name='Histogram', 
                               marker_color='green', opacity=0.7), row=2, col=1)
            
            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
            
            # Style
            fig.update_layout(
                title='RSI & MACD Analysis',
                template='plotly_dark',
                paper_bgcolor=self.chart_colors['background'],
                plot_bgcolor=self.chart_colors['background'],
                font=dict(color=self.chart_colors['text']),
                height=600,
                showlegend=True
            )
            
            fig.update_xaxes(gridcolor=self.chart_colors['grid'])
            fig.update_yaxes(gridcolor=self.chart_colors['grid'])
            
            return fig
            
        except Exception as e:
            print(f"RSI/MACD chart creation error: {e}")
            return self.create_empty_chart('RSI & MACD Analysis')
    
    def create_bollinger_volume_chart(self, data):
        """
        Create Bollinger Bands with Volume chart
        """
        try:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Price with Bollinger Bands', 'Volume Analysis')
            )
            
            # Candlestick with Bollinger Bands
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing=dict(fillcolor=self.chart_colors['bullish'], line=dict(color=self.chart_colors['bullish'])),
                decreasing=dict(fillcolor=self.chart_colors['bearish'], line=dict(color=self.chart_colors['bearish']))
            ), row=1, col=1)
            
            # Bollinger Bands
            if TALIB_AVAILABLE:
                upper, middle, lower = talib.BBANDS(data['Close'].values, timeperiod=20)
            else:
                middle = data['Close'].rolling(20).mean()
                std = data['Close'].rolling(20).std()
                upper = middle + (std * 2)
                lower = middle - (std * 2)
            fig.add_trace(go.Scatter(x=data.index, y=upper, name='BB Upper', 
                                   line=dict(color='purple', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=middle, name='BB Middle', 
                                   line=dict(color='orange', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=lower, name='BB Lower', 
                                   line=dict(color='purple', width=2)), row=1, col=1)
            
            # Volume with moving average
            volume_ma = data['Volume'].rolling(20).mean()
            colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' for i in range(len(data))]
            
            fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', 
                               marker_color=colors, opacity=0.7), row=2, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=volume_ma, name='Volume MA', 
                                   line=dict(color='white', width=2)), row=2, col=1)
            
            # Style
            fig.update_layout(
                title='Bollinger Bands & Volume Analysis',
                template='plotly_dark',
                paper_bgcolor=self.chart_colors['background'],
                plot_bgcolor=self.chart_colors['background'],
                font=dict(color=self.chart_colors['text']),
                height=600,
                showlegend=True,
                xaxis_rangeslider_visible=False
            )
            
            fig.update_xaxes(gridcolor=self.chart_colors['grid'])
            fig.update_yaxes(gridcolor=self.chart_colors['grid'])
            
            return fig
            
        except Exception as e:
            print(f"Bollinger/Volume chart creation error: {e}")
            return self.create_empty_chart('Bollinger Bands & Volume Analysis')
    
    def create_stochastic_chart(self, data):
        """
        Create Stochastic oscillator chart
        """
        try:
            fig = go.Figure()
            
            # Stochastic
            slowk, slowd = talib.STOCH(data['High'].values, data['Low'].values, data['Close'].values)
            
            fig.add_trace(go.Scatter(x=data.index, y=slowk, name='%K', 
                                   line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=data.index, y=slowd, name='%D', 
                                   line=dict(color='red', width=2)))
            
            # Overbought/Oversold levels
            fig.add_hline(y=80, line_dash="dash", line_color="red")
            fig.add_hline(y=20, line_dash="dash", line_color="green")
            fig.add_hline(y=50, line_dash="dot", line_color="gray")
            
            fig.update_layout(
                title='Stochastic Oscillator (%K, %D)',
                template='plotly_dark',
                paper_bgcolor=self.chart_colors['background'],
                plot_bgcolor=self.chart_colors['background'],
                font=dict(color=self.chart_colors['text']),
                height=400,
                yaxis=dict(range=[0, 100])
            )
            
            fig.update_xaxes(gridcolor=self.chart_colors['grid'])
            fig.update_yaxes(gridcolor=self.chart_colors['grid'])
            
            return fig
            
        except Exception as e:
            print(f"Stochastic chart creation error: {e}")
            return self.create_empty_chart('Stochastic Oscillator')
    
    def create_momentum_chart(self, data):
        """
        Create momentum indicators chart
        """
        try:
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('ROC (Rate of Change)', 'Williams %R', 'CCI')
            )
            
            # ROC
            roc = talib.ROC(data['Close'].values, timeperiod=12)
            fig.add_trace(go.Scatter(x=data.index, y=roc, name='ROC', 
                                   line=dict(color='blue', width=2)), row=1, col=1)
            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
            
            # Williams %R
            willr = talib.WILLR(data['High'].values, data['Low'].values, data['Close'].values)
            fig.add_trace(go.Scatter(x=data.index, y=willr, name='Williams %R', 
                                   line=dict(color='red', width=2)), row=2, col=1)
            fig.add_hline(y=-20, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=-80, line_dash="dash", line_color="green", row=2, col=1)
            
            # CCI
            cci = talib.CCI(data['High'].values, data['Low'].values, data['Close'].values)
            fig.add_trace(go.Scatter(x=data.index, y=cci, name='CCI', 
                                   line=dict(color='purple', width=2)), row=3, col=1)
            fig.add_hline(y=100, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=-100, line_dash="dash", line_color="green", row=3, col=1)
            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)
            
            fig.update_layout(
                title='Momentum Indicators',
                template='plotly_dark',
                paper_bgcolor=self.chart_colors['background'],
                plot_bgcolor=self.chart_colors['background'],
                font=dict(color=self.chart_colors['text']),
                height=600
            )
            
            fig.update_xaxes(gridcolor=self.chart_colors['grid'])
            fig.update_yaxes(gridcolor=self.chart_colors['grid'])
            
            return fig
            
        except Exception as e:
            print(f"Momentum chart creation error: {e}")
            return self.create_empty_chart('Momentum Indicators')
    
    def create_pattern_detection_chart(self, data):
        """
        Create chart with pattern detection overlays
        """
        try:
            fig = go.Figure()
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing=dict(fillcolor=self.chart_colors['bullish'], line=dict(color=self.chart_colors['bullish'])),
                decreasing=dict(fillcolor=self.chart_colors['bearish'], line=dict(color=self.chart_colors['bearish']))
            ))
            
            # Add pattern detection markers
            self.add_candlestick_pattern_markers(fig, data)
            
            # Add trend lines
            self.add_trend_lines(fig, data)
            
            fig.update_layout(
                title='Pattern Detection Analysis',
                template='plotly_dark',
                paper_bgcolor=self.chart_colors['background'],
                plot_bgcolor=self.chart_colors['background'],
                font=dict(color=self.chart_colors['text']),
                height=600,
                xaxis_rangeslider_visible=False
            )
            
            fig.update_xaxes(gridcolor=self.chart_colors['grid'])
            fig.update_yaxes(gridcolor=self.chart_colors['grid'])
            
            return fig
            
        except Exception as e:
            print(f"Pattern detection chart creation error: {e}")
            return self.create_empty_chart('Pattern Detection Analysis')
    
    def create_support_resistance_chart(self, data):
        """
        Create chart highlighting support and resistance levels
        """
        try:
            fig = go.Figure()
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ))
            
            # Add comprehensive support/resistance levels
            self.add_comprehensive_support_resistance(fig, data)
            
            fig.update_layout(
                title='Support & Resistance Analysis',
                template='plotly_dark',
                paper_bgcolor=self.chart_colors['background'],
                plot_bgcolor=self.chart_colors['background'],
                font=dict(color=self.chart_colors['text']),
                height=600,
                xaxis_rangeslider_visible=False
            )
            
            return fig
            
        except Exception as e:
            print(f"Support/Resistance chart creation error: {e}")
            return self.create_empty_chart('Support & Resistance Analysis')
    
    def create_volume_analysis_chart(self, data):
        """
        Create detailed volume analysis chart
        """
        try:
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Price & Volume', 'OBV (On Balance Volume)', 'Volume Rate of Change')
            )
            
            # Price and Volume
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ), row=1, col=1)
            
            # Volume bars with color coding
            colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' for i in range(len(data))]
            fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', 
                               marker_color=colors, opacity=0.7, yaxis='y2'), row=1, col=1)
            
            # OBV
            obv = talib.OBV(data['Close'].values, data['Volume'].values)
            fig.add_trace(go.Scatter(x=data.index, y=obv, name='OBV', 
                                   line=dict(color='blue', width=2)), row=2, col=1)
            
            # Volume ROC
            volume_roc = data['Volume'].pct_change(periods=10) * 100
            fig.add_trace(go.Scatter(x=data.index, y=volume_roc, name='Volume ROC', 
                                   line=dict(color='purple', width=2)), row=3, col=1)
            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)
            
            fig.update_layout(
                title='Volume Analysis',
                template='plotly_dark',
                paper_bgcolor=self.chart_colors['background'],
                plot_bgcolor=self.chart_colors['background'],
                font=dict(color=self.chart_colors['text']),
                height=700,
                yaxis2=dict(overlaying='y', side='right')
            )
            
            return fig
            
        except Exception as e:
            print(f"Volume analysis chart creation error: {e}")
            return self.create_empty_chart('Volume Analysis')
    
    def add_support_resistance_levels(self, fig, data, row=1):
        """
        Add support and resistance levels to chart
        """
        try:
            from scipy.signal import find_peaks
            
            # Find peaks and troughs
            highs = data['High'].values
            lows = data['Low'].values
            
            peaks = find_peaks(highs, distance=10, prominence=np.std(highs)*0.3)[0]
            troughs = find_peaks(-lows, distance=10, prominence=np.std(lows)*0.3)[0]
            
            # Add resistance levels (last 3 peaks)
            if len(peaks) > 0:
                for peak in peaks[-3:]:
                    fig.add_hline(
                        y=highs[peak],
                        line_dash="dash",
                        line_color=self.chart_colors['resistance'],
                        line_width=1,
                        opacity=0.7,
                        row=row, col=1
                    )
            
            # Add support levels (last 3 troughs)
            if len(troughs) > 0:
                for trough in troughs[-3:]:
                    fig.add_hline(
                        y=lows[trough],
                        line_dash="dash",
                        line_color=self.chart_colors['support'],
                        line_width=1,
                        opacity=0.7,
                        row=row, col=1
                    )
                    
        except Exception as e:
            print(f"Support/Resistance levels error: {e}")
    
    def add_comprehensive_support_resistance(self, fig, data):
        """
        Add comprehensive support and resistance analysis
        """
        try:
            # Volume-weighted support/resistance
            price_levels = np.concatenate([data['High'].values, data['Low'].values])
            volumes = np.concatenate([data['Volume'].values, data['Volume'].values])
            
            # Find most significant levels
            from sklearn.cluster import KMeans
            if len(price_levels) > 20:
                price_volume = np.column_stack([price_levels, volumes])
                kmeans = KMeans(n_clusters=min(6, len(price_levels)//10), random_state=42)
                clusters = kmeans.fit(price_volume)
                
                for center in clusters.cluster_centers_:
                    level = center[0]
                    volume_weight = center[1]
                    
                    # Color intensity based on volume
                    opacity = min(1.0, volume_weight / np.mean(data['Volume']) * 0.5)
                    
                    fig.add_hline(
                        y=level,
                        line_dash="dot",
                        line_color=self.chart_colors['pattern'],
                        line_width=2,
                        opacity=opacity
                    )
                    
        except Exception as e:
            print(f"Comprehensive S/R error: {e}")
    
    def add_candlestick_pattern_markers(self, fig, data):
        """
        Add candlestick pattern detection markers
        """
        try:
            # Detect some basic patterns and mark them
            for i in range(1, len(data)-1):
                current = data.iloc[i]
                prev = data.iloc[i-1]
                next_candle = data.iloc[i+1]
                
                # Doji pattern detection
                body_size = abs(current['Close'] - current['Open'])
                candle_range = current['High'] - current['Low']
                
                if body_size < candle_range * 0.1 and candle_range > 0:  # Doji
                    fig.add_annotation(
                        x=data.index[i],
                        y=current['High'],
                        text="DOJI",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor=self.chart_colors['pattern'],
                        bgcolor=self.chart_colors['pattern'],
                        bordercolor=self.chart_colors['pattern']
                    )
                
                # Hammer pattern
                lower_shadow = min(current['Open'], current['Close']) - current['Low']
                upper_shadow = current['High'] - max(current['Open'], current['Close'])
                
                if lower_shadow > body_size * 2 and upper_shadow < body_size * 0.5:
                    fig.add_annotation(
                        x=data.index[i],
                        y=current['Low'],
                        text="HAMMER",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor=self.chart_colors['support']
                    )
                    
        except Exception as e:
            print(f"Pattern markers error: {e}")
    
    def add_trend_lines(self, fig, data):
        """
        Add trend lines to the chart
        """
        try:
            from scipy.signal import find_peaks
            
            # Find significant peaks and troughs for trend lines
            highs = data['High'].values
            lows = data['Low'].values
            
            peaks = find_peaks(highs, distance=15)[0]
            troughs = find_peaks(-lows, distance=15)[0]
            
            # Draw trend line connecting last few peaks
            if len(peaks) >= 2:
                x_vals = [data.index[peaks[-2]], data.index[peaks[-1]]]
                y_vals = [highs[peaks[-2]], highs[peaks[-1]]]
                
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines',
                    name='Resistance Trend',
                    line=dict(color=self.chart_colors['resistance'], width=2, dash='dash')
                ))
            
            # Draw trend line connecting last few troughs
            if len(troughs) >= 2:
                x_vals = [data.index[troughs[-2]], data.index[troughs[-1]]]
                y_vals = [lows[troughs[-2]], lows[troughs[-1]]]
                
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines',
                    name='Support Trend',
                    line=dict(color=self.chart_colors['support'], width=2, dash='dash')
                ))
                
        except Exception as e:
            print(f"Trend lines error: {e}")
    
    def create_empty_chart(self, title):
        """
        Create an empty chart as fallback
        """
        fig = go.Figure()
        fig.add_annotation(
            text="Chart data unavailable",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=self.chart_colors['text'])
        )
        
        fig.update_layout(
            title=title,
            template='plotly_dark',
            paper_bgcolor=self.chart_colors['background'],
            plot_bgcolor=self.chart_colors['background'],
            font=dict(color=self.chart_colors['text']),
            height=400
        )
        
        return fig
    
    def create_default_charts(self):
        """
        Create default charts when data is unavailable
        """
        return {
            'candlestick_charts': {
                '5T': self.create_empty_chart('5-Minute Analysis'),
                '15T': self.create_empty_chart('15-Minute Analysis'),
                '1D': self.create_empty_chart('Daily Analysis'),
                '1W': self.create_empty_chart('Weekly Analysis')
            },
            'indicator_charts': {
                'rsi_macd': self.create_empty_chart('RSI & MACD Analysis'),
                'bollinger_volume': self.create_empty_chart('Bollinger Bands & Volume Analysis')
            },
            'pattern_charts': {
                'detected_patterns': self.create_empty_chart('Pattern Detection Analysis'),
                'support_resistance': self.create_empty_chart('Support & Resistance Analysis')
            }
        }
    
    # Helper functions for manual technical indicator calculations
    def calculate_rsi(self, data, period=14):
        """Calculate RSI manually"""
        try:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except:
            return pd.Series([50] * len(data), index=data.index)
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD manually"""
        try:
            ema_fast = data.ewm(span=fast).mean()
            ema_slow = data.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        except:
            zero_series = pd.Series([0] * len(data), index=data.index)
            return zero_series, zero_series, zero_series
    
    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """Calculate Stochastic oscillator manually"""
        try:
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            return k_percent, d_percent
        except:
            zero_series = pd.Series([50] * len(close), index=close.index)
            return zero_series, zero_series
