import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from models.deep_learning_engine import DeepLearningEngine
from models.pattern_discovery import PatternDiscoveryEngine
from models.technical_indicators import TechnicalIndicatorEngine
from models.prediction_engine import PredictionEngine
from data.data_pipeline import DataPipeline
from analysis.technical_analysis import TechnicalAnalyzer
from analysis.price_action_analysis import PriceActionAnalyzer
from visualization.chart_engine import ChartEngine
from reports.report_generator import ReportGenerator
from utils.market_utils import MarketUtils

# Page configuration
st.set_page_config(
    page_title="🚀 Advanced Stock Market Intelligence System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional trading interface
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #00ff88;
    }
    .prediction-box {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #00ff88;
    }
    .confidence-high { color: #00ff88; }
    .confidence-medium { color: #ffaa00; }
    .confidence-low { color: #ff4444; }
</style>
""", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 ULTIMATE DEEP LEARNING STOCK MARKET ANALYSIS SYSTEM</h1>
        <p>Autonomous AI-Powered Market Intelligence for Indian Stocks</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for stock selection
    with st.sidebar:
        st.header("🎯 Stock Selection")
        
        # Stock symbol input (only editable parameter)
        stock_symbol = st.text_input(
            "Enter Stock Symbol (NSE Format)",
            value="RELIANCE.NS",
            help="Format: STOCKNAME.NS (e.g., RELIANCE.NS, TCS.NS, INFY.NS)"
        ).upper()
        
        # Analysis trigger
        analyze_button = st.button("🔍 Execute Deep Analysis", type="primary")
        
        st.markdown("---")
        st.markdown("""
        ### 🧠 System Features
        - **Deep Learning Engine**: LSTM + CNN + Transformers
        - **Autonomous Pattern Discovery**
        - **40+ Advanced Technical Indicators**
        - **ML-Powered Predictions**
        - **Real-time Market Intelligence**
        """)
    
    # Initialize system components
    if analyze_button and stock_symbol:
        with st.spinner("🚀 Initializing AI Market Intelligence System..."):
            try:
                # Initialize all system components
                data_pipeline = DataPipeline()
                deep_learning_engine = DeepLearningEngine()
                pattern_discovery = PatternDiscoveryEngine()
                technical_engine = TechnicalIndicatorEngine()
                prediction_engine = PredictionEngine()
                technical_analyzer = TechnicalAnalyzer()
                price_action_analyzer = PriceActionAnalyzer()
                chart_engine = ChartEngine()
                report_generator = ReportGenerator()
                market_utils = MarketUtils()
                
                # Fetch and process data
                st.info("📊 Fetching live market data and historical analysis...")
                market_data = data_pipeline.fetch_comprehensive_data(stock_symbol)
                
                if market_data is None or market_data.empty:
                    st.error("❌ Unable to fetch data for the specified stock symbol. Please check the symbol format.")
                    return
                
                # Display stock information
                stock_info = market_utils.get_stock_info(stock_symbol)
                display_stock_overview(stock_info, market_data)
                
                # Main analysis tabs
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "🧠 Deep Learning Analysis",
                    "📊 Technical Analysis Report", 
                    "📈 Price Action Analysis Report",
                    "🎯 ML Predictions",
                    "📉 Advanced Charts",
                    "📋 Comprehensive Reports"
                ])
                
                with tab1:
                    display_deep_learning_analysis(deep_learning_engine, pattern_discovery, market_data)
                
                with tab2:
                    display_technical_analysis(technical_analyzer, technical_engine, market_data)
                
                with tab3:
                    display_price_action_analysis(price_action_analyzer, pattern_discovery, market_data)
                
                with tab4:
                    display_ml_predictions(prediction_engine, market_data)
                
                with tab5:
                    display_advanced_charts(chart_engine, market_data)
                
                with tab6:
                    display_comprehensive_reports(report_generator, market_data, stock_symbol)
                    
            except Exception as e:
                st.error(f"❌ System Error: {str(e)}")
                st.info("Please try again with a valid NSE stock symbol (e.g., RELIANCE.NS)")

def display_stock_overview(stock_info, market_data):
    """Display stock overview and current market status"""
    st.subheader("📊 Stock Overview & Market Status")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    current_price = market_data['Close'].iloc[-1]
    prev_close = market_data['Close'].iloc[-2]
    price_change = current_price - prev_close
    price_change_pct = (price_change / prev_close) * 100
    
    with col1:
        st.metric("Current Price", f"₹{current_price:.2f}", f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
    
    with col2:
        st.metric("Volume", f"{market_data['Volume'].iloc[-1]:,.0f}")
    
    with col3:
        high_52w = market_data['High'].rolling(252).max().iloc[-1]
        st.metric("52W High", f"₹{high_52w:.2f}")
    
    with col4:
        low_52w = market_data['Low'].rolling(252).min().iloc[-1]
        st.metric("52W Low", f"₹{low_52w:.2f}")
    
    with col5:
        volatility = market_data['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
        st.metric("Volatility (20D)", f"{volatility:.1f}%")

def display_deep_learning_analysis(deep_learning_engine, pattern_discovery, market_data):
    """Display deep learning analysis results"""
    st.subheader("🧠 Deep Learning Market Intelligence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔮 Neural Network Ensemble Analysis")
        
        # Train and analyze with deep learning models
        with st.spinner("Training neural network ensemble..."):
            dl_results = deep_learning_engine.analyze_market(market_data)
            
        st.markdown("#### Model Performance Metrics")
        for model_name, metrics in dl_results['model_performance'].items():
            st.write(f"**{model_name}**: Accuracy {metrics['accuracy']:.2f}%, Loss: {metrics['loss']:.4f}")
        
        st.markdown("#### Feature Importance (Top 10)")
        feature_importance = dl_results['feature_importance']
        for feature, importance in feature_importance.items():
            st.write(f"• **{feature}**: {importance:.3f}")
    
    with col2:
        st.markdown("### 🔍 Autonomous Pattern Discovery")
        
        # Discover patterns autonomously
        with st.spinner("Discovering market patterns..."):
            patterns = pattern_discovery.discover_patterns(market_data)
        
        st.markdown("#### Newly Discovered Patterns")
        for pattern in patterns['discovered_patterns']:
            confidence_class = "confidence-high" if pattern['confidence'] > 0.7 else "confidence-medium" if pattern['confidence'] > 0.5 else "confidence-low"
            st.markdown(f"""
            <div class="metric-card">
                <strong>{pattern['name']}</strong><br>
                Confidence: <span class="{confidence_class}">{pattern['confidence']:.2f}</span><br>
                Occurrences: {pattern['occurrences']}<br>
                Success Rate: {pattern['success_rate']:.1f}%
            </div>
            """, unsafe_allow_html=True)

def display_technical_analysis(technical_analyzer, technical_engine, market_data):
    """Display comprehensive technical analysis report"""
    st.subheader("📊 Technical Analysis Intelligence Report")
    
    # Generate technical analysis
    with st.spinner("Computing 40+ advanced technical indicators..."):
        tech_analysis = technical_analyzer.generate_comprehensive_analysis(market_data)
        indicators = technical_engine.calculate_all_indicators(market_data)
    
    # Technical Summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📈 Trend Analysis")
        trend_strength = tech_analysis['trend_strength']
        trend_direction = tech_analysis['trend_direction']
        
        st.markdown(f"""
        <div class="metric-card">
            <strong>Primary Trend</strong>: {trend_direction}<br>
            <strong>Strength</strong>: {trend_strength}/10<br>
            <strong>Confidence</strong>: {tech_analysis['trend_confidence']:.1f}%
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Momentum Indicators")
        for indicator, value in tech_analysis['momentum_signals'].items():
            signal_class = "confidence-high" if value['signal'] == 'Strong Buy' else "confidence-medium" if 'Buy' in value['signal'] else "confidence-low"
            st.markdown(f"""
            <div class="metric-card">
                <strong>{indicator}</strong>: <span class="{signal_class}">{value['signal']}</span><br>
                Value: {value['value']:.2f}
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("### 🎯 Support & Resistance")
        levels = tech_analysis['support_resistance']
        st.markdown(f"""
        <div class="metric-card">
            <strong>Resistance Levels</strong>:<br>
            R1: ₹{levels['resistance'][0]:.2f}<br>
            R2: ₹{levels['resistance'][1]:.2f}<br>
            <strong>Support Levels</strong>:<br>
            S1: ₹{levels['support'][0]:.2f}<br>
            S2: ₹{levels['support'][1]:.2f}
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed indicator analysis
    st.markdown("### 📋 Comprehensive Indicator Analysis")
    
    # Create indicator summary table
    indicator_df = pd.DataFrame([
        {
            'Indicator': name,
            'Current Value': data['current_value'],
            'Signal': data['signal'],
            'Pattern': data['pattern_detected'],
            'Confidence': f"{data['confidence']:.1f}%"
        }
        for name, data in indicators.items()
    ])
    
    st.dataframe(indicator_df, use_container_width=True)

def display_price_action_analysis(price_action_analyzer, pattern_discovery, market_data):
    """Display comprehensive price action analysis report"""
    st.subheader("📈 Price Action Analysis Intelligence Report")
    
    # Generate price action analysis
    with st.spinner("Analyzing price action patterns and market structure..."):
        price_analysis = price_action_analyzer.analyze_price_action(market_data)
        candlestick_patterns = pattern_discovery.detect_candlestick_patterns(market_data)
        chart_patterns = pattern_discovery.detect_chart_patterns(market_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🕯️ Candlestick Pattern Analysis")
        
        st.markdown("#### Traditional Patterns Detected")
        for pattern in candlestick_patterns['traditional_patterns']:
            reliability_class = "confidence-high" if pattern['reliability'] > 70 else "confidence-medium" if pattern['reliability'] > 50 else "confidence-low"
            st.markdown(f"""
            <div class="metric-card">
                <strong>{pattern['name']}</strong><br>
                Type: {pattern['type']}<br>
                Reliability: <span class="{reliability_class}">{pattern['reliability']:.1f}%</span><br>
                Timeframe: {pattern['timeframe']}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### ML-Discovered Patterns")
        for pattern in candlestick_patterns['ml_discovered_patterns']:
            st.markdown(f"""
            <div class="metric-card">
                <strong>{pattern['custom_name']}</strong><br>
                Formation: {pattern['formation_type']}<br>
                Success Rate: {pattern['historical_success']:.1f}%<br>
                Occurrences: {pattern['total_occurrences']}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Chart Pattern Recognition")
        
        for pattern in chart_patterns['detected_patterns']:
            completion_class = "confidence-high" if pattern['completion_probability'] > 0.7 else "confidence-medium" if pattern['completion_probability'] > 0.5 else "confidence-low"
            st.markdown(f"""
            <div class="metric-card">
                <strong>{pattern['pattern_type']}</strong><br>
                Stage: {pattern['formation_stage']}<br>
                Completion Probability: <span class="{completion_class}">{pattern['completion_probability']:.2f}</span><br>
                Breakout Direction: {pattern['expected_breakout']}<br>
                Target: ₹{pattern['price_target']:.2f}
            </div>
            """, unsafe_allow_html=True)
    
    # Volume-Price Analysis
    st.markdown("### 📊 Volume-Price Relationship Analysis")
    
    volume_analysis = price_analysis['volume_price_analysis']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Volume Trend</strong>: {volume_analysis['volume_trend']}<br>
            <strong>Price-Volume Correlation</strong>: {volume_analysis['correlation']:.3f}<br>
            <strong>Accumulation/Distribution</strong>: {volume_analysis['accumulation_distribution']}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Volume Breakouts</strong>: {volume_analysis['volume_breakouts']}<br>
            <strong>Average Volume (20D)</strong>: {volume_analysis['avg_volume_20d']:,.0f}<br>
            <strong>Volume Ratio</strong>: {volume_analysis['volume_ratio']:.2f}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Volume Confirmation</strong>: {volume_analysis['volume_confirmation']}<br>
            <strong>Buying Pressure</strong>: {volume_analysis['buying_pressure']:.1f}%<br>
            <strong>Selling Pressure</strong>: {volume_analysis['selling_pressure']:.1f}%
        </div>
        """, unsafe_allow_html=True)

def display_ml_predictions(prediction_engine, market_data):
    """Display ML-powered predictions with confidence scores"""
    st.subheader("🎯 Machine Learning Predictions & Intelligence")
    
    # Generate predictions
    with st.spinner("Generating ML-powered market predictions..."):
        predictions = prediction_engine.generate_predictions(market_data)
    
    # Main prediction display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔮 Primary Market Prediction")
        
        main_prediction = predictions['primary_prediction']
        confidence_class = "confidence-high" if main_prediction['confidence'] > 0.75 else "confidence-medium" if main_prediction['confidence'] > 0.5 else "confidence-low"
        
        st.markdown(f"""
        <div class="prediction-box">
            <h3>Direction: {main_prediction['direction']}</h3>
            <p><strong>Probability</strong>: <span class="{confidence_class}">{main_prediction['probability']:.1f}%</span></p>
            <p><strong>Target Price Range</strong>: ₹{main_prediction['target_range']['low']:.2f} - ₹{main_prediction['target_range']['high']:.2f}</p>
            <p><strong>Time Horizon</strong>: {main_prediction['time_horizon']}</p>
            <p><strong>Confidence Level</strong>: <span class="{confidence_class}">{main_prediction['confidence_level']} ({main_prediction['confidence']:.1f}%)</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature importance for prediction
        st.markdown("### 🔍 Prediction Feature Importance")
        feature_importance = predictions['feature_importance']
        
        importance_df = pd.DataFrame([
            {'Feature': feature, 'Importance': f"{importance:.3f}", 'Contribution': f"{importance*100:.1f}%"}
            for feature, importance in feature_importance.items()
        ])
        st.dataframe(importance_df, use_container_width=True)
    
    with col2:
        st.markdown("### ⚠️ Risk Assessment")
        
        risk_assessment = predictions['risk_assessment']
        risk_class = "confidence-low" if risk_assessment['risk_level'] == 'High' else "confidence-medium" if risk_assessment['risk_level'] == 'Medium' else "confidence-high"
        
        st.markdown(f"""
        <div class="metric-card">
            <strong>Risk Level</strong>: <span class="{risk_class}">{risk_assessment['risk_level']}</span><br>
            <strong>Volatility Risk</strong>: {risk_assessment['volatility_risk']:.1f}%<br>
            <strong>Stop Loss</strong>: ₹{risk_assessment['stop_loss']:.2f}<br>
            <strong>Risk-Reward Ratio</strong>: {risk_assessment['risk_reward_ratio']:.2f}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Scenario Analysis")
        scenarios = predictions['scenario_analysis']
        
        for scenario_name, scenario_data in scenarios.items():
            st.markdown(f"""
            <div class="metric-card">
                <strong>{scenario_name}</strong><br>
                Price: ₹{scenario_data['price']:.2f}<br>
                Probability: {scenario_data['probability']:.1f}%
            </div>
            """, unsafe_allow_html=True)
    
    # Pattern trigger conditions
    st.markdown("### 🎯 Pattern Trigger Conditions")
    
    trigger_conditions = predictions['trigger_conditions']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Entry Conditions")
        for condition in trigger_conditions['entry_conditions']:
            st.write(f"• {condition}")
    
    with col2:
        st.markdown("#### Confirmation Signals")
        for signal in trigger_conditions['confirmation_signals']:
            st.write(f"• {signal}")
    
    with col3:
        st.markdown("#### Exit Conditions")
        for condition in trigger_conditions['exit_conditions']:
            st.write(f"• {condition}")

def display_advanced_charts(chart_engine, market_data):
    """Display professional candlestick charts with advanced features"""
    st.subheader("📉 Professional Candlestick Chart Analysis")
    
    # Generate advanced charts
    with st.spinner("Generating professional trading charts..."):
        charts = chart_engine.generate_comprehensive_charts(market_data)
    
    # Multi-timeframe analysis
    timeframes = ['5T', '15T', '1D', '1W']
    timeframe_names = ['5-Minute', '15-Minute', 'Daily', 'Weekly']
    
    selected_timeframe = st.selectbox(
        "Select Timeframe Analysis",
        options=timeframe_names,
        index=2  # Default to Daily
    )
    
    timeframe_key = timeframes[timeframe_names.index(selected_timeframe)]
    
    # Main candlestick chart with overlays
    fig = charts['candlestick_charts'][timeframe_key]
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical indicator charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 RSI & MACD Analysis")
        rsi_macd_fig = charts['indicator_charts']['rsi_macd']
        st.plotly_chart(rsi_macd_fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Bollinger Bands & Volume")
        bb_volume_fig = charts['indicator_charts']['bollinger_volume']
        st.plotly_chart(bb_volume_fig, use_container_width=True)
    
    # Pattern detection overlays
    st.markdown("### 🔍 Pattern Detection Visualization")
    pattern_fig = charts['pattern_charts']['detected_patterns']
    st.plotly_chart(pattern_fig, use_container_width=True)

def display_comprehensive_reports(report_generator, market_data, stock_symbol):
    """Display and generate comprehensive reports"""
    st.subheader("📋 Comprehensive Analysis Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Excel Report", type="primary"):
            with st.spinner("Generating comprehensive Excel report..."):
                excel_file = report_generator.generate_excel_report(market_data, stock_symbol)
                
                with open(excel_file, "rb") as file:
                    st.download_button(
                        label="Download Excel Report",
                        data=file.read(),
                        file_name=f"{stock_symbol}_analysis_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    
    with col2:
        if st.button("📄 Generate PDF Reports", type="primary"):
            with st.spinner("Generating PDF analysis reports..."):
                pdf_files = report_generator.generate_pdf_reports(market_data, stock_symbol)
                
                for report_name, pdf_file in pdf_files.items():
                    with open(pdf_file, "rb") as file:
                        st.download_button(
                            label=f"Download {report_name} PDF",
                            data=file.read(),
                            file_name=f"{stock_symbol}_{report_name.lower().replace(' ', '_')}_report.pdf",
                            mime="application/pdf"
                        )
    
    with col3:
        if st.button("📁 Generate JSON Export", type="primary"):
            with st.spinner("Generating JSON data export..."):
                json_file = report_generator.generate_json_export(market_data, stock_symbol)
                
                with open(json_file, "rb") as file:
                    st.download_button(
                        label="Download JSON Export",
                        data=file.read(),
                        file_name=f"{stock_symbol}_data_export.json",
                        mime="application/json"
                    )
    
    # Executive Summary
    st.markdown("### 📋 Executive Summary")
    
    summary = report_generator.generate_executive_summary(market_data, stock_symbol)
    
    st.markdown(f"""
    <div class="prediction-box">
        <h4>Key Insights & Recommendations</h4>
        <p><strong>Overall Rating</strong>: {summary['overall_rating']}</p>
        <p><strong>Primary Recommendation</strong>: {summary['primary_recommendation']}</p>
        <p><strong>Risk Level</strong>: {summary['risk_level']}</p>
        <p><strong>Time Horizon</strong>: {summary['time_horizon']}</p>
        
        <h5>Key Findings:</h5>
        <ul>
            {''.join([f'<li>{finding}</li>' for finding in summary['key_findings']])}
        </ul>
        
        <h5>Action Items:</h5>
        <ul>
            {''.join([f'<li>{action}</li>' for action in summary['action_items']])}
        </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
