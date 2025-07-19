import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataPipeline:
    """
    Comprehensive data pipeline for fetching and processing Indian stock market data
    with multi-timeframe synchronization and real-time capabilities
    """
    
    def __init__(self):
        self.data_cache = {}
        self.last_fetch_time = {}
        
    def fetch_comprehensive_data(self, stock_symbol, period="2y"):
        """
        Fetch comprehensive market data for Indian stocks
        """
        try:
            # Ensure proper NSE format
            if not stock_symbol.endswith('.NS'):
                stock_symbol = f"{stock_symbol}.NS"
            
            # Check cache (refresh if older than 5 minutes)
            current_time = datetime.now()
            if (stock_symbol in self.data_cache and 
                stock_symbol in self.last_fetch_time and
                (current_time - self.last_fetch_time[stock_symbol]).seconds < 300):
                return self.data_cache[stock_symbol]
            
            # Fetch data from yfinance
            ticker = yf.Ticker(stock_symbol)
            
            # Get historical data
            hist_data = ticker.history(period=period, interval="1d")
            
            if hist_data.empty:
                print(f"No data found for {stock_symbol}")
                return None
            
            # Clean and process data
            processed_data = self.process_raw_data(hist_data)
            
            # Add multi-timeframe data
            processed_data = self.add_multi_timeframe_data(ticker, processed_data)
            
            # Cache the data
            self.data_cache[stock_symbol] = processed_data
            self.last_fetch_time[stock_symbol] = current_time
            
            return processed_data
            
        except Exception as e:
            print(f"Error fetching data for {stock_symbol}: {e}")
            return None
    
    def process_raw_data(self, raw_data):
        """
        Process and clean raw market data
        """
        try:
            df = raw_data.copy()
            
            # Ensure we have required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    print(f"Missing required column: {col}")
                    return pd.DataFrame()
            
            # Remove any rows with NaN values
            df = df.dropna()
            
            # Ensure data types
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            
            # Remove any remaining NaN values
            df = df.dropna()
            
            # Basic data validation
            df = self.validate_ohlc_data(df)
            
            # Add basic derived columns
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['True_Range'] = np.maximum(
                df['High'] - df['Low'],
                np.maximum(
                    abs(df['High'] - df['Close'].shift(1)),
                    abs(df['Low'] - df['Close'].shift(1))
                )
            )
            
            # Add volume analysis
            df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
            
            # Add volatility measures
            df['Volatility_20'] = df['Returns'].rolling(20).std() * np.sqrt(252)
            
            return df.dropna()
            
        except Exception as e:
            print(f"Error processing raw data: {e}")
            return pd.DataFrame()
    
    def validate_ohlc_data(self, df):
        """
        Validate OHLC data consistency
        """
        try:
            # Check for invalid OHLC relationships
            valid_data = (
                (df['High'] >= df['Open']) &
                (df['High'] >= df['Close']) &
                (df['Low'] <= df['Open']) &
                (df['Low'] <= df['Close']) &
                (df['High'] >= df['Low']) &
                (df['Volume'] >= 0)
            )
            
            df_clean = df[valid_data].copy()
            
            if len(df_clean) < len(df) * 0.95:  # More than 5% data loss
                print("Warning: Significant data quality issues detected")
            
            return df_clean
            
        except Exception as e:
            print(f"Error validating OHLC data: {e}")
            return df
    
    def add_multi_timeframe_data(self, ticker, daily_data):
        """
        Add multi-timeframe analysis data
        """
        try:
            # For now, we'll work primarily with daily data
            # In a production system, you might fetch intraday data separately
            
            # Add weekly and monthly summaries
            weekly_data = self.resample_to_weekly(daily_data)
            monthly_data = self.resample_to_monthly(daily_data)
            
            # Store in the main dataframe as additional info
            daily_data.attrs['weekly_summary'] = weekly_data.tail(52)  # Last 52 weeks
            daily_data.attrs['monthly_summary'] = monthly_data.tail(24)  # Last 24 months
            
            return daily_data
            
        except Exception as e:
            print(f"Error adding multi-timeframe data: {e}")
            return daily_data
    
    def resample_to_weekly(self, daily_data):
        """
        Resample daily data to weekly
        """
        try:
            weekly = daily_data.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            return weekly
            
        except Exception as e:
            print(f"Error resampling to weekly: {e}")
            return pd.DataFrame()
    
    def resample_to_monthly(self, daily_data):
        """
        Resample daily data to monthly
        """
        try:
            monthly = daily_data.resample('M').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            return monthly
            
        except Exception as e:
            print(f"Error resampling to monthly: {e}")
            return pd.DataFrame()
    
    def get_stock_info(self, stock_symbol):
        """
        Get additional stock information
        """
        try:
            if not stock_symbol.endswith('.NS'):
                stock_symbol = f"{stock_symbol}.NS"
            
            ticker = yf.Ticker(stock_symbol)
            info = ticker.info
            
            return {
                'name': info.get('longName', stock_symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'INR'),
                'exchange': info.get('exchange', 'NSE')
            }
            
        except Exception as e:
            print(f"Error getting stock info: {e}")
            return {
                'name': stock_symbol,
                'sector': 'Unknown',
                'industry': 'Unknown',
                'market_cap': 0,
                'currency': 'INR',
                'exchange': 'NSE'
            }
    
    def get_realtime_data(self, stock_symbol):
        """
        Get real-time data (latest available)
        """
        try:
            if not stock_symbol.endswith('.NS'):
                stock_symbol = f"{stock_symbol}.NS"
            
            ticker = yf.Ticker(stock_symbol)
            
            # Get the most recent data
            hist = ticker.history(period="1d", interval="1m")
            
            if not hist.empty:
                latest = hist.iloc[-1]
                return {
                    'current_price': latest['Close'],
                    'open': latest['Open'],
                    'high': latest['High'],
                    'low': latest['Low'],
                    'volume': latest['Volume'],
                    'timestamp': hist.index[-1]
                }
            else:
                # Fallback to daily data
                hist = ticker.history(period="2d", interval="1d")
                if not hist.empty:
                    latest = hist.iloc[-1]
                    return {
                        'current_price': latest['Close'],
                        'open': latest['Open'],
                        'high': latest['High'],
                        'low': latest['Low'],
                        'volume': latest['Volume'],
                        'timestamp': hist.index[-1]
                    }
            
            return None
            
        except Exception as e:
            print(f"Error getting real-time data: {e}")
            return None
    
    def validate_symbol(self, stock_symbol):
        """
        Validate if stock symbol exists and has data
        """
        try:
            if not stock_symbol.endswith('.NS'):
                stock_symbol = f"{stock_symbol}.NS"
            
            ticker = yf.Ticker(stock_symbol)
            hist = ticker.history(period="5d")
            
            return not hist.empty
            
        except:
            return False
    
    def get_market_hours_info(self):
        """
        Get Indian market hours information
        """
        return {
            'market_open': '09:15',
            'market_close': '15:30',
            'timezone': 'Asia/Kolkata',
            'trading_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        }
    
    def is_market_open(self):
        """
        Check if Indian market is currently open
        """
        try:
            import pytz
            from datetime import datetime
            
            ist = pytz.timezone('Asia/Kolkata')
            now = datetime.now(ist)
            
            # Check if it's a weekday
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            # Check market hours (9:15 AM to 3:30 PM IST)
            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            return market_open <= now <= market_close
            
        except:
            # If timezone library not available, assume market might be open
            return True
    
    def get_corporate_actions(self, stock_symbol):
        """
        Get corporate actions data (dividends, splits, etc.)
        """
        try:
            if not stock_symbol.endswith('.NS'):
                stock_symbol = f"{stock_symbol}.NS"
            
            ticker = yf.Ticker(stock_symbol)
            
            # Get dividends and stock splits
            dividends = ticker.dividends
            splits = ticker.splits
            
            return {
                'dividends': dividends.tail(10) if not dividends.empty else pd.Series(),
                'splits': splits.tail(10) if not splits.empty else pd.Series()
            }
            
        except Exception as e:
            print(f"Error getting corporate actions: {e}")
            return {
                'dividends': pd.Series(),
                'splits': pd.Series()
            }
    
    def get_financial_data(self, stock_symbol):
        """
        Get basic financial data
        """
        try:
            if not stock_symbol.endswith('.NS'):
                stock_symbol = f"{stock_symbol}.NS"
            
            ticker = yf.Ticker(stock_symbol)
            info = ticker.info
            
            return {
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'market_cap': info.get('marketCap', 0),
                'book_value': info.get('bookValue', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'profit_margin': info.get('profitMargins', 0),
                'roe': info.get('returnOnEquity', 0),
                'debt_to_equity': info.get('debtToEquity', 0)
            }
            
        except Exception as e:
            print(f"Error getting financial data: {e}")
            return {
                'pe_ratio': 0,
                'pb_ratio': 0,
                'market_cap': 0,
                'book_value': 0,
                'dividend_yield': 0,
                'profit_margin': 0,
                'roe': 0,
                'debt_to_equity': 0
            }
