import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings('ignore')

class MarketUtils:
    """
    Comprehensive Market Utilities for Indian Stock Market operations,
    stock information retrieval, market timing, and general utility functions
    """
    
    def __init__(self):
        self.indian_timezone = pytz.timezone('Asia/Kolkata')
        self.market_holidays = self.get_indian_market_holidays()
        self.stock_cache = {}
        
    def get_stock_info(self, stock_symbol):
        """
        Get comprehensive stock information for Indian stocks
        """
        try:
            # Ensure proper NSE format
            if not stock_symbol.endswith('.NS'):
                stock_symbol = f"{stock_symbol}.NS"
            
            # Check cache first
            if stock_symbol in self.stock_cache:
                cache_time, cached_info = self.stock_cache[stock_symbol]
                if (datetime.now() - cache_time).seconds < 3600:  # 1 hour cache
                    return cached_info
            
            # Fetch from yfinance
            ticker = yf.Ticker(stock_symbol)
            info = ticker.info
            
            # Extract relevant information
            stock_info = {
                'symbol': stock_symbol,
                'name': info.get('longName', info.get('shortName', stock_symbol)),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'currency': info.get('currency', 'INR'),
                'exchange': info.get('exchange', 'NSE'),
                'country': info.get('country', 'India'),
                
                # Valuation metrics
                'pe_ratio': info.get('trailingPE', info.get('forwardPE', 0)),
                'pb_ratio': info.get('priceToBook', 0),
                'ps_ratio': info.get('priceToSalesTrailing12Months', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'ev_revenue': info.get('enterpriseToRevenue', 0),
                'ev_ebitda': info.get('enterpriseToEbitda', 0),
                
                # Financial metrics
                'profit_margin': info.get('profitMargins', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'return_on_assets': info.get('returnOnAssets', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                
                # Dividend information
                'dividend_yield': info.get('dividendYield', 0),
                'dividend_rate': info.get('dividendRate', 0),
                'payout_ratio': info.get('payoutRatio', 0),
                'ex_dividend_date': info.get('exDividendDate', None),
                
                # Trading metrics
                'beta': info.get('beta', 1.0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'average_volume': info.get('averageVolume', 0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'float_shares': info.get('floatShares', 0),
                
                # Business description
                'business_summary': info.get('businessSummary', 'No description available'),
                'website': info.get('website', ''),
                'employees': info.get('fullTimeEmployees', 0),
                
                # Price information
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'previous_close': info.get('previousClose', 0),
                'open_price': info.get('open', 0),
                'day_low': info.get('dayLow', 0),
                'day_high': info.get('dayHigh', 0),
                
                # Additional metrics
                'book_value': info.get('bookValue', 0),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                'held_percent_institutions': info.get('heldPercentInstitutions', 0),
                'held_percent_insiders': info.get('heldPercentInsiders', 0),
                'short_ratio': info.get('shortRatio', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'quick_ratio': info.get('quickRatio', 0),
                'current_ratio': info.get('currentRatio', 0),
                
                # Analyst information
                'analyst_target_price': info.get('targetMeanPrice', 0),
                'recommendation_key': info.get('recommendationKey', 'none'),
                'number_of_analyst_opinions': info.get('numberOfAnalystOpinions', 0)
            }
            
            # Cache the information
            self.stock_cache[stock_symbol] = (datetime.now(), stock_info)
            
            return stock_info
            
        except Exception as e:
            print(f"Error getting stock info for {stock_symbol}: {e}")
            return self.get_default_stock_info(stock_symbol)
    
    def get_default_stock_info(self, stock_symbol):
        """Return default stock info when fetch fails"""
        return {
            'symbol': stock_symbol,
            'name': stock_symbol.replace('.NS', ''),
            'sector': 'Unknown',
            'industry': 'Unknown',
            'market_cap': 0,
            'currency': 'INR',
            'exchange': 'NSE',
            'country': 'India',
            'pe_ratio': 0,
            'pb_ratio': 0,
            'dividend_yield': 0,
            'beta': 1.0,
            'business_summary': 'Information not available',
            'current_price': 0,
            'previous_close': 0
        }
    
    def is_market_open(self):
        """
        Check if Indian stock market is currently open
        """
        try:
            now = datetime.now(self.indian_timezone)
            
            # Check if it's a weekend
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            # Check if it's a market holiday
            today_date = now.date()
            if today_date in self.market_holidays:
                return False
            
            # Check market hours (9:15 AM to 3:30 PM IST)
            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            return market_open <= now <= market_close
            
        except Exception as e:
            print(f"Error checking market status: {e}")
            return False
    
    def get_market_status(self):
        """
        Get detailed market status information
        """
        try:
            now = datetime.now(self.indian_timezone)
            
            status = {
                'is_open': self.is_market_open(),
                'current_time': now.strftime('%Y-%m-%d %H:%M:%S %Z'),
                'next_open': None,
                'next_close': None,
                'market_session': 'Closed'
            }
            
            if status['is_open']:
                # Market is open, calculate next close
                next_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
                status['next_close'] = next_close.strftime('%Y-%m-%d %H:%M:%S %Z')
                status['market_session'] = 'Open'
                
                # Determine session type
                if now.hour < 10:
                    status['market_session'] = 'Opening Session'
                elif now.hour >= 15:
                    status['market_session'] = 'Closing Session'
                else:
                    status['market_session'] = 'Regular Session'
            else:
                # Market is closed, calculate next open
                next_open = self.get_next_market_open(now)
                status['next_open'] = next_open.strftime('%Y-%m-%d %H:%M:%S %Z')
                
                # Determine why market is closed
                if now.weekday() >= 5:
                    status['market_session'] = 'Weekend'
                elif now.date() in self.market_holidays:
                    status['market_session'] = 'Holiday'
                elif now.hour < 9 or (now.hour == 9 and now.minute < 15):
                    status['market_session'] = 'Pre-Market'
                else:
                    status['market_session'] = 'After-Market'
            
            return status
            
        except Exception as e:
            print(f"Error getting market status: {e}")
            return {
                'is_open': False,
                'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'market_session': 'Unknown'
            }
    
    def get_next_market_open(self, current_time):
        """Calculate next market opening time"""
        try:
            # Start with next day if after market hours
            next_day = current_time + timedelta(days=1)
            
            # Find next trading day
            while (next_day.weekday() >= 5 or  # Weekend
                   next_day.date() in self.market_holidays):  # Holiday
                next_day += timedelta(days=1)
            
            # Set to market opening time
            next_open = next_day.replace(hour=9, minute=15, second=0, microsecond=0)
            
            return next_open
            
        except Exception as e:
            print(f"Error calculating next market open: {e}")
            return current_time + timedelta(days=1)
    
    def get_indian_market_holidays(self):
        """
        Get list of Indian stock market holidays for current year
        Note: This is a simplified list. In practice, you'd want to fetch from NSE/BSE APIs
        """
        current_year = datetime.now().year
        
        # Major Indian market holidays (simplified list)
        holidays = [
            datetime(current_year, 1, 26).date(),   # Republic Day
            datetime(current_year, 8, 15).date(),   # Independence Day
            datetime(current_year, 10, 2).date(),   # Gandhi Jayanti
            # Add more holidays as needed
        ]
        
        return holidays
    
    def validate_stock_symbol(self, symbol):
        """
        Validate if a stock symbol is valid for Indian markets
        """
        try:
            # Ensure NSE format
            if not symbol.endswith('.NS'):
                symbol = f"{symbol}.NS"
            
            # Try to fetch basic info
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            
            return not hist.empty, symbol
            
        except Exception as e:
            print(f"Symbol validation error: {e}")
            return False, symbol
    
    def get_sector_info(self, sector_name):
        """
        Get information about a specific sector
        """
        sector_info = {
            'Technology': {
                'description': 'Information Technology and Software companies',
                'major_players': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS'],
                'characteristics': 'Export-oriented, USD revenue exposure, talent-intensive'
            },
            'Banking': {
                'description': 'Banking and Financial Services',
                'major_players': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS'],
                'characteristics': 'Interest rate sensitive, credit growth dependent'
            },
            'Pharmaceuticals': {
                'description': 'Pharmaceutical and Healthcare companies',
                'major_players': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS'],
                'characteristics': 'Regulated industry, export-oriented, R&D intensive'
            },
            'Energy': {
                'description': 'Oil, Gas and Energy companies',
                'major_players': ['RELIANCE.NS', 'ONGC.NS', 'IOC.NS', 'BPCL.NS'],
                'characteristics': 'Commodity price sensitive, government policy dependent'
            },
            'Automobiles': {
                'description': 'Automotive and Auto Components',
                'major_players': ['MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS'],
                'characteristics': 'Cyclical, interest rate sensitive, commodity dependent'
            },
            'FMCG': {
                'description': 'Fast Moving Consumer Goods',
                'major_players': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS'],
                'characteristics': 'Defensive, volume growth dependent, rural exposure'
            }
        }
        
        return sector_info.get(sector_name, {
            'description': 'Sector information not available',
            'major_players': [],
            'characteristics': 'No specific characteristics available'
        })
    
    def calculate_position_in_range(self, current_price, high, low):
        """
        Calculate position of current price in a range
        """
        try:
            if high == low:
                return 0.5
            
            position = (current_price - low) / (high - low)
            return max(0, min(1, position))
            
        except:
            return 0.5
    
    def format_indian_currency(self, amount):
        """
        Format amount in Indian currency style
        """
        try:
            if amount >= 10000000:  # 1 crore
                return f"₹{amount/10000000:.2f} Cr"
            elif amount >= 100000:  # 1 lakh
                return f"₹{amount/100000:.2f} L"
            elif amount >= 1000:
                return f"₹{amount/1000:.2f} K"
            else:
                return f"₹{amount:.2f}"
        except:
            return f"₹{amount}"
    
    def calculate_market_timing_score(self, current_time=None):
        """
        Calculate market timing score based on various factors
        """
        try:
            if current_time is None:
                current_time = datetime.now(self.indian_timezone)
            
            score = 50  # Base score
            
            # Time of day factor
            hour = current_time.hour
            minute = current_time.minute
            
            # Opening session (9:15-10:00) - high volatility
            if hour == 9 and minute >= 15:
                score += 10
            elif hour == 10:
                score += 5
            
            # Mid-day session (11:00-14:00) - stable
            elif 11 <= hour <= 13:
                score += 15
            
            # Closing session (14:30-15:30) - high activity
            elif hour == 14 and minute >= 30:
                score += 10
            elif hour == 15:
                score += 10
            
            # Day of week factor
            weekday = current_time.weekday()
            if weekday == 0:  # Monday
                score += 5  # Fresh start
            elif weekday == 4:  # Friday
                score -= 5  # End of week, position squaring
            
            # Month factor
            month = current_time.month
            if month in [3, 9]:  # Quarter end months
                score -= 5
            elif month in [4, 10]:  # Post quarter months
                score += 5
            
            return max(0, min(100, score))
            
        except Exception as e:
            print(f"Market timing score error: {e}")
            return 50
    
    def get_index_info(self, index_name='NIFTY'):
        """
        Get information about major Indian indices
        """
        indices_info = {
            'NIFTY': {
                'symbol': '^NSEI',
                'name': 'NIFTY 50',
                'description': 'Top 50 companies by market cap on NSE',
                'sector_weightage': {
                    'Financial Services': 30,
                    'Information Technology': 18,
                    'Oil & Gas': 11,
                    'Consumer Goods': 8,
                    'Automobiles': 7
                }
            },
            'SENSEX': {
                'symbol': '^BSESN',
                'name': 'BSE SENSEX',
                'description': 'Top 30 companies by market cap on BSE',
                'sector_weightage': {
                    'Financial Services': 35,
                    'Information Technology': 20,
                    'Oil & Gas': 10,
                    'FMCG': 8,
                    'Healthcare': 6
                }
            },
            'BANKNIFTY': {
                'symbol': '^NSEBANK',
                'name': 'BANK NIFTY',
                'description': 'Banking sector index',
                'sector_weightage': {
                    'Private Banks': 70,
                    'Public Banks': 25,
                    'NBFCs': 5
                }
            }
        }
        
        return indices_info.get(index_name.upper(), {
            'symbol': index_name,
            'name': index_name,
            'description': 'Index information not available'
        })
    
    def calculate_beta(self, stock_returns, market_returns):
        """
        Calculate beta of a stock relative to market
        """
        try:
            # Align the series
            aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
            
            if len(aligned_data) < 30:  # Need sufficient data
                return 1.0
            
            stock_ret = aligned_data.iloc[:, 0]
            market_ret = aligned_data.iloc[:, 1]
            
            # Calculate beta using covariance method
            covariance = np.cov(stock_ret, market_ret)[0, 1]
            market_variance = np.var(market_ret)
            
            beta = covariance / market_variance if market_variance != 0 else 1.0
            
            return max(0, min(3, beta))  # Cap beta between 0 and 3
            
        except Exception as e:
            print(f"Beta calculation error: {e}")
            return 1.0
    
    def get_earnings_calendar(self, stock_symbol):
        """
        Get earnings calendar information (simplified)
        """
        try:
            # This would typically connect to an earnings calendar API
            # For now, providing a template structure
            
            return {
                'next_earnings_date': None,
                'last_earnings_date': None,
                'earnings_frequency': 'Quarterly',
                'estimate_eps': None,
                'actual_eps': None,
                'earnings_surprise': None
            }
            
        except Exception as e:
            print(f"Earnings calendar error: {e}")
            return {}
    
    def calculate_risk_metrics(self, returns_series, benchmark_returns=None):
        """
        Calculate comprehensive risk metrics
        """
        try:
            if len(returns_series) < 30:
                return {}
            
            returns = returns_series.dropna()
            
            metrics = {
                'volatility_daily': returns.std(),
                'volatility_annual': returns.std() * np.sqrt(252),
                'var_95': np.percentile(returns, 5),
                'var_99': np.percentile(returns, 1),
                'max_drawdown': self.calculate_max_drawdown(returns),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'positive_periods': (returns > 0).sum() / len(returns),
                'sharpe_ratio': self.calculate_sharpe_ratio(returns),
                'sortino_ratio': self.calculate_sortino_ratio(returns)
            }
            
            if benchmark_returns is not None and len(benchmark_returns) >= len(returns):
                aligned_bench = benchmark_returns.tail(len(returns))
                metrics['beta'] = self.calculate_beta(returns, aligned_bench)
                metrics['alpha'] = self.calculate_alpha(returns, aligned_bench)
                metrics['correlation'] = returns.corr(aligned_bench)
            
            return metrics
            
        except Exception as e:
            print(f"Risk metrics calculation error: {e}")
            return {}
    
    def calculate_max_drawdown(self, returns_series):
        """Calculate maximum drawdown from returns"""
        try:
            cumulative = (1 + returns_series).cumprod()
            peak = cumulative.expanding().max()
            drawdown = (cumulative - peak) / peak
            return drawdown.min()
        except:
            return 0
    
    def calculate_sharpe_ratio(self, returns_series, risk_free_rate=0.06):
        """Calculate Sharpe ratio"""
        try:
            excess_return = returns_series.mean() * 252 - risk_free_rate
            volatility = returns_series.std() * np.sqrt(252)
            return excess_return / volatility if volatility != 0 else 0
        except:
            return 0
    
    def calculate_sortino_ratio(self, returns_series, risk_free_rate=0.06):
        """Calculate Sortino ratio"""
        try:
            excess_return = returns_series.mean() * 252 - risk_free_rate
            downside_deviation = returns_series[returns_series < 0].std() * np.sqrt(252)
            return excess_return / downside_deviation if downside_deviation != 0 else 0
        except:
            return 0
    
    def calculate_alpha(self, stock_returns, market_returns, risk_free_rate=0.06):
        """Calculate alpha"""
        try:
            beta = self.calculate_beta(stock_returns, market_returns)
            stock_return_annual = stock_returns.mean() * 252
            market_return_annual = market_returns.mean() * 252
            
            alpha = stock_return_annual - (risk_free_rate + beta * (market_return_annual - risk_free_rate))
            return alpha
        except:
            return 0
    
    def get_market_breadth_info(self):
        """
        Get market breadth information
        """
        try:
            # This would typically fetch real market breadth data
            # Providing template structure
            
            return {
                'advances': None,
                'declines': None,
                'unchanged': None,
                'advance_decline_ratio': None,
                'new_highs': None,
                'new_lows': None,
                'high_low_ratio': None,
                'up_volume': None,
                'down_volume': None,
                'volume_ratio': None
            }
            
        except Exception as e:
            print(f"Market breadth error: {e}")
            return {}
    
    def convert_timeframe(self, data, target_timeframe):
        """
        Convert data to different timeframes
        """
        try:
            if target_timeframe.upper() == 'W':
                # Weekly conversion
                return data.resample('W').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            elif target_timeframe.upper() == 'M':
                # Monthly conversion
                return data.resample('M').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            else:
                return data
                
        except Exception as e:
            print(f"Timeframe conversion error: {e}")
            return data
