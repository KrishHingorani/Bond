"""
Sector Rotation Strategy Dashboard - Streamlit Application
========================================================

Interactive Streamlit application for mean reversion strategies across:

Bond Sector Strategies:
- HYG/TLT: High Yield vs Long Treasuries
- LQD/IEF: Investment Grade vs Intermediate Treasuries  
- JNK/TLT: Junk Bonds vs Long Treasuries
- AGG/TLT: Aggregate Bonds vs Long Treasuries

Geographic Sector Strategies:
- SPY/EFA: US vs Developed International
- SPY/EEM: US vs Emerging Markets
- EFA/EEM: Developed vs Emerging
- And 7 additional geographic pairs

Author: Research Team
Date: September 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sector Rotation Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Bond strategy configurations
BOND_STRATEGIES = {
    'HYG_TLT': {
        'symbols': ['HYG', 'TLT'],
        'name': 'High Yield vs Long Treasuries',
        'description': 'Credit risk strategy: High yield corporate bonds vs long-term government bonds',
        'color': '#FF6B6B'
    },
    'LQD_IEF': {
        'symbols': ['LQD', 'IEF'],
        'name': 'Investment Grade vs Intermediate Treasuries',
        'description': 'Credit quality strategy: Investment grade corporates vs intermediate treasuries',
        'color': '#4ECDC4'
    },
    'JNK_TLT': {
        'symbols': ['JNK', 'TLT'],
        'name': 'Junk Bonds vs Long Treasuries',
        'description': 'High risk credit strategy: Junk bonds vs long-term treasuries',
        'color': '#45B7D1'
    },
    'AGG_TLT': {
        'symbols': ['AGG', 'TLT'],
        'name': 'Aggregate Bonds vs Long Treasuries',
        'description': 'Broad bond strategy: Total bond market vs long-term treasuries',
        'color': '#96CEB4'
    }
}

# Geographic strategy configurations (excluding VEJ - delisted)
GEOGRAPHIC_STRATEGIES = {
    'SPY_EFA': {
        'symbols': ['SPY', 'EFA'],
        'name': 'US vs Developed International',
        'description': 'US Large Cap vs EAFE Developed Markets',
        'color': '#2E86AB'
    },
    'SPY_EEM': {
        'symbols': ['SPY', 'EEM'],
        'name': 'US vs Emerging Markets',
        'description': 'US Large Cap vs Emerging Markets',
        'color': '#A23B72'
    },
    'EFA_EEM': {
        'symbols': ['EFA', 'EEM'],
        'name': 'Developed vs Emerging',
        'description': 'EAFE Developed vs Emerging Markets',
        'color': '#F18F01'
    },
    'QQQ_VGK': {
        'symbols': ['QQQ', 'VGK'],
        'name': 'US Tech vs Europe',
        'description': 'US Tech/Growth vs Europe',
        'color': '#C73E1D'
    },
    'SPY_VGK': {
        'symbols': ['SPY', 'VGK'],
        'name': 'US vs Europe',
        'description': 'US Large Cap vs Europe',
        'color': '#6A994E'
    },
    'EEM_VGK': {
        'symbols': ['EEM', 'VGK'],
        'name': 'Emerging vs Europe',
        'description': 'Emerging Markets vs Europe',
        'color': '#8E44AD'
    },
    'FXI_EWJ': {
        'symbols': ['FXI', 'EWJ'],
        'name': 'China vs Japan',
        'description': 'China Large Cap vs Japan',
        'color': '#E74C3C'
    },
    'VTI_VEA': {
        'symbols': ['VTI', 'VEA'],
        'name': 'US Total vs Developed',
        'description': 'US Total Market vs Developed Markets',
        'color': '#3498DB'
    },
    'IWM_EFA': {
        'symbols': ['IWM', 'EFA'],
        'name': 'US Small Cap vs EAFE',
        'description': 'US Small Cap vs EAFE Developed',
        'color': '#1ABC9C'
    },
    'SPY_FXI': {
        'symbols': ['SPY', 'FXI'],
        'name': 'US vs China',
        'description': 'US Large Cap vs China Large Cap',
        'color': '#F39C12'
    }
}

# Factor ETF strategy configurations
FACTOR_STRATEGIES = {
    'VUG_VYM': {
        'symbols': ['VUG', 'VYM'],
        'name': 'Growth vs Dividend',
        'description': 'Vanguard Growth ETF vs Vanguard High Dividend Yield ETF',
        'color': '#E74C3C'
    },
    'VFVA_VUG': {
        'symbols': ['VFVA', 'VUG'],
        'name': 'Value vs Growth Factor',
        'description': 'Vanguard U.S. Value Factor ETF vs Vanguard Growth ETF',
        'color': '#9B59B6'
    },
    'VFQY_SPMO': {
        'symbols': ['VFQY', 'SPMO'],
        'name': 'Quality vs Momentum Factor',
        'description': 'Vanguard U.S. Quality Factor ETF vs Invesco S&P 500 Momentum ETF',
        'color': '#3498DB'
    },
    'SIZE_VUG': {
        'symbols': ['SIZE', 'VUG'],
        'name': 'Size vs Growth Factor',
        'description': 'iShares MSCI USA Size Factor ETF vs Vanguard Growth ETF',
        'color': '#E67E22'
    },
    'VFVA_VFQY': {
        'symbols': ['VFVA', 'VFQY'],
        'name': 'Value vs Quality Factor',
        'description': 'Vanguard U.S. Value Factor ETF vs Vanguard U.S. Quality Factor ETF',
        'color': '#27AE60'
    },
    'VYM_SPMO': {
        'symbols': ['VYM', 'SPMO'],
        'name': 'Dividend vs Momentum',
        'description': 'Vanguard High Dividend Yield ETF vs Invesco S&P 500 Momentum ETF',
        'color': '#F39C12'
    },
    'SIZE_VYM': {
        'symbols': ['SIZE', 'VYM'],
        'name': 'Size vs Dividend Factor',
        'description': 'iShares MSCI USA Size Factor ETF vs Vanguard High Dividend Yield ETF',
        'color': '#1ABC9C'
    },
    'VFQY_VUG': {
        'symbols': ['VFQY', 'VUG'],
        'name': 'Quality vs Growth Factor',
        'description': 'Vanguard U.S. Quality Factor ETF vs Vanguard Growth ETF',
        'color': '#34495E'
    }
}

class BondStrategyEngine:
    """Bond strategy calculation engine"""
    
    def __init__(self, lookback_years=5):
        self.lookback_years = lookback_years
        
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def fetch_data(_self, symbols):
        """Fetch market data with caching"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=_self.lookback_years * 365)
            
            with st.spinner(f"Fetching data for {', '.join(symbols)}..."):
                data = yf.download(symbols, start=start_date, end=end_date, progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                data = data['Close']
            elif isinstance(data, pd.Series):
                data = data.to_frame(symbols[0])
            
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
    
    def optimize_parameters(self, asset1_symbol, asset2_symbol, 
                           start_date='2021-01-01', end_date='2024-12-31'):
        """
        Optimize strategy parameters using Sharpe ratio on in-sample data (2021-2024)
        """
        try:
            # Parameter grids for optimization
            windows = [20, 40, 60, 90, 120, 180, 252]
            entry_thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
            exit_thresholds = [0.1, 0.3, 0.5, 0.7, 1.0]
            
            # Fetch historical data for optimization period
            symbols = [asset1_symbol, asset2_symbol]
            data = yf.download(symbols, start=start_date, end=end_date, progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                data = data['Close']
            
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            if data.empty or asset1_symbol not in data.columns or asset2_symbol not in data.columns:
                return {'error': 'Data not available for optimization period'}
            
            price1 = data[asset1_symbol].dropna()
            price2 = data[asset2_symbol].dropna()
            
            best_sharpe = -np.inf
            best_params = None
            best_in_sample_metrics = None
            results = []
            
            total_combinations = len(windows) * len(entry_thresholds) * len(exit_thresholds)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            combination_count = 0
            
            for window in windows:
                for entry_thresh in entry_thresholds:
                    for exit_thresh in exit_thresholds:
                        combination_count += 1
                        progress = combination_count / total_combinations
                        progress_bar.progress(progress)
                        status_text.text(f"Testing combination {combination_count}/{total_combinations}: window={window}, entry={entry_thresh}, exit={exit_thresh}")
                        
                        try:
                            # Calculate strategy for this parameter combination
                            common_dates = price1.index.intersection(price2.index)
                            if len(common_dates) < window:
                                continue
                                
                            price1_aligned = price1[common_dates]
                            price2_aligned = price2[common_dates]
                            
                            # Calculate ratio and z-score
                            ratio = price1_aligned / price2_aligned
                            rolling_mean = ratio.rolling(window=window).mean()
                            rolling_std = ratio.rolling(window=window).std()
                            zscore = (ratio - rolling_mean) / rolling_std
                            
                            # Generate signals
                            signals = pd.Series(0, index=zscore.index)
                            position = 0
                            
                            for i in range(1, len(zscore)):
                                if pd.isna(zscore.iloc[i]):
                                    signals.iloc[i] = position
                                    continue
                                    
                                current_zscore = zscore.iloc[i]
                                
                                if position == 0:
                                    if current_zscore < -entry_thresh:
                                        position = 1
                                    elif current_zscore > entry_thresh:
                                        position = -1
                                elif abs(current_zscore) < exit_thresh:
                                    position = 0
                                
                                signals.iloc[i] = position
                            
                            # Calculate returns
                            returns1 = price1_aligned.pct_change()
                            returns2 = price2_aligned.pct_change()
                            strategy_returns = signals.shift(1) * (returns1 - returns2)
                            strategy_returns = strategy_returns.fillna(0)
                            
                            # Calculate Sharpe ratio
                            if len(strategy_returns) > 0 and strategy_returns.std() > 0:
                                annualized_return = (1 + strategy_returns.mean()) ** 252 - 1
                                annualized_vol = strategy_returns.std() * np.sqrt(252)
                                sharpe_ratio = annualized_return / annualized_vol
                                
                                total_return = (1 + strategy_returns).prod() - 1
                                cumulative = (1 + strategy_returns).cumprod()
                                rolling_max = cumulative.expanding().max()
                                drawdown = (cumulative - rolling_max) / rolling_max
                                max_drawdown = drawdown.min()
                                
                                winning_trades = (strategy_returns > 0).sum()
                                total_trades = (strategy_returns != 0).sum()
                                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                                
                                result = {
                                    'window': window,
                                    'entry_threshold': entry_thresh,
                                    'exit_threshold': exit_thresh,
                                    'sharpe_ratio': sharpe_ratio,
                                    'total_return': total_return,
                                    'annualized_return': annualized_return,
                                    'annualized_vol': annualized_vol,
                                    'max_drawdown': max_drawdown,
                                    'win_rate': win_rate,
                                    'total_trades': total_trades
                                }
                                
                                results.append(result)
                                
                                if sharpe_ratio > best_sharpe:
                                    best_sharpe = sharpe_ratio
                                    best_params = {
                                        'window': window,
                                        'entry_threshold': entry_thresh,
                                        'exit_threshold': exit_thresh
                                    }
                                    best_in_sample_metrics = {
                                        'Total Return': total_return,
                                        'Annualized Return': annualized_return,
                                        'Sharpe Ratio': sharpe_ratio,
                                        'Max Drawdown': max_drawdown,
                                        'Win Rate': win_rate,
                                        'Total Trades': total_trades
                                    }
                                    
                        except Exception:
                            continue
            
            progress_bar.empty()
            status_text.empty()
            
            return {
                'success': True,
                'best_params': best_params,
                'best_sharpe': best_sharpe,
                'best_in_sample_metrics': best_in_sample_metrics,
                'results': pd.DataFrame(results) if results else pd.DataFrame(),
                'total_combinations': len(results)
            }
            
        except Exception as e:
            return {'error': f'Optimization failed: {str(e)}'}
    
    def calculate_out_of_sample(self, asset1_symbol, asset2_symbol, best_params,
                               start_date='2025-01-01'):
        """Calculate out-of-sample performance using optimized parameters"""
        try:
            end_date = datetime.now()
            symbols = [asset1_symbol, asset2_symbol]
            data = yf.download(symbols, start=start_date, end=end_date, progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                data = data['Close']
            
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            if data.empty or asset1_symbol not in data.columns or asset2_symbol not in data.columns:
                return {'error': 'Data not available for out-of-sample period'}
            
            # Apply optimized parameters
            window = best_params['window']
            entry_threshold = best_params['entry_threshold']
            exit_threshold = best_params['exit_threshold']
            
            price1 = data[asset1_symbol].dropna()
            price2 = data[asset2_symbol].dropna()
            
            common_dates = price1.index.intersection(price2.index)
            if len(common_dates) < window:
                return {'error': f'Insufficient out-of-sample data points: {len(common_dates)}'}
            
            price1_aligned = price1[common_dates]
            price2_aligned = price2[common_dates]
            
            # Calculate ratio and z-score
            ratio = price1_aligned / price2_aligned
            rolling_mean = ratio.rolling(window=window).mean()
            rolling_std = ratio.rolling(window=window).std()
            zscore = (ratio - rolling_mean) / rolling_std
            
            # Generate signals
            signals = pd.Series(0, index=zscore.index)
            position = 0
            
            for i in range(1, len(zscore)):
                if pd.isna(zscore.iloc[i]):
                    signals.iloc[i] = position
                    continue
                    
                current_zscore = zscore.iloc[i]
                
                if position == 0:
                    if current_zscore < -entry_threshold:
                        position = 1
                    elif current_zscore > entry_threshold:
                        position = -1
                elif abs(current_zscore) < exit_threshold:
                    position = 0
                
                signals.iloc[i] = position
            
            # Calculate returns
            returns1 = price1_aligned.pct_change()
            returns2 = price2_aligned.pct_change()
            strategy_returns = signals.shift(1) * (returns1 - returns2)
            strategy_returns = strategy_returns.fillna(0)
            
            # Performance metrics
            total_return = (1 + strategy_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
            annualized_vol = strategy_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
            
            cumulative = (1 + strategy_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            winning_trades = (strategy_returns > 0).sum()
            total_trades = (strategy_returns != 0).sum()
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            downside_returns = strategy_returns[strategy_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
            
            return {
                'success': True,
                'ratio': ratio,
                'zscore': zscore,
                'signals': signals,
                'returns': strategy_returns,
                'cumulative_returns': cumulative,
                'price1': price1_aligned,
                'price2': price2_aligned,
                'metrics': {
                    'Total Return': total_return,
                    'Annualized Return': annualized_return,
                    'Annualized Volatility': annualized_vol,
                    'Sharpe Ratio': sharpe_ratio,
                    'Sortino Ratio': sortino_ratio,
                    'Max Drawdown': max_drawdown,
                    'Win Rate': win_rate,
                    'Total Trades': total_trades
                },
                'current_signals': {
                    'ratio': ratio.iloc[-1] if len(ratio) > 0 else np.nan,
                    'zscore': zscore.iloc[-1] if len(zscore) > 0 else np.nan,
                    'signal': signals.iloc[-1] if len(signals) > 0 else 0
                },
                'parameters': best_params
            }
            
        except Exception as e:
            return {'error': f'Out-of-sample calculation failed: {str(e)}'}

    def calculate_strategy(self, asset1_symbol, asset2_symbol, window=60, 
                          entry_threshold=2.0, exit_threshold=0.5):
        """Calculate complete strategy analysis"""
        try:
            symbols = [asset1_symbol, asset2_symbol]
            data = self.fetch_data(symbols)
            
            if data.empty or asset1_symbol not in data.columns or asset2_symbol not in data.columns:
                return {'error': f'Data not available for {asset1_symbol}/{asset2_symbol}'}
            
            price1 = data[asset1_symbol].dropna()
            price2 = data[asset2_symbol].dropna()
            
            common_dates = price1.index.intersection(price2.index)
            if len(common_dates) < window:
                return {'error': f'Insufficient data points: {len(common_dates)}'}
            
            price1_aligned = price1[common_dates]
            price2_aligned = price2[common_dates]
            
            # Calculate ratio and z-score
            ratio = price1_aligned / price2_aligned
            rolling_mean = ratio.rolling(window=window).mean()
            rolling_std = ratio.rolling(window=window).std()
            zscore = (ratio - rolling_mean) / rolling_std
            
            # Generate signals
            signals = pd.Series(0, index=zscore.index)
            position = 0
            
            for i in range(1, len(zscore)):
                if pd.isna(zscore.iloc[i]):
                    signals.iloc[i] = position
                    continue
                    
                current_zscore = zscore.iloc[i]
                
                if position == 0:
                    if current_zscore < -entry_threshold:
                        position = 1  # Long ratio
                    elif current_zscore > entry_threshold:
                        position = -1  # Short ratio
                elif abs(current_zscore) < exit_threshold:
                    position = 0
                
                signals.iloc[i] = position
            
            # Calculate returns
            returns1 = price1_aligned.pct_change()
            returns2 = price2_aligned.pct_change()
            strategy_returns = signals.shift(1) * (returns1 - returns2)
            strategy_returns = strategy_returns.fillna(0)
            
            # Performance metrics
            total_return = (1 + strategy_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
            annualized_vol = strategy_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
            
            cumulative = (1 + strategy_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            winning_trades = (strategy_returns > 0).sum()
            total_trades = (strategy_returns != 0).sum()
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            downside_returns = strategy_returns[strategy_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
            
            return {
                'success': True,
                'ratio': ratio,
                'zscore': zscore,
                'signals': signals,
                'returns': strategy_returns,
                'cumulative_returns': cumulative,
                'price1': price1_aligned,
                'price2': price2_aligned,
                'metrics': {
                    'Total Return': total_return,
                    'Annualized Return': annualized_return,
                    'Annualized Volatility': annualized_vol,
                    'Sharpe Ratio': sharpe_ratio,
                    'Sortino Ratio': sortino_ratio,
                    'Max Drawdown': max_drawdown,
                    'Win Rate': win_rate,
                    'Total Trades': total_trades
                },
                'current_signals': {
                    'ratio': ratio.iloc[-1] if len(ratio) > 0 else np.nan,
                    'zscore': zscore.iloc[-1] if len(zscore) > 0 else np.nan,
                    'signal': signals.iloc[-1] if len(signals) > 0 else 0
                },
                'parameters': {
                    'window': window,
                    'entry_threshold': entry_threshold,
                    'exit_threshold': exit_threshold
                }
            }
            
        except Exception as e:
            return {'error': f'Strategy calculation failed: {str(e)}'}

# Initialize engine
if 'engine' not in st.session_state:
    st.session_state.engine = BondStrategyEngine()

def main():
    """Main application function"""
    
    # Header
    st.title("Sector Rotation Strategy Dashboard")
    st.markdown("### Analyze mean reversion strategies across bond, geographic, and factor investing sectors")
    
    # Sidebar strategy type selection
    strategy_type = st.sidebar.radio(
        "Strategy Type:",
        ["Bond Strategies", "Geographic Strategies", "Factor Strategies"],
        index=0
    )
    
    # Clear main content and show appropriate strategy
    if strategy_type == "Bond Strategies":
        bond_strategy_tab()
    elif strategy_type == "Geographic Strategies":
        geographic_strategy_tab()
    else:
        factor_strategy_tab()

def bond_strategy_tab():
    """Bond strategy tab content"""
    st.header("Bond Sector Rotation Strategies")
    st.markdown("Credit cycle and duration risk strategies across bond sectors")
    
    # Sidebar - Strategy Selection
    st.sidebar.header("Bond Strategy Configuration")
    
    # Strategy selection
    strategy_options = list(BOND_STRATEGIES.keys())
    
    selected_strategy = st.sidebar.selectbox(
        "Select Bond Strategy:",
        options=strategy_options,
        format_func=lambda x: f"{BOND_STRATEGIES[x]['name']} ({'/'.join(BOND_STRATEGIES[x]['symbols'])})",
        index=0,
        key="bond_strategy_select"
    )
    
    # Parameter controls
    st.sidebar.subheader("Parameters")
    
    window = st.sidebar.slider(
        "Lookback Window (days):",
        min_value=20, max_value=252, value=60, step=10,
        help="Rolling window for calculating statistics",
        key="bond_window"
    )
    
    entry_threshold = st.sidebar.slider(
        "Entry Threshold (σ):",
        min_value=1.0, max_value=3.0, value=2.0, step=0.25,
        help="Z-score threshold for entering positions",
        key="bond_entry"
    )
    
    exit_threshold = st.sidebar.slider(
        "Exit Threshold (σ):",
        min_value=0.1, max_value=1.0, value=0.5, step=0.1,
        help="Z-score threshold for exiting positions",
        key="bond_exit"
    )
    
    # Refresh button
    if st.sidebar.button("Refresh Data", type="primary", key="bond_refresh"):
        st.cache_data.clear()
        st.rerun()
    
    # Optimization section
    st.sidebar.header("Parameter Optimization")
    st.sidebar.markdown("**Optimize parameters using 2021-2024 data**")
    
    if st.sidebar.button("Optimize Parameters", type="secondary", key="bond_optimize"):
        st.session_state.run_optimization = True
    
    # Get strategy configuration
    config = BOND_STRATEGIES[selected_strategy]
    symbols = config['symbols']
    
    # Calculate strategy
    with st.spinner("Calculating strategy..."):
        result = st.session_state.engine.calculate_strategy(
            symbols[0], symbols[1], 
            window, entry_threshold, exit_threshold
        )
    
    if result.get('error'):
        st.error(f"Error: {result['error']}")
        return
    
    # Display strategy info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"{config['name']}")
        st.write(config['description'])
        st.write(f"**Symbols:** {' vs '.join(symbols)}")
    
    with col2:
        # Current signal badge
        current_signals = result.get('current_signals', {})
        signal_val = current_signals.get('signal', 0)
        zscore_val = current_signals.get('zscore', 0)
        
        if signal_val > 0:
            st.success(f"LONG {symbols[0]}")
            st.write(f"**Action:** Buy {symbols[0]}, Sell {symbols[1]}")
        elif signal_val < 0:
            st.error(f"SHORT {symbols[0]}")
            st.write(f"**Action:** Sell {symbols[0]}, Buy {symbols[1]}")
        else:
            st.warning("NEUTRAL")
            st.write("**Action:** No position")
        
        st.write(f"**Current Z-Score:** {zscore_val:+.2f}")
    
    # Performance metrics
    metrics = result.get('metrics', {})
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            f"{metrics.get('Total Return', 0):.1%}",
            help="Total strategy return over the period"
        )
    
    with col2:
        st.metric(
            "Sharpe Ratio",
            f"{metrics.get('Sharpe Ratio', 0):.2f}",
            help="Risk-adjusted return measure"
        )
    
    with col3:
        st.metric(
            "Max Drawdown",
            f"{metrics.get('Max Drawdown', 0):.1%}",
            help="Maximum peak-to-trough decline"
        )
    
    with col4:
        st.metric(
            "Win Rate",
            f"{metrics.get('Win Rate', 0):.1%}",
            help="Percentage of profitable trades"
        )
    
    # Charts
    st.subheader("Strategy Analysis")
    
    # Create 4-panel chart
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            f'Price Evolution: {" vs ".join(symbols)}',
            'Price Ratio',
            'Z-Score with Entry/Exit Thresholds',
            'Cumulative Strategy Returns'
        ],
        vertical_spacing=0.08
    )
    
    # Get data
    ratio = result['ratio']
    zscore = result['zscore']
    signals = result['signals']
    cumulative = result['cumulative_returns']
    price1 = result['price1']
    price2 = result['price2']
    
    # 1. Price Evolution
    fig.add_trace(
        go.Scatter(x=price1.index, y=price1.values, 
                  name=symbols[0], line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=price2.index, y=price2.values, 
                  name=symbols[1], line=dict(color='red')),
        row=1, col=1
    )
    
    # 2. Price Ratio
    fig.add_trace(
        go.Scatter(x=ratio.index, y=ratio.values, 
                  name='Price Ratio', line=dict(color='purple')),
        row=2, col=1
    )
    
    # 3. Z-Score with thresholds
    fig.add_trace(
        go.Scatter(x=zscore.index, y=zscore.values, 
                  name='Z-Score', line=dict(color='black')),
        row=3, col=1
    )
    
    fig.add_hline(y=entry_threshold, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=-entry_threshold, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=exit_threshold, line_dash="dot", line_color="orange", row=3, col=1)
    fig.add_hline(y=-exit_threshold, line_dash="dot", line_color="orange", row=3, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="gray", row=3, col=1)
    
    # 4. Cumulative Returns
    fig.add_trace(
        go.Scatter(x=cumulative.index, y=cumulative.values, 
                  name='Strategy Returns', line=dict(color='green', width=2)),
        row=4, col=1
    )
    
    fig.update_layout(height=800, showlegend=True, title_text=f"Bond Strategy Analysis: {config['name']}")
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance Table
    st.subheader("Detailed Performance Metrics")
    
    performance_data = {
        "Metric": [
            "Total Return", "Annualized Return", "Annualized Volatility",
            "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", 
            "Win Rate", "Total Trades"
        ],
        "Value": [
            f"{metrics.get('Total Return', 0):.2%}",
            f"{metrics.get('Annualized Return', 0):.2%}",
            f"{metrics.get('Annualized Volatility', 0):.2%}",
            f"{metrics.get('Sharpe Ratio', 0):.3f}",
            f"{metrics.get('Sortino Ratio', 0):.3f}",
            f"{metrics.get('Max Drawdown', 0):.2%}",
            f"{metrics.get('Win Rate', 0):.1%}",
            f"{metrics.get('Total Trades', 0):.0f}"
        ]
    }
    
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True, hide_index=True)
    
    # Strategy Parameters
    with st.expander("Current Strategy Parameters"):
        params = result.get('parameters', {})
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Lookback Window:** {params.get('window', 0)} days")
        with col2:
            st.write(f"**Entry Threshold:** ±{params.get('entry_threshold', 0)} σ")
        with col3:
            st.write(f"**Exit Threshold:** ±{params.get('exit_threshold', 0)} σ")
    
    # Parameter Optimization Section
    if st.session_state.get('run_optimization', False):
        st.header("Parameter Optimization & Out-of-Sample Testing")
        
        with st.spinner("Running parameter optimization on 2021-2024 data..."):
            optimization_result = st.session_state.engine.optimize_parameters(
                symbols[0], symbols[1]
            )
        
        if optimization_result.get('error'):
            st.error(f"Optimization failed: {optimization_result['error']}")
        elif optimization_result.get('success'):
            best_params = optimization_result['best_params']
            best_sharpe = optimization_result['best_sharpe']
            best_in_sample_metrics = optimization_result['best_in_sample_metrics']
            
            st.success(f"Optimization completed! Best Sharpe ratio: {best_sharpe:.3f}")
            
            # Display best parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best Window", f"{best_params['window']} days")
            with col2:
                st.metric("Best Entry Threshold", f"±{best_params['entry_threshold']} σ")
            with col3:
                st.metric("Best Exit Threshold", f"±{best_params['exit_threshold']} σ")
            
            # Calculate out-of-sample performance
            with st.spinner("Calculating out-of-sample performance on 2025+ data..."):
                oos_result = st.session_state.engine.calculate_out_of_sample(
                    symbols[0], symbols[1], best_params
                )
            
            if oos_result.get('error'):
                st.error(f"Out-of-sample calculation failed: {oos_result['error']}")
            elif oos_result.get('success'):
                oos_metrics = oos_result['metrics']
                
                # Performance comparison table
                st.subheader("In-Sample vs Out-of-Sample Performance")
                
                comparison_data = {
                    "Metric": [
                        "Total Return", "Annualized Return", "Sharpe Ratio", 
                        "Max Drawdown", "Win Rate", "Total Trades"
                    ],
                    "In-Sample (2021-2024)": [
                        f"{best_in_sample_metrics.get('Total Return', 0):.2%}",
                        f"{best_in_sample_metrics.get('Annualized Return', 0):.2%}",
                        f"{best_in_sample_metrics.get('Sharpe Ratio', 0):.3f}",
                        f"{best_in_sample_metrics.get('Max Drawdown', 0):.2%}",
                        f"{best_in_sample_metrics.get('Win Rate', 0):.1%}",
                        f"{best_in_sample_metrics.get('Total Trades', 0):.0f}"
                    ],
                    "Out-of-Sample (2025+)": [
                        f"{oos_metrics.get('Total Return', 0):.2%}",
                        f"{oos_metrics.get('Annualized Return', 0):.2%}",
                        f"{oos_metrics.get('Sharpe Ratio', 0):.3f}",
                        f"{oos_metrics.get('Max Drawdown', 0):.2%}",
                        f"{oos_metrics.get('Win Rate', 0):.1%}",
                        f"{oos_metrics.get('Total Trades', 0):.0f}"
                    ]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # Out-of-sample chart
                st.subheader("Out-of-Sample Strategy Performance (2025+)")
                
                oos_ratio = oos_result['ratio']
                oos_zscore = oos_result['zscore']
                oos_cumulative = oos_result['cumulative_returns']
                oos_signals = oos_result['signals']
                
                oos_fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=[
                        'Price Ratio (Out-of-Sample)',
                        'Z-Score with Optimized Thresholds',
                        'Cumulative Returns (Out-of-Sample)'
                    ],
                    vertical_spacing=0.1
                )
                
                # Price ratio
                oos_fig.add_trace(
                    go.Scatter(x=oos_ratio.index, y=oos_ratio.values, 
                              name='Price Ratio', line=dict(color='blue')),
                    row=1, col=1
                )
                
                # Z-score with thresholds
                oos_fig.add_trace(
                    go.Scatter(x=oos_zscore.index, y=oos_zscore.values, 
                              name='Z-Score', line=dict(color='black')),
                    row=2, col=1
                )
                
                oos_fig.add_hline(y=best_params['entry_threshold'], line_dash="dash", 
                                 line_color="red", row=2, col=1)
                oos_fig.add_hline(y=-best_params['entry_threshold'], line_dash="dash", 
                                 line_color="red", row=2, col=1)
                oos_fig.add_hline(y=best_params['exit_threshold'], line_dash="dot", 
                                 line_color="orange", row=2, col=1)
                oos_fig.add_hline(y=-best_params['exit_threshold'], line_dash="dot", 
                                 line_color="orange", row=2, col=1)
                oos_fig.add_hline(y=0, line_dash="solid", line_color="gray", row=2, col=1)
                
                # Cumulative returns
                oos_fig.add_trace(
                    go.Scatter(x=oos_cumulative.index, y=oos_cumulative.values, 
                              name='Out-of-Sample Returns', line=dict(color='green', width=2)),
                    row=3, col=1
                )
                
                oos_fig.update_layout(height=600, showlegend=True, 
                                     title_text=f"Out-of-Sample Analysis: {config['name']}")
                oos_fig.update_xaxes(showgrid=True)
                oos_fig.update_yaxes(showgrid=True)
                
                st.plotly_chart(oos_fig, use_container_width=True)
                
                # Optimization results table
                with st.expander("View All Optimization Results"):
                    if not optimization_result['results'].empty:
                        opt_df = optimization_result['results'].round(4)
                        opt_df_sorted = opt_df.sort_values('sharpe_ratio', ascending=False)
                        st.dataframe(opt_df_sorted, use_container_width=True)
                    else:
                        st.write("No optimization results available")
        
        # Reset optimization flag
        st.session_state.run_optimization = False
    
    # Footer
    st.markdown("---")
    st.markdown("**Disclaimer:** This application is for educational and research purposes only. Past performance does not guarantee future results.")
    
    # Auto-refresh
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Last updated: {timestamp}")

def geographic_strategy_tab():
    """Geographic strategy tab content"""
    st.header("Geographic Sector Rotation Strategies")
    st.markdown("Regional allocation strategies across global equity markets")
    
    # Sidebar - Strategy Selection
    st.sidebar.header("Geographic Strategy Configuration")
    
    # Strategy selection
    strategy_options = list(GEOGRAPHIC_STRATEGIES.keys())
    
    selected_strategy = st.sidebar.selectbox(
        "Select Geographic Strategy:",
        options=strategy_options,
        format_func=lambda x: f"{GEOGRAPHIC_STRATEGIES[x]['name']} ({'/'.join(GEOGRAPHIC_STRATEGIES[x]['symbols'])})",
        index=0,
        key="geo_strategy_select"
    )
    
    # Parameter controls
    st.sidebar.subheader("Parameters")
    
    window = st.sidebar.slider(
        "Lookback Window (days):",
        min_value=20, max_value=252, value=40, step=10,
        help="Rolling window for calculating statistics",
        key="geo_window"
    )
    
    entry_threshold = st.sidebar.slider(
        "Entry Threshold (σ):",
        min_value=1.0, max_value=3.0, value=3.0, step=0.25,
        help="Z-score threshold for entering positions",
        key="geo_entry"
    )
    
    exit_threshold = st.sidebar.slider(
        "Exit Threshold (σ):",
        min_value=0.1, max_value=1.0, value=1.0, step=0.1,
        help="Z-score threshold for exiting positions",
        key="geo_exit"
    )
    
    # Refresh button
    if st.sidebar.button("Refresh Geographic Data", type="primary", key="geo_refresh"):
        st.cache_data.clear()
        st.rerun()
    
    # Optimization section
    st.sidebar.header("Parameter Optimization")
    st.sidebar.markdown("**Optimize parameters using 2021-2024 data**")
    
    if st.sidebar.button("Optimize Geographic Parameters", type="secondary", key="geo_optimize"):
        st.session_state.run_geographic_optimization = True
    
    # Get strategy configuration
    config = GEOGRAPHIC_STRATEGIES[selected_strategy]
    symbols = config['symbols']
    
    # Calculate strategy
    with st.spinner("Calculating geographic strategy..."):
        result = st.session_state.engine.calculate_strategy(
            symbols[0], symbols[1], 
            window, entry_threshold, exit_threshold
        )
    
    if result.get('error'):
        st.error(f"Error: {result['error']}")
        return
    
    # Display strategy info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"{config['name']}")
        st.write(config['description'])
        st.write(f"**ETFs:** {' vs '.join(symbols)}")
    
    with col2:
        # Current signal badge
        current_signals = result.get('current_signals', {})
        signal_val = current_signals.get('signal', 0)
        zscore_val = current_signals.get('zscore', 0)
        
        if signal_val > 0:
            st.success(f"LONG {symbols[0]}")
            st.write(f"**Action:** Buy {symbols[0]}, Sell {symbols[1]}")
        elif signal_val < 0:
            st.error(f"SHORT {symbols[0]}")
            st.write(f"**Action:** Sell {symbols[0]}, Buy {symbols[1]}")
        else:
            st.warning("NEUTRAL")
            st.write("**Action:** No position")
        
        st.write(f"**Current Z-Score:** {zscore_val:+.2f}")
    
    # Performance metrics
    metrics = result.get('metrics', {})
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            f"{metrics.get('Total Return', 0):.1%}",
            help="Total strategy return over the period"
        )
    
    with col2:
        st.metric(
            "Sharpe Ratio",
            f"{metrics.get('Sharpe Ratio', 0):.2f}",
            help="Risk-adjusted return measure"
        )
    
    with col3:
        st.metric(
            "Max Drawdown",
            f"{metrics.get('Max Drawdown', 0):.1%}",
            help="Maximum peak-to-trough decline"
        )
    
    with col4:
        st.metric(
            "Win Rate",
            f"{metrics.get('Win Rate', 0):.1%}",
            help="Percentage of profitable trades"
        )
    
    # Charts
    st.subheader("Geographic Strategy Analysis")
    
    # Create 4-panel chart
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            f'Price Evolution: {" vs ".join(symbols)}',
            'Price Ratio',
            'Z-Score with Entry/Exit Thresholds',
            'Cumulative Strategy Returns'
        ],
        vertical_spacing=0.08
    )
    
    # Get data
    ratio = result['ratio']
    zscore = result['zscore']
    cumulative = result['cumulative_returns']
    price1 = result['price1']
    price2 = result['price2']
    
    # 1. Price Evolution
    fig.add_trace(
        go.Scatter(x=price1.index, y=price1.values, 
                  name=symbols[0], line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=price2.index, y=price2.values, 
                  name=symbols[1], line=dict(color='red')),
        row=1, col=1
    )
    
    # 2. Price Ratio
    fig.add_trace(
        go.Scatter(x=ratio.index, y=ratio.values, 
                  name='Price Ratio', line=dict(color='purple')),
        row=2, col=1
    )
    
    # 3. Z-Score with thresholds
    fig.add_trace(
        go.Scatter(x=zscore.index, y=zscore.values, 
                  name='Z-Score', line=dict(color='black')),
        row=3, col=1
    )
    
    fig.add_hline(y=entry_threshold, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=-entry_threshold, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=exit_threshold, line_dash="dot", line_color="orange", row=3, col=1)
    fig.add_hline(y=-exit_threshold, line_dash="dot", line_color="orange", row=3, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="gray", row=3, col=1)
    
    # 4. Cumulative Returns
    fig.add_trace(
        go.Scatter(x=cumulative.index, y=cumulative.values, 
                  name='Strategy Returns', line=dict(color='green', width=2)),
        row=4, col=1
    )
    
    fig.update_layout(height=800, showlegend=True, title_text=f"Geographic Strategy Analysis: {config['name']}")
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance Table
    st.subheader("Detailed Performance Metrics")
    
    performance_data = {
        "Metric": [
            "Total Return", "Annualized Return", "Annualized Volatility",
            "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", 
            "Win Rate", "Total Trades"
        ],
        "Value": [
            f"{metrics.get('Total Return', 0):.2%}",
            f"{metrics.get('Annualized Return', 0):.2%}",
            f"{metrics.get('Annualized Volatility', 0):.2%}",
            f"{metrics.get('Sharpe Ratio', 0):.3f}",
            f"{metrics.get('Sortino Ratio', 0):.3f}",
            f"{metrics.get('Max Drawdown', 0):.2%}",
            f"{metrics.get('Win Rate', 0):.1%}",
            f"{metrics.get('Total Trades', 0):.0f}"
        ]
    }
    
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True, hide_index=True)
    
    # Strategy Parameters
    with st.expander("Current Strategy Parameters"):
        params = result.get('parameters', {})
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Lookback Window:** {params.get('window', 0)} days")
        with col2:
            st.write(f"**Entry Threshold:** ±{params.get('entry_threshold', 0)} σ")
        with col3:
            st.write(f"**Exit Threshold:** ±{params.get('exit_threshold', 0)} σ")
    
    # Parameter Optimization Section
    if st.session_state.get('run_geographic_optimization', False):
        st.header("Geographic Parameter Optimization & Out-of-Sample Testing")
        
        with st.spinner("Running parameter optimization on 2021-2024 data..."):
            optimization_result = st.session_state.engine.optimize_parameters(
                symbols[0], symbols[1]
            )
        
        if optimization_result.get('error'):
            st.error(f"Optimization failed: {optimization_result['error']}")
        elif optimization_result.get('success'):
            best_params = optimization_result['best_params']
            best_sharpe = optimization_result['best_sharpe']
            best_in_sample_metrics = optimization_result['best_in_sample_metrics']
            
            st.success(f"Optimization completed! Best Sharpe ratio: {best_sharpe:.3f}")
            
            # Display best parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best Window", f"{best_params['window']} days")
            with col2:
                st.metric("Best Entry Threshold", f"±{best_params['entry_threshold']} σ")
            with col3:
                st.metric("Best Exit Threshold", f"±{best_params['exit_threshold']} σ")
            
            # Calculate out-of-sample performance
            with st.spinner("Calculating out-of-sample performance on 2025+ data..."):
                oos_result = st.session_state.engine.calculate_out_of_sample(
                    symbols[0], symbols[1], best_params
                )
            
            if oos_result.get('error'):
                st.error(f"Out-of-sample calculation failed: {oos_result['error']}")
            elif oos_result.get('success'):
                oos_metrics = oos_result['metrics']
                
                # Performance comparison table
                st.subheader("In-Sample vs Out-of-Sample Performance")
                
                comparison_data = {
                    "Metric": [
                        "Total Return", "Annualized Return", "Sharpe Ratio", 
                        "Max Drawdown", "Win Rate", "Total Trades"
                    ],
                    "In-Sample (2021-2024)": [
                        f"{best_in_sample_metrics.get('Total Return', 0):.2%}",
                        f"{best_in_sample_metrics.get('Annualized Return', 0):.2%}",
                        f"{best_in_sample_metrics.get('Sharpe Ratio', 0):.3f}",
                        f"{best_in_sample_metrics.get('Max Drawdown', 0):.2%}",
                        f"{best_in_sample_metrics.get('Win Rate', 0):.1%}",
                        f"{best_in_sample_metrics.get('Total Trades', 0):.0f}"
                    ],
                    "Out-of-Sample (2025+)": [
                        f"{oos_metrics.get('Total Return', 0):.2%}",
                        f"{oos_metrics.get('Annualized Return', 0):.2%}",
                        f"{oos_metrics.get('Sharpe Ratio', 0):.3f}",
                        f"{oos_metrics.get('Max Drawdown', 0):.2%}",
                        f"{oos_metrics.get('Win Rate', 0):.1%}",
                        f"{oos_metrics.get('Total Trades', 0):.0f}"
                    ]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # Out-of-sample chart
                st.subheader("Out-of-Sample Strategy Performance (2025+)")
                
                oos_ratio = oos_result['ratio']
                oos_zscore = oos_result['zscore']
                oos_cumulative = oos_result['cumulative_returns']
                
                oos_fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=[
                        'Price Ratio (Out-of-Sample)',
                        'Z-Score with Optimized Thresholds',
                        'Cumulative Returns (Out-of-Sample)'
                    ],
                    vertical_spacing=0.1
                )
                
                # Price ratio
                oos_fig.add_trace(
                    go.Scatter(x=oos_ratio.index, y=oos_ratio.values, 
                              name='Price Ratio', line=dict(color='blue')),
                    row=1, col=1
                )
                
                # Z-score with thresholds
                oos_fig.add_trace(
                    go.Scatter(x=oos_zscore.index, y=oos_zscore.values, 
                              name='Z-Score', line=dict(color='black')),
                    row=2, col=1
                )
                
                oos_fig.add_hline(y=best_params['entry_threshold'], line_dash="dash", 
                                 line_color="red", row=2, col=1)
                oos_fig.add_hline(y=-best_params['entry_threshold'], line_dash="dash", 
                                 line_color="red", row=2, col=1)
                oos_fig.add_hline(y=best_params['exit_threshold'], line_dash="dot", 
                                 line_color="orange", row=2, col=1)
                oos_fig.add_hline(y=-best_params['exit_threshold'], line_dash="dot", 
                                 line_color="orange", row=2, col=1)
                oos_fig.add_hline(y=0, line_dash="solid", line_color="gray", row=2, col=1)
                
                # Cumulative returns
                oos_fig.add_trace(
                    go.Scatter(x=oos_cumulative.index, y=oos_cumulative.values, 
                              name='Out-of-Sample Returns', line=dict(color='green', width=2)),
                    row=3, col=1
                )
                
                oos_fig.update_layout(height=600, showlegend=True, 
                                     title_text=f"Out-of-Sample Analysis: {config['name']}")
                oos_fig.update_xaxes(showgrid=True)
                oos_fig.update_yaxes(showgrid=True)
                
                st.plotly_chart(oos_fig, use_container_width=True)
                
                # Optimization results table
                with st.expander("View All Optimization Results"):
                    if not optimization_result['results'].empty:
                        opt_df = optimization_result['results'].round(4)
                        opt_df_sorted = opt_df.sort_values('sharpe_ratio', ascending=False)
                        st.dataframe(opt_df_sorted, use_container_width=True)
                    else:
                        st.write("No optimization results available")
        
        # Reset optimization flag
        st.session_state.run_geographic_optimization = False

def factor_strategy_tab():
    """Factor strategy tab content"""
    st.header("Factor ETF Rotation Strategies")
    st.markdown("Factor-based mean reversion strategies across different investment factors")
    
    # Sidebar - Strategy Selection
    st.sidebar.header("Factor Strategy Configuration")
    
    # Strategy selection
    strategy_options = list(FACTOR_STRATEGIES.keys())
    
    selected_strategy = st.sidebar.selectbox(
        "Select Factor Strategy:",
        options=strategy_options,
        format_func=lambda x: f"{FACTOR_STRATEGIES[x]['name']} ({'/'.join(FACTOR_STRATEGIES[x]['symbols'])})",
        index=0,
        key="factor_strategy_select"
    )
    
    # Parameter controls
    st.sidebar.subheader("Parameters")
    
    window = st.sidebar.slider(
        "Lookback Window (days):",
        min_value=20, max_value=252, value=50, step=10,
        help="Rolling window for calculating statistics",
        key="factor_window"
    )
    
    entry_threshold = st.sidebar.slider(
        "Entry Threshold (σ):",
        min_value=1.0, max_value=3.0, value=2.5, step=0.25,
        help="Z-score threshold for entering positions",
        key="factor_entry"
    )
    
    exit_threshold = st.sidebar.slider(
        "Exit Threshold (σ):",
        min_value=0.1, max_value=1.0, value=0.75, step=0.1,
        help="Z-score threshold for exiting positions",
        key="factor_exit"
    )
    
    # Refresh button
    if st.sidebar.button("Refresh Factor Data", type="primary", key="factor_refresh"):
        st.cache_data.clear()
        st.rerun()
    
    # Optimization section
    st.sidebar.header("Parameter Optimization")
    st.sidebar.markdown("**Optimize parameters using 2021-2024 data**")
    
    if st.sidebar.button("Optimize Factor Parameters", type="secondary", key="factor_optimize"):
        st.session_state.run_factor_optimization = True
    
    # Get strategy configuration
    config = FACTOR_STRATEGIES[selected_strategy]
    symbols = config['symbols']
    
    # Calculate strategy
    with st.spinner("Calculating factor strategy..."):
        result = st.session_state.engine.calculate_strategy(
            symbols[0], symbols[1], 
            window, entry_threshold, exit_threshold
        )
    
    if result.get('error'):
        st.error(f"Error: {result['error']}")
        return
    
    # Display strategy info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"{config['name']}")
        st.write(config['description'])
        st.write(f"**ETFs:** {' vs '.join(symbols)}")
    
    with col2:
        # Current signal badge
        current_signals = result.get('current_signals', {})
        signal_val = current_signals.get('signal', 0)
        zscore_val = current_signals.get('zscore', 0)
        
        if signal_val > 0:
            st.success(f"LONG {symbols[0]}")
            st.write(f"**Action:** Buy {symbols[0]}, Sell {symbols[1]}")
        elif signal_val < 0:
            st.error(f"SHORT {symbols[0]}")
            st.write(f"**Action:** Sell {symbols[0]}, Buy {symbols[1]}")
        else:
            st.warning("NEUTRAL")
            st.write("**Action:** No position")
        
        st.write(f"**Current Z-Score:** {zscore_val:+.2f}")
    
    # Performance metrics
    metrics = result.get('metrics', {})
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            f"{metrics.get('Total Return', 0):.1%}",
            help="Total strategy return over the period"
        )
    
    with col2:
        st.metric(
            "Sharpe Ratio",
            f"{metrics.get('Sharpe Ratio', 0):.2f}",
            help="Risk-adjusted return measure"
        )
    
    with col3:
        st.metric(
            "Max Drawdown",
            f"{metrics.get('Max Drawdown', 0):.1%}",
            help="Maximum peak-to-trough decline"
        )
    
    with col4:
        st.metric(
            "Win Rate",
            f"{metrics.get('Win Rate', 0):.1%}",
            help="Percentage of profitable trades"
        )
    
    # Charts
    st.subheader("Factor Strategy Analysis")
    
    # Create 4-panel chart
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            f'Price Evolution: {" vs ".join(symbols)}',
            'Price Ratio',
            'Z-Score with Entry/Exit Thresholds',
            'Cumulative Strategy Returns'
        ],
        vertical_spacing=0.08
    )
    
    # Get data
    ratio = result['ratio']
    zscore = result['zscore']
    cumulative = result['cumulative_returns']
    price1 = result['price1']
    price2 = result['price2']
    
    # 1. Price Evolution
    fig.add_trace(
        go.Scatter(x=price1.index, y=price1.values, 
                  name=symbols[0], line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=price2.index, y=price2.values, 
                  name=symbols[1], line=dict(color='red')),
        row=1, col=1
    )
    
    # 2. Price Ratio
    fig.add_trace(
        go.Scatter(x=ratio.index, y=ratio.values, 
                  name='Price Ratio', line=dict(color='purple')),
        row=2, col=1
    )
    
    # 3. Z-Score with thresholds
    fig.add_trace(
        go.Scatter(x=zscore.index, y=zscore.values, 
                  name='Z-Score', line=dict(color='black')),
        row=3, col=1
    )
    
    fig.add_hline(y=entry_threshold, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=-entry_threshold, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=exit_threshold, line_dash="dot", line_color="orange", row=3, col=1)
    fig.add_hline(y=-exit_threshold, line_dash="dot", line_color="orange", row=3, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="gray", row=3, col=1)
    
    # 4. Cumulative Returns
    fig.add_trace(
        go.Scatter(x=cumulative.index, y=cumulative.values, 
                  name='Strategy Returns', line=dict(color='green', width=2)),
        row=4, col=1
    )
    
    fig.update_layout(height=800, showlegend=True, title_text=f"Factor Strategy Analysis: {config['name']}")
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance Table
    st.subheader("Detailed Performance Metrics")
    
    performance_data = {
        "Metric": [
            "Total Return", "Annualized Return", "Annualized Volatility",
            "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", 
            "Win Rate", "Total Trades"
        ],
        "Value": [
            f"{metrics.get('Total Return', 0):.2%}",
            f"{metrics.get('Annualized Return', 0):.2%}",
            f"{metrics.get('Annualized Volatility', 0):.2%}",
            f"{metrics.get('Sharpe Ratio', 0):.3f}",
            f"{metrics.get('Sortino Ratio', 0):.3f}",
            f"{metrics.get('Max Drawdown', 0):.2%}",
            f"{metrics.get('Win Rate', 0):.1%}",
            f"{metrics.get('Total Trades', 0):.0f}"
        ]
    }
    
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True, hide_index=True)
    
    # Strategy Parameters
    with st.expander("Current Strategy Parameters"):
        params = result.get('parameters', {})
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Lookback Window:** {params.get('window', 0)} days")
        with col2:
            st.write(f"**Entry Threshold:** ±{params.get('entry_threshold', 0)} σ")
        with col3:
            st.write(f"**Exit Threshold:** ±{params.get('exit_threshold', 0)} σ")
    
    # Parameter Optimization Section
    if st.session_state.get('run_factor_optimization', False):
        st.header("Factor Parameter Optimization & Out-of-Sample Testing")
        
        with st.spinner("Running parameter optimization on 2021-2024 data..."):
            optimization_result = st.session_state.engine.optimize_parameters(
                symbols[0], symbols[1]
            )
        
        if optimization_result.get('error'):
            st.error(f"Optimization failed: {optimization_result['error']}")
        elif optimization_result.get('success'):
            best_params = optimization_result['best_params']
            best_sharpe = optimization_result['best_sharpe']
            best_in_sample_metrics = optimization_result['best_in_sample_metrics']
            
            st.success(f"Optimization completed! Best Sharpe ratio: {best_sharpe:.3f}")
            
            # Display best parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best Window", f"{best_params['window']} days")
            with col2:
                st.metric("Best Entry Threshold", f"±{best_params['entry_threshold']} σ")
            with col3:
                st.metric("Best Exit Threshold", f"±{best_params['exit_threshold']} σ")
            
            # Calculate out-of-sample performance
            with st.spinner("Calculating out-of-sample performance on 2025+ data..."):
                oos_result = st.session_state.engine.calculate_out_of_sample(
                    symbols[0], symbols[1], best_params
                )
            
            if oos_result.get('error'):
                st.error(f"Out-of-sample calculation failed: {oos_result['error']}")
            elif oos_result.get('success'):
                oos_metrics = oos_result['metrics']
                
                # Performance comparison table
                st.subheader("In-Sample vs Out-of-Sample Performance")
                
                comparison_data = {
                    "Metric": [
                        "Total Return", "Annualized Return", "Sharpe Ratio", 
                        "Max Drawdown", "Win Rate", "Total Trades"
                    ],
                    "In-Sample (2021-2024)": [
                        f"{best_in_sample_metrics.get('Total Return', 0):.2%}",
                        f"{best_in_sample_metrics.get('Annualized Return', 0):.2%}",
                        f"{best_in_sample_metrics.get('Sharpe Ratio', 0):.3f}",
                        f"{best_in_sample_metrics.get('Max Drawdown', 0):.2%}",
                        f"{best_in_sample_metrics.get('Win Rate', 0):.1%}",
                        f"{best_in_sample_metrics.get('Total Trades', 0):.0f}"
                    ],
                    "Out-of-Sample (2025+)": [
                        f"{oos_metrics.get('Total Return', 0):.2%}",
                        f"{oos_metrics.get('Annualized Return', 0):.2%}",
                        f"{oos_metrics.get('Sharpe Ratio', 0):.3f}",
                        f"{oos_metrics.get('Max Drawdown', 0):.2%}",
                        f"{oos_metrics.get('Win Rate', 0):.1%}",
                        f"{oos_metrics.get('Total Trades', 0):.0f}"
                    ]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # Out-of-sample chart
                st.subheader("Out-of-Sample Strategy Performance (2025+)")
                
                oos_ratio = oos_result['ratio']
                oos_zscore = oos_result['zscore']
                oos_cumulative = oos_result['cumulative_returns']
                
                oos_fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=[
                        'Price Ratio (Out-of-Sample)',
                        'Z-Score with Optimized Thresholds',
                        'Cumulative Returns (Out-of-Sample)'
                    ],
                    vertical_spacing=0.1
                )
                
                # Price ratio
                oos_fig.add_trace(
                    go.Scatter(x=oos_ratio.index, y=oos_ratio.values, 
                              name='Price Ratio', line=dict(color='blue')),
                    row=1, col=1
                )
                
                # Z-score with thresholds
                oos_fig.add_trace(
                    go.Scatter(x=oos_zscore.index, y=oos_zscore.values, 
                              name='Z-Score', line=dict(color='black')),
                    row=2, col=1
                )
                
                oos_fig.add_hline(y=best_params['entry_threshold'], line_dash="dash", 
                                 line_color="red", row=2, col=1)
                oos_fig.add_hline(y=-best_params['entry_threshold'], line_dash="dash", 
                                 line_color="red", row=2, col=1)
                oos_fig.add_hline(y=best_params['exit_threshold'], line_dash="dot", 
                                 line_color="orange", row=2, col=1)
                oos_fig.add_hline(y=-best_params['exit_threshold'], line_dash="dot", 
                                 line_color="orange", row=2, col=1)
                oos_fig.add_hline(y=0, line_dash="solid", line_color="gray", row=2, col=1)
                
                # Cumulative returns
                oos_fig.add_trace(
                    go.Scatter(x=oos_cumulative.index, y=oos_cumulative.values, 
                              name='Out-of-Sample Returns', line=dict(color='green', width=2)),
                    row=3, col=1
                )
                
                oos_fig.update_layout(height=600, showlegend=True, 
                                     title_text=f"Out-of-Sample Analysis: {config['name']}")
                oos_fig.update_xaxes(showgrid=True)
                oos_fig.update_yaxes(showgrid=True)
                
                st.plotly_chart(oos_fig, use_container_width=True)
                
                # Optimization results table
                with st.expander("View All Optimization Results"):
                    if not optimization_result['results'].empty:
                        opt_df = optimization_result['results'].round(4)
                        opt_df_sorted = opt_df.sort_values('sharpe_ratio', ascending=False)
                        st.dataframe(opt_df_sorted, use_container_width=True)
                    else:
                        st.write("No optimization results available")
        
        # Reset optimization flag
        st.session_state.run_factor_optimization = False

if __name__ == "__main__":
    main()