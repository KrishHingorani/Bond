"""
Mean Reversion Testing Framework
===============================

Statistical tests to validate mean reversion assumptions before implementing strategies.
Includes stationarity tests, cointegration analysis, half-life calculations, and 
comprehensive pair screening framework.

Author: Research Team
Date: October 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.regression.linear_model import OLS
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class MeanReversionTester:
    """Comprehensive mean reversion testing framework"""
    
    def __init__(self):
        self.test_results = {}
    
    @st.cache_data(ttl=3600)
    def fetch_data(_self, symbols, period='5y'):
        """Fetch data for testing with caching"""
        try:
            data = yf.download(symbols, period=period, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data = data['Close']
            data = data.fillna(method='ffill').fillna(method='bfill')
            return data
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
    
    def test_stationarity(self, series, name="Series"):
        """
        Test if a time series is stationary (mean reverting)
        Uses both ADF and KPSS tests for robustness
        """
        series_clean = series.dropna()
        
        if len(series_clean) < 10:
            return {
                'series_name': name,
                'error': 'Insufficient data for stationarity testing'
            }
        
        try:
            # Augmented Dickey-Fuller Test (H0: non-stationary)
            adf_stat, adf_pvalue, adf_lags, adf_nobs, adf_critical, adf_icbest = adfuller(
                series_clean, autolag='AIC'
            )
            
            # KPSS Test (H0: stationary)
            kpss_stat, kpss_pvalue, kpss_lags, kpss_critical = kpss(
                series_clean, regression='c', nlags='auto'
            )
            
            # Interpretation
            is_stationary_adf = adf_pvalue < 0.05  # Reject H0 = stationary
            is_stationary_kpss = kpss_pvalue > 0.05  # Fail to reject H0 = stationary
            
            # Combined interpretation
            if is_stationary_adf and is_stationary_kpss:
                stationarity_conclusion = "Stationary (both tests agree)"
                confidence = "High"
            elif is_stationary_adf or is_stationary_kpss:
                stationarity_conclusion = "Possibly Stationary (tests disagree)"
                confidence = "Medium"
            else:
                stationarity_conclusion = "Non-Stationary (both tests agree)"
                confidence = "High"
            
            return {
                'series_name': name,
                'adf_statistic': adf_stat,
                'adf_pvalue': adf_pvalue,
                'adf_critical_1pct': adf_critical['1%'],
                'adf_critical_5pct': adf_critical['5%'],
                'is_stationary_adf': is_stationary_adf,
                'kpss_statistic': kpss_stat,
                'kpss_pvalue': kpss_pvalue,
                'kpss_critical_1pct': kpss_critical['1%'],
                'kpss_critical_5pct': kpss_critical['5%'],
                'is_stationary_kpss': is_stationary_kpss,
                'conclusion': stationarity_conclusion,
                'confidence': confidence,
                'sample_size': len(series_clean)
            }
            
        except Exception as e:
            return {
                'series_name': name,
                'error': f'Stationarity test failed: {str(e)}'
            }
    
    def test_cointegration(self, price1, price2, pair_name="Pair"):
        """
        Test if two price series are cointegrated
        Uses Engle-Granger two-step method
        """
        try:
            # Align series
            common_dates = price1.index.intersection(price2.index)
            p1 = price1[common_dates].dropna()
            p2 = price2[common_dates].dropna()
            
            if len(p1) < 30 or len(p2) < 30:
                return {
                    'pair_name': pair_name,
                    'error': 'Insufficient data for cointegration testing'
                }
            
            # Use log prices for better statistical properties
            log_p1 = np.log(p1)
            log_p2 = np.log(p2)
            
            # Engle-Granger test (both directions)
            # Direction 1: p1 ~ p2
            try:
                result1 = coint(log_p1, log_p2)
                coint_stat1, coint_pvalue1, coint_critical1 = result1[0], result1[1], result1[2]
                # Convert critical values array to dict if needed
                if isinstance(coint_critical1, np.ndarray):
                    coint_critical1 = {'1%': coint_critical1[0], '5%': coint_critical1[1], '10%': coint_critical1[2]}
            except:
                coint_stat1, coint_pvalue1 = np.nan, 1.0
                coint_critical1 = {'1%': np.nan, '5%': np.nan, '10%': np.nan}
            
            # Direction 2: p2 ~ p1
            try:
                result2 = coint(log_p2, log_p1)
                coint_stat2, coint_pvalue2, coint_critical2 = result2[0], result2[1], result2[2]
                # Convert critical values array to dict if needed
                if isinstance(coint_critical2, np.ndarray):
                    coint_critical2 = {'1%': coint_critical2[0], '5%': coint_critical2[1], '10%': coint_critical2[2]}
            except:
                coint_stat2, coint_pvalue2 = np.nan, 1.0
                coint_critical2 = {'1%': np.nan, '5%': np.nan, '10%': np.nan}
            
            # Take the better result
            if coint_pvalue1 <= coint_pvalue2:
                best_pvalue = coint_pvalue1
                best_stat = coint_stat1
                best_critical = coint_critical1
                direction = f"{pair_name.split('/')[0]} ~ {pair_name.split('/')[1] if '/' in pair_name else 'Asset2'}"
            else:
                best_pvalue = coint_pvalue2
                best_stat = coint_stat2
                best_critical = coint_critical2
                direction = f"{pair_name.split('/')[1] if '/' in pair_name else 'Asset2'} ~ {pair_name.split('/')[0]}"
            
            # Calculate hedge ratio using OLS
            model = OLS(log_p1, log_p2).fit()
            hedge_ratio = model.params[0]
            
            # Spread for half-life calculation
            spread = log_p1 - hedge_ratio * log_p2
            
            return {
                'pair_name': pair_name,
                'cointegrated': best_pvalue < 0.05,
                'coint_pvalue': best_pvalue,
                'coint_statistic': best_stat,
                'coint_critical_5pct': best_critical.get('5%', np.nan),
                'hedge_ratio': hedge_ratio,
                'spread_series': spread,
                'direction': direction,
                'sample_size': len(p1)
            }
            
        except Exception as e:
            return {
                'pair_name': pair_name,
                'error': f'Cointegration test failed: {str(e)}'
            }
    
    def calculate_half_life(self, series, name="Series"):
        """
        Calculate mean reversion half-life using AR(1) model
        """
        try:
            series_clean = series.dropna()
            
            if len(series_clean) < 20:
                return {
                    'series_name': name,
                    'error': 'Insufficient data for half-life calculation'
                }
            
            # Lag the series
            lagged_series = series_clean.shift(1).dropna()
            delta_series = series_clean.diff().dropna()
            
            # Align the series
            aligned_idx = lagged_series.index.intersection(delta_series.index)
            lagged_aligned = lagged_series[aligned_idx]
            delta_aligned = delta_series[aligned_idx]
            
            if len(aligned_idx) < 10:
                return {
                    'series_name': name,
                    'error': 'Insufficient aligned data for half-life calculation'
                }
            
            # AR(1) regression: Δy = α + βy(t-1) + ε
            model = OLS(delta_aligned, lagged_aligned).fit()
            beta = model.params[0]
            beta_pvalue = model.pvalues[0]
            
            # Half-life calculation
            if beta < 0 and beta > -1:
                half_life = -np.log(2) / np.log(1 + beta)
                is_mean_reverting = beta_pvalue < 0.05
            else:
                half_life = np.inf
                is_mean_reverting = False
            
            # Mean reversion strength
            reversion_strength = abs(beta) if beta < 0 else 0
            
            return {
                'series_name': name,
                'half_life_days': half_life,
                'mean_revert_coefficient': beta,
                'coefficient_pvalue': beta_pvalue,
                'is_mean_reverting': is_mean_reverting,
                'reversion_strength': reversion_strength,
                'r_squared': model.rsquared,
                'sample_size': len(aligned_idx)
            }
            
        except Exception as e:
            return {
                'series_name': name,
                'error': f'Half-life calculation failed: {str(e)}'
            }
    
    def calculate_hurst_exponent(self, series, max_lag=100, name="Series"):
        """
        Calculate Hurst exponent
        H < 0.5: mean reverting
        H = 0.5: random walk
        H > 0.5: trending
        """
        try:
            series_clean = series.dropna()
            
            if len(series_clean) < max_lag * 2:
                max_lag = len(series_clean) // 3
            
            if max_lag < 5:
                return {
                    'series_name': name,
                    'error': 'Insufficient data for Hurst exponent calculation'
                }
            
            lags = range(2, min(max_lag, len(series_clean) // 2))
            rs_values = []
            
            for lag in lags:
                # Calculate rescaled range
                ts = series_clean[:lag]
                mean_ts = ts.mean()
                cum_devs = (ts - mean_ts).cumsum()
                r = cum_devs.max() - cum_devs.min()
                s = ts.std()
                
                if s > 0:
                    rs = r / s
                    rs_values.append(rs)
                else:
                    rs_values.append(np.nan)
            
            # Remove NaN values
            valid_rs = [(lag, rs) for lag, rs in zip(lags, rs_values) if not np.isnan(rs) and rs > 0]
            
            if len(valid_rs) < 5:
                return {
                    'series_name': name,
                    'error': 'Insufficient valid data points for Hurst calculation'
                }
            
            lags_valid, rs_valid = zip(*valid_rs)
            
            # Fit log(R/S) = H * log(lag) + constant
            log_lags = np.log(lags_valid)
            log_rs = np.log(rs_valid)
            
            hurst = np.polyfit(log_lags, log_rs, 1)[0]
            
            # Interpretation
            if hurst < 0.45:
                interpretation = "Mean Reverting"
            elif hurst > 0.55:
                interpretation = "Trending"
            else:
                interpretation = "Random Walk"
            
            return {
                'series_name': name,
                'hurst_exponent': hurst,
                'interpretation': interpretation,
                'sample_size': len(series_clean),
                'lags_used': len(valid_rs)
            }
            
        except Exception as e:
            return {
                'series_name': name,
                'error': f'Hurst exponent calculation failed: {str(e)}'
            }
    
    def rolling_stationarity_test(self, series, window=252, name="Series"):
        """
        Test stationarity over rolling windows to detect regime changes
        """
        try:
            if len(series) < window * 2:
                return {
                    'series_name': name,
                    'error': f'Insufficient data for rolling analysis (need {window*2}, have {len(series)})'
                }
            
            rolling_results = []
            
            for i in range(window, len(series), 20):  # Test every 20 days
                window_data = series.iloc[i-window:i]
                
                if len(window_data.dropna()) < window * 0.8:  # Need at least 80% valid data
                    continue
                
                stationarity_result = self.test_stationarity(window_data, f"{name}_window_{i}")
                
                if 'error' not in stationarity_result:
                    stationarity_result['date'] = series.index[i-1]
                    stationarity_result['window_end'] = i
                    rolling_results.append(stationarity_result)
            
            if len(rolling_results) == 0:
                return {
                    'series_name': name,
                    'error': 'No valid rolling windows for analysis'
                }
            
            rolling_df = pd.DataFrame(rolling_results)
            
            # Calculate stability metrics
            adf_stable = rolling_df['is_stationary_adf'].std()
            kpss_stable = rolling_df['is_stationary_kpss'].std()
            stationarity_stability = 1 - (adf_stable + kpss_stable) / 2
            
            return {
                'series_name': name,
                'rolling_results': rolling_df,
                'stationarity_stability': stationarity_stability,
                'avg_adf_pvalue': rolling_df['adf_pvalue'].mean(),
                'avg_kpss_pvalue': rolling_df['kpss_pvalue'].mean(),
                'consistent_stationarity': rolling_df['is_stationary_adf'].mean(),
                'num_windows': len(rolling_results)
            }
            
        except Exception as e:
            return {
                'series_name': name,
                'error': f'Rolling stationarity test failed: {str(e)}'
            }
    
    def detect_regime_changes(self, series, name="Series"):
        """
        Detect regime changes using unsupervised learning
        """
        try:
            if len(series) < 120:  # Need at least 4 months of data
                return {
                    'series_name': name,
                    'error': 'Insufficient data for regime detection'
                }
            
            window = 60
            rolling_vol = series.rolling(window).std()
            rolling_autocorr = series.rolling(window).apply(lambda x: x.autocorr(lag=1))
            rolling_mean = series.rolling(window).mean()
            
            # Create features for regime detection
            features = pd.DataFrame({
                'volatility': rolling_vol,
                'autocorr': rolling_autocorr,
                'level': rolling_mean
            }).dropna()
            
            if len(features) < 20:
                return {
                    'series_name': name,
                    'error': 'Insufficient features for regime detection'
                }
            
            # Normalize features
            features_norm = (features - features.mean()) / features.std()
            
            # K-means clustering for regime detection
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            regimes = kmeans.fit_predict(features_norm)
            
            regime_series = pd.Series(regimes, index=features.index)
            
            # Calculate regime statistics
            regime_changes = (regime_series.diff() != 0).sum()
            avg_regime_length = len(regime_series) / max(regime_changes, 1)
            
            # Regime characteristics
            regime_0_vol = features[regime_series == 0]['volatility'].mean()
            regime_1_vol = features[regime_series == 1]['volatility'].mean()
            
            regime_0_autocorr = features[regime_series == 0]['autocorr'].mean()
            regime_1_autocorr = features[regime_series == 1]['autocorr'].mean()
            
            return {
                'series_name': name,
                'regime_series': regime_series,
                'num_regime_changes': regime_changes,
                'avg_regime_length_days': avg_regime_length,
                'regime_0_characteristics': {
                    'volatility': regime_0_vol,
                    'autocorr': regime_0_autocorr,
                    'periods': (regime_series == 0).sum()
                },
                'regime_1_characteristics': {
                    'volatility': regime_1_vol,
                    'autocorr': regime_1_autocorr,
                    'periods': (regime_series == 1).sum()
                },
                'current_regime': regime_series.iloc[-1],
                'sample_size': len(features)
            }
            
        except Exception as e:
            return {
                'series_name': name,
                'error': f'Regime detection failed: {str(e)}'
            }
    
    def comprehensive_pair_analysis(self, symbol1, symbol2, pair_name=None, period='5y'):
        """
        Complete statistical analysis of a trading pair
        """
        if pair_name is None:
            pair_name = f"{symbol1}/{symbol2}"
        
        # Fetch data
        data = self.fetch_data([symbol1, symbol2], period=period)
        
        if data.empty or symbol1 not in data.columns or symbol2 not in data.columns:
            return {
                'pair_name': pair_name,
                'error': f'Could not fetch data for {symbol1}/{symbol2}'
            }
        
        price1 = data[symbol1].dropna()
        price2 = data[symbol2].dropna()
        
        # Calculate ratio
        common_dates = price1.index.intersection(price2.index)
        if len(common_dates) < 100:
            return {
                'pair_name': pair_name,
                'error': f'Insufficient overlapping data: {len(common_dates)} days'
            }
        
        price1_aligned = price1[common_dates]
        price2_aligned = price2[common_dates]
        ratio = price1_aligned / price2_aligned
        
        # Run all tests
        results = {
            'pair_name': pair_name,
            'symbol1': symbol1,
            'symbol2': symbol2,
            'data_start': common_dates[0],
            'data_end': common_dates[-1],
            'num_observations': len(common_dates),
            'price_data': {
                'price1': price1_aligned,
                'price2': price2_aligned,
                'ratio': ratio
            }
        }
        
        # Stationarity tests
        ratio_stationarity = self.test_stationarity(ratio, f"{pair_name}_ratio")
        results['ratio_stationarity'] = ratio_stationarity
        
        # Cointegration test
        cointegration = self.test_cointegration(price1_aligned, price2_aligned, pair_name)
        results['cointegration'] = cointegration
        
        # Half-life calculation
        if 'spread_series' in cointegration and len(cointegration['spread_series']) > 20:
            half_life = self.calculate_half_life(cointegration['spread_series'], f"{pair_name}_spread")
        else:
            half_life = self.calculate_half_life(ratio, f"{pair_name}_ratio")
        results['half_life'] = half_life
        
        # Hurst exponent
        hurst = self.calculate_hurst_exponent(ratio, name=f"{pair_name}_ratio")
        results['hurst'] = hurst
        
        # Rolling stationarity
        rolling_stationarity = self.rolling_stationarity_test(ratio, name=f"{pair_name}_ratio")
        results['rolling_stationarity'] = rolling_stationarity
        
        # Regime detection
        regime_detection = self.detect_regime_changes(ratio, name=f"{pair_name}_ratio")
        results['regime_detection'] = regime_detection
        
        # Calculate composite mean reversion score
        score_components = []
        
        # Stationarity score (40% weight)
        if 'error' not in ratio_stationarity:
            if ratio_stationarity['confidence'] == 'High' and 'Stationary' in ratio_stationarity['conclusion']:
                score_components.append(0.4)
            elif 'Possibly' in ratio_stationarity['conclusion']:
                score_components.append(0.2)
            else:
                score_components.append(0.0)
        
        # Cointegration score (30% weight)
        if 'error' not in cointegration:
            if cointegration['cointegrated']:
                score_components.append(0.3)
            else:
                score_components.append(0.0)
        
        # Half-life score (20% weight)
        if 'error' not in half_life:
            if half_life['is_mean_reverting'] and half_life['half_life_days'] < 100:
                score_components.append(0.2)
            elif half_life['is_mean_reverting']:
                score_components.append(0.1)
            else:
                score_components.append(0.0)
        
        # Hurst score (10% weight)
        if 'error' not in hurst:
            if hurst['interpretation'] == 'Mean Reverting':
                score_components.append(0.1)
            else:
                score_components.append(0.0)
        
        mean_reversion_score = sum(score_components)
        results['mean_reversion_score'] = mean_reversion_score
        
        # Trading recommendation
        if mean_reversion_score >= 0.7:
            recommendation = "Strong Mean Reversion - Highly Tradeable"
        elif mean_reversion_score >= 0.5:
            recommendation = "Moderate Mean Reversion - Tradeable with Caution"
        elif mean_reversion_score >= 0.3:
            recommendation = "Weak Mean Reversion - Not Recommended"
        else:
            recommendation = "No Mean Reversion - Avoid"
        
        results['recommendation'] = recommendation
        
        return results
    
    def screen_multiple_pairs(self, pairs_list, period='5y'):
        """
        Screen multiple pairs and rank by mean reversion strength
        """
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (symbol1, symbol2) in enumerate(pairs_list):
            progress = (i + 1) / len(pairs_list)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing pair {i+1}/{len(pairs_list)}: {symbol1}/{symbol2}")
            
            pair_result = self.comprehensive_pair_analysis(symbol1, symbol2, period=period)
            results.append(pair_result)
        
        progress_bar.empty()
        status_text.empty()
        
        # Create summary DataFrame
        summary_data = []
        for result in results:
            if 'error' not in result:
                summary_data.append({
                    'Pair': result['pair_name'],
                    'Mean Reversion Score': result['mean_reversion_score'],
                    'Recommendation': result['recommendation'],
                    'Stationarity': 'Pass' if 'Stationary' in result['ratio_stationarity'].get('conclusion', '') else 'Fail',
                    'Cointegration': 'Pass' if result['cointegration'].get('cointegrated', False) else 'Fail',
                    'Half-Life (days)': result['half_life'].get('half_life_days', np.inf),
                    'Hurst Exponent': result['hurst'].get('hurst_exponent', np.nan),
                    'Data Points': result['num_observations']
                })
            else:
                summary_data.append({
                    'Pair': result['pair_name'],
                    'Mean Reversion Score': 0.0,
                    'Recommendation': f"Error: {result['error']}",
                    'Stationarity': 'Error',
                    'Cointegration': 'Error',
                    'Half-Life (days)': np.inf,
                    'Hurst Exponent': np.nan,
                    'Data Points': 0
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Mean Reversion Score', ascending=False)
        
        return {
            'summary': summary_df,
            'detailed_results': results,
            'best_pairs': summary_df[summary_df['Mean Reversion Score'] >= 0.5],
            'analysis_date': pd.Timestamp.now()
        }