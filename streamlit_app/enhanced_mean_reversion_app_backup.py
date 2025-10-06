"""
Enhanced Sector Rotation Strategy Dashboard with Mean Reversion Testing
======================================================================

Advanced Streamlit application that validates mean reversion assumptions before
implementing strategies. Includes comprehensive statistical testing framework.

Features:
- Statistical validation of mean reversion
- Stationarity tests (ADF, KPSS)
- Cointegration analysis
- Half-life calculations
- Hurst exponent analysis
- Regime detection
- Comprehensive pair screening

Author: Research Team
Date: October 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from mean_reversion_tests import MeanReversionTester

# Page configuration
st.set_page_config(
    page_title="Enhanced Mean Reversion Strategy Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Strategy configurations
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
}
}

FACTOR_STRATEGIES = {
'VUG_VYM': {
'symbols': ['VUG', 'VYM'],
'name': 'Growth vs Dividend',
'description': 'Vanguard Growth ETF vs Vanguard High Dividend Yield ETF',
'color': '#E74C3C'
},
'VUG_VTV': {
'symbols': ['VUG', 'VTV'],
'name': 'Growth vs Value',
'description': 'Vanguard Growth ETF vs Vanguard Value ETF',
'color': '#9B59B6'
},
'IWM_SPY': {
'symbols': ['IWM', 'SPY'],
'name': 'Small Cap vs Large Cap',
'description': 'Russell 2000 vs S&P 500',
'color': '#3498DB'
}
}

# Initialize tester
if 'tester' not in st.session_state:
st.session_state.tester = MeanReversionTester()

def main():
"""Main application function"""

# Header
    st.title("Enhanced Mean Reversion Strategy Dashboard")
    st.markdown("### Statistical validation and analysis of mean reversion strategies")

# Sidebar navigation
analysis_type = st.sidebar.radio(
"Analysis Type:",
[
"Single Pair Analysis", 
"Multiple Pair Screening",
"Strategy Implementation",
"Quick Strategy Overview",
"Scoring Methodology"
],
index=0
)

if analysis_type == "Single Pair Analysis":
single_pair_analysis()
elif analysis_type == "Multiple Pair Screening":
multiple_pair_screening()
elif analysis_type == "Strategy Implementation":
strategy_implementation()
elif analysis_type == "Scoring Methodology":
scoring_methodology()
else:
quick_overview()

def single_pair_analysis():
"""Detailed analysis of a single trading pair"""
    st.header(" Single Pair Statistical Analysis")
    st.markdown("Comprehensive mean reversion testing for individual trading pairs")

# Sidebar configuration
    st.sidebar.header("Pair Configuration")

# Strategy category selection
strategy_category = st.sidebar.selectbox(
"Strategy Category:",
["Bond Strategies", "Geographic Strategies", "Factor Strategies", "Custom Pair"],
index=0
)

if strategy_category == "Custom Pair":
symbol1 = st.sidebar.text_input("First Symbol:", value="SPY").upper()
symbol2 = st.sidebar.text_input("Second Symbol:", value="QQQ").upper()
pair_name = f"{symbol1}/{symbol2}"
else:
# Select from predefined strategies
if strategy_category == "Bond Strategies":
strategies = BOND_STRATEGIES
elif strategy_category == "Geographic Strategies":
strategies = GEOGRAPHIC_STRATEGIES
else:
strategies = FACTOR_STRATEGIES

selected_strategy = st.sidebar.selectbox(
"Select Strategy:",
list(strategies.keys()),
format_func=lambda x: f"{strategies[x]['name']}"
)

symbol1, symbol2 = strategies[selected_strategy]['symbols']
pair_name = strategies[selected_strategy]['name']

# Analysis parameters
period = st.sidebar.selectbox(
"Data Period:",
["1y", "2y", "3y", "5y", "max"],
index=3
)

# Run analysis button
if st.sidebar.button(" Run Statistical Analysis", type="primary"):
st.session_state.run_single_analysis = True
st.session_state.analysis_symbols = (symbol1, symbol2, pair_name, period)

# Display analysis results
if st.session_state.get('run_single_analysis', False):
symbol1, symbol2, pair_name, period = st.session_state.analysis_symbols

with st.spinner(f"Running comprehensive analysis for {symbol1}/{symbol2}..."):
result = st.session_state.tester.comprehensive_pair_analysis(
symbol1, symbol2, pair_name, period
)

if 'error' in result:
st.error(f"Analysis failed: {result['error']}")
return

# Display results
display_single_pair_results(result)

# Reset flag
st.session_state.run_single_analysis = False

def display_single_pair_results(result):
"""Display comprehensive results for a single pair"""

# Header with key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
score = result['mean_reversion_score']
    st.metric(
"Mean Reversion Score",
f"{score:.2f}",
help="Composite score from 0 (no mean reversion) to 1 (strong mean reversion)"
)

with col2:
stationarity = result['ratio_stationarity']
if 'error' not in stationarity:
status = " Pass" if 'Stationary' in stationarity['conclusion'] else " Fail"
else:
status = " Error"
st.metric("Stationarity Test", status)

with col3:
cointegration = result['cointegration']
if 'error' not in cointegration:
status = " Pass" if cointegration['cointegrated'] else " Fail"
else:
status = " Error"
st.metric("Cointegration Test", status)

with col4:
half_life = result['half_life']
if 'error' not in half_life:
hl_days = half_life['half_life_days']
if hl_days < np.inf:
st.metric("Half-Life", f"{hl_days:.0f} days")
else:
st.metric("Half-Life", "∞ (No reversion)")
else:
st.metric("Half-Life", "Error")

# Recommendation banner
recommendation = result['recommendation']
if "Strong" in recommendation:
st.success(f" **Recommendation:** {recommendation}")
elif "Moderate" in recommendation:
st.warning(f" **Recommendation:** {recommendation}")
else:
st.error(f" **Recommendation:** {recommendation}")

# Detailed test results
st.subheader(" Detailed Test Results")

# Create tabs for different tests
tab1, tab2, tab3, tab4, tab5 = st.tabs([
"Stationarity", "Cointegration", "Half-Life", "Hurst Exponent", "Regime Analysis"
])

with tab1:
display_stationarity_results(result['ratio_stationarity'])

with tab2:
display_cointegration_results(result['cointegration'])

with tab3:
display_half_life_results(result['half_life'])

with tab4:
display_hurst_results(result['hurst'])

with tab5:
display_regime_results(result['regime_detection'])

# Visualization
st.subheader(" Visual Analysis")
create_comprehensive_charts(result)

def display_stationarity_results(stationarity):
"""Display stationarity test results"""
if 'error' in stationarity:
    st.error(f"Stationarity test error: {stationarity['error']}")
return

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Augmented Dickey-Fuller Test**")
    st.write(f"• Statistic: {stationarity['adf_statistic']:.4f}")
st.write(f"• P-value: {stationarity['adf_pvalue']:.4f}")
st.write(f"• Critical Value (5%): {stationarity['adf_critical_5pct']:.4f}")
st.write(f"• Result: {' Stationary' if stationarity['is_stationary_adf'] else ' Non-stationary'}")

with col2:
st.markdown("**KPSS Test**")
st.write(f"• Statistic: {stationarity['kpss_statistic']:.4f}")
st.write(f"• P-value: {stationarity['kpss_pvalue']:.4f}")
st.write(f"• Critical Value (5%): {stationarity['kpss_critical_5pct']:.4f}")
st.write(f"• Result: {' Stationary' if stationarity['is_stationary_kpss'] else ' Non-stationary'}")

st.info(f"**Overall Conclusion:** {stationarity['conclusion']} (Confidence: {stationarity['confidence']})")

def display_cointegration_results(cointegration):
"""Display cointegration test results"""
if 'error' in cointegration:
    st.error(f"Cointegration test error: {cointegration['error']}")
return

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Engle-Granger Cointegration Test**")
    st.write(f"• Test Statistic: {cointegration['coint_statistic']:.4f}")
st.write(f"• P-value: {cointegration['coint_pvalue']:.4f}")
st.write(f"• Critical Value (5%): {cointegration['coint_critical_5pct']:.4f}")
st.write(f"• Result: {' Cointegrated' if cointegration['cointegrated'] else ' Not cointegrated'}")

with col2:
st.markdown("**Cointegration Details**")
st.write(f"• Hedge Ratio: {cointegration['hedge_ratio']:.4f}")
st.write(f"• Direction: {cointegration['direction']}")
st.write(f"• Sample Size: {cointegration['sample_size']} observations")

def display_half_life_results(half_life):
"""Display half-life calculation results"""
if 'error' in half_life:
    st.error(f"Half-life calculation error: {half_life['error']}")
return

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Mean Reversion Analysis**")
hl_days = half_life['half_life_days']
if hl_days < np.inf:
st.write(f"• Half-Life: {hl_days:.1f} days")
st.write(f"• Time to 90% reversion: ~{hl_days * 3.3:.0f} days")
else:
st.write("• Half-Life: ∞ (No mean reversion)")

st.write(f"• Mean Reversion: {' Yes' if half_life['is_mean_reverting'] else ' No'}")
st.write(f"• Reversion Strength: {half_life['reversion_strength']:.4f}")

with col2:
st.markdown("**Statistical Details**")
st.write(f"• AR(1) Coefficient: {half_life['mean_revert_coefficient']:.4f}")
st.write(f"• Coefficient P-value: {half_life['coefficient_pvalue']:.4f}")
st.write(f"• R-squared: {half_life['r_squared']:.4f}")
st.write(f"• Sample Size: {half_life['sample_size']} observations")

def display_hurst_results(hurst):
"""Display Hurst exponent results"""
if 'error' in hurst:
    st.error(f"Hurst exponent calculation error: {hurst['error']}")
return

hurst_value = hurst['hurst_exponent']
interpretation = hurst['interpretation']

col1, col2 = st.columns(2)

with col1:
st.markdown("**Hurst Exponent Analysis**")
st.write(f"• Hurst Exponent: {hurst_value:.4f}")
st.write(f"• Interpretation: {interpretation}")

if hurst_value < 0.45:
st.success(" Strong mean reversion tendency")
elif hurst_value < 0.5:
st.info(" Weak mean reversion tendency")
elif hurst_value > 0.55:
st.warning(" Trending behavior")
else:
st.info(" Random walk behavior")

with col2:
st.markdown("**Analysis Details**")
st.write(f"• Sample Size: {hurst['sample_size']} observations")
st.write(f"• Lags Used: {hurst['lags_used']}")
st.write("• **Hurst Interpretation:**")
st.write(" - H < 0.5: Mean reverting")
st.write(" - H = 0.5: Random walk")
st.write(" - H > 0.5: Trending")

def display_regime_results(regime_detection):
"""Display regime detection results"""
if 'error' in regime_detection:
    st.error(f"Regime detection error: {regime_detection['error']}")
return

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Regime Analysis**")
    st.write(f"• Number of Regime Changes: {regime_detection['num_regime_changes']}")
st.write(f"• Average Regime Length: {regime_detection['avg_regime_length_days']:.1f} days")
st.write(f"• Current Regime: {regime_detection['current_regime']}")

with col2:
st.markdown("**Regime Characteristics**")
regime_0 = regime_detection['regime_0_characteristics']
regime_1 = regime_detection['regime_1_characteristics']

st.write("**Regime 0:**")
st.write(f" - Volatility: {regime_0['volatility']:.4f}")
st.write(f" - Autocorr: {regime_0['autocorr']:.4f}")
st.write(f" - Periods: {regime_0['periods']}")

st.write("**Regime 1:**")
st.write(f" - Volatility: {regime_1['volatility']:.4f}")
st.write(f" - Autocorr: {regime_1['autocorr']:.4f}")
st.write(f" - Periods: {regime_1['periods']}")

def create_comprehensive_charts(result):
"""Create comprehensive visualization charts"""
price_data = result['price_data']
ratio = price_data['ratio']
price1 = price_data['price1']
price2 = price_data['price2']

# Create subplots
fig = make_subplots(
rows=4, cols=1,
subplot_titles=[
f"Price Evolution: {result['symbol1']} vs {result['symbol2']}",
"Price Ratio",
"Rolling Stationarity Analysis",
"Regime Detection"
],
vertical_spacing=0.08
)

# 1. Price evolution
fig.add_trace(
go.Scatter(x=price1.index, y=price1.values, 
name=result['symbol1'], line=dict(color='blue')),
row=1, col=1
)
fig.add_trace(
go.Scatter(x=price2.index, y=price2.values, 
name=result['symbol2'], line=dict(color='red')),
row=1, col=1
)

# 2. Price ratio
fig.add_trace(
go.Scatter(x=ratio.index, y=ratio.values, 
name='Price Ratio', line=dict(color='purple')),
row=2, col=1
)

# Add rolling mean and bands
if len(ratio) > 60:
rolling_mean = ratio.rolling(60).mean()
rolling_std = ratio.rolling(60).std()

fig.add_trace(
go.Scatter(x=rolling_mean.index, y=rolling_mean.values,
name='60-day Mean', line=dict(color='orange', dash='dash')),
row=2, col=1
)

fig.add_trace(
go.Scatter(x=rolling_mean.index, y=rolling_mean + 2*rolling_std,
fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False),
row=2, col=1
)
fig.add_trace(
go.Scatter(x=rolling_mean.index, y=rolling_mean - 2*rolling_std,
fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
name='±2σ Band', fillcolor='rgba(128,128,128,0.2)'),
row=2, col=1
)

# 3. Rolling stationarity (if available)
rolling_results = result.get('rolling_stationarity', {})
if 'rolling_results' in rolling_results:
rolling_df = rolling_results['rolling_results']
fig.add_trace(
go.Scatter(x=rolling_df['date'], y=rolling_df['adf_pvalue'],
name='ADF P-value', line=dict(color='green')),
row=3, col=1
)
fig.add_hline(y=0.05, line_dash="dash", line_color="red", row=3, col=1)

# 4. Regime detection (if available)
regime_results = result.get('regime_detection', {})
if 'regime_series' in regime_results:
regime_series = regime_results['regime_series']
fig.add_trace(
go.Scatter(x=regime_series.index, y=regime_series.values,
name='Regime', mode='markers', marker=dict(color=regime_series.values, colorscale='viridis')),
row=4, col=1
)

fig.update_layout(height=1000, showlegend=True, 
title_text=f"Comprehensive Analysis: {result['pair_name']}")
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

st.plotly_chart(fig, use_container_width=True)

def multiple_pair_screening():
"""Screen multiple pairs for mean reversion"""
    st.header(" Multiple Pair Screening")
    st.markdown("Comprehensive screening and ranking of trading pairs by mean reversion strength")

# Configuration
    st.sidebar.header("Screening Configuration")

screening_type = st.sidebar.selectbox(
"Screening Type:",
["All Predefined Strategies", "Bond Strategies Only", "Geographic Strategies Only", "Factor Strategies Only", "Custom List"],
index=0
)

period = st.sidebar.selectbox(
"Data Period:",
["1y", "2y", "3y", "5y", "max"],
index=3
)

# Build pairs list
pairs_list = []

if screening_type == "All Predefined Strategies":
for strategy_dict in [BOND_STRATEGIES, GEOGRAPHIC_STRATEGIES, FACTOR_STRATEGIES]:
for config in strategy_dict.values():
pairs_list.append(tuple(config['symbols']))
elif screening_type == "Bond Strategies Only":
for config in BOND_STRATEGIES.values():
pairs_list.append(tuple(config['symbols']))
elif screening_type == "Geographic Strategies Only":
for config in GEOGRAPHIC_STRATEGIES.values():
pairs_list.append(tuple(config['symbols']))
elif screening_type == "Factor Strategies Only":
for config in FACTOR_STRATEGIES.values():
pairs_list.append(tuple(config['symbols']))
else: # Custom List
st.sidebar.markdown("**Add Custom Pairs:**")
custom_pairs = st.sidebar.text_area(
"Enter pairs (one per line, format: SYMBOL1,SYMBOL2):",
value="SPY,QQQ\nHYG,TLT\nVUG,VYM"
)
pairs_list = []
for line in custom_pairs.strip().split('\n'):
if ',' in line:
symbols = [s.strip().upper() for s in line.split(',')]
if len(symbols) == 2:
pairs_list.append(tuple(symbols))

# Display pairs to be screened
st.write(f"**Pairs to screen:** {len(pairs_list)}")
with st.expander("View pairs list"):
for i, (s1, s2) in enumerate(pairs_list, 1):
st.write(f"{i}. {s1}/{s2}")

# Run screening
if st.sidebar.button(" Run Screening Analysis", type="primary"):
if len(pairs_list) == 0:
st.error("No pairs to screen. Please select or add pairs.")
return

with st.spinner(f"Screening {len(pairs_list)} pairs..."):
screening_results = st.session_state.tester.screen_multiple_pairs(pairs_list, period)

# Display results
display_screening_results(screening_results)

def display_screening_results(results):
"""Display multiple pair screening results"""
summary_df = results['summary']
detailed_results = results['detailed_results']
best_pairs = results['best_pairs']

# Summary statistics
    st.subheader(" Screening Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
st.metric("Total Pairs Analyzed", len(summary_df))

with col2:
strong_pairs = len(summary_df[summary_df['Mean Reversion Score'] >= 0.7])
st.metric("Strong Mean Reversion", strong_pairs)

with col3:
moderate_pairs = len(summary_df[summary_df['Mean Reversion Score'].between(0.5, 0.7)])
st.metric("Moderate Mean Reversion", moderate_pairs)

with col4:
avg_score = summary_df['Mean Reversion Score'].mean()
st.metric("Average Score", f"{avg_score:.3f}")

# Results table
st.subheader(" Screening Results")

# Color code the dataframe
def color_score(val):
if val >= 0.7:
return 'background-color: #d4edda' # Green
elif val >= 0.5:
return 'background-color: #fff3cd' # Yellow
elif val >= 0.3:
return 'background-color: #f8d7da' # Red
else:
return 'background-color: #f1f1f1' # Gray

styled_df = summary_df.style.applymap(color_score, subset=['Mean Reversion Score'])
st.dataframe(styled_df, use_container_width=True)

# Best pairs section
if not best_pairs.empty:
st.subheader(" Recommended Trading Pairs")
st.markdown("Pairs with mean reversion score ≥ 0.5")

for _, row in best_pairs.iterrows():
with st.expander(f" {row['Pair']} (Score: {row['Mean Reversion Score']:.3f})"):
# Find detailed results for this pair
pair_detail = None
for detail in detailed_results:
if detail['pair_name'] == row['Pair']:
pair_detail = detail
break

if pair_detail and 'error' not in pair_detail:
col1, col2, col3 = st.columns(3)

with col1:
st.write("**Statistical Tests:**")
st.write(f"• Stationarity: {row['Stationarity']}")
st.write(f"• Cointegration: {row['Cointegration']}")
st.write(f"• Half-Life: {row['Half-Life (days)']:.0f} days")

with col2:
st.write("**Risk Metrics:**")
st.write(f"• Hurst Exponent: {row['Hurst Exponent']:.3f}")
regime = pair_detail.get('regime_detection', {})
if 'num_regime_changes' in regime:
st.write(f"• Regime Changes: {regime['num_regime_changes']}")
st.write(f"• Data Points: {row['Data Points']}")

with col3:
st.write("**Implementation:**")
st.write(f"• Recommendation: {row['Recommendation']}")

# Quick implementation notes
if row['Half-Life (days)'] < 30:
st.write("• Strategy: Short-term mean reversion")
elif row['Half-Life (days)'] < 90:
st.write("• Strategy: Medium-term rotation")
else:
st.write("• Strategy: Long-term allocation")

else:
st.error(f"Detailed analysis not available: {row['Recommendation']}")

# Download results
csv = summary_df.to_csv(index=False)
st.download_button(
label=" Download Screening Results",
data=csv,
file_name=f"mean_reversion_screening_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
mime="text/csv"
)

def strategy_implementation():
"""Strategy implementation with validated pairs"""
    st.header(" Strategy Implementation")
    st.markdown("Implement mean reversion strategies using statistically validated pairs")

    st.info(" **Coming Soon:** Advanced strategy implementation with backtesting, parameter optimization, and live trading signals based on validated mean reversion relationships.")

# For now, show a preview of what's coming
    st.subheader(" Planned Features")

col1, col2 = st.columns(2)

with col1:
st.markdown("""
**Statistical Strategy Builder:**
- Parameter optimization based on half-life
- Dynamic threshold adjustment
- Regime-aware position sizing
- Out-of-sample validation
""")

with col2:
st.markdown("""
**Advanced Analytics:**
- Real-time regime detection
- Cointegration stability monitoring
- Performance attribution
- Risk management alerts
""")

def quick_overview():
"""Quick overview of strategies"""
    st.header(" Quick Strategy Overview")
    st.markdown("Overview of all available strategy categories")

# Strategy category tabs
tab1, tab2, tab3 = st.tabs(["Bond Strategies", "Geographic Strategies", "Factor Strategies"])

with tab1:
    st.subheader(" Bond Strategies")
for key, config in BOND_STRATEGIES.items():
st.write(f"**{config['name']}:** {config['description']}")

with tab2:
st.subheader(" Geographic Strategies")
for key, config in GEOGRAPHIC_STRATEGIES.items():
st.write(f"**{config['name']}:** {config['description']}")

with tab3:
st.subheader(" Factor Strategies")
for key, config in FACTOR_STRATEGIES.items():
st.write(f"**{config['name']}:** {config['description']}")

def scoring_methodology():
"""Detailed explanation of the mean reversion scoring methodology"""
    st.header("🧮 Mean Reversion Scoring Methodology")
    st.markdown("### Understanding how pairs are evaluated and ranked")

# Overview
    st.subheader(" Overview")
    st.markdown("""
The **Mean Reversion Score** is a composite metric ranging from 0.0 to 1.0 that quantifies 
the strength of mean-reverting behavior in a trading pair. Higher scores indicate stronger 
statistical evidence for mean reversion and better trading opportunities.
""")

# Score breakdown
st.subheader(" Score Components (Weighted)")

col1, col2 = st.columns([1, 2])

with col1:
st.markdown("""
**Component Weights:**
- Stationarity: **40%**
- Cointegration: **30%** 
- ⏱ Half-Life: **20%**
- Hurst Exponent: **10%**
""")

with col2:
# Create a visual breakdown
import plotly.graph_objects as go

labels = ['Stationarity (40%)', 'Cointegration (30%)', 'Half-Life (20%)', 'Hurst Exponent (10%)']
values = [0.4, 0.3, 0.2, 0.1]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, 
marker_colors=colors, hole=0.4)])
fig.update_layout(title="Score Component Weights", height=300)
st.plotly_chart(fig, use_container_width=True)

# Detailed component explanations
st.subheader(" Component Details")

# Stationarity section
with st.expander(" Stationarity Tests (40% Weight) - Most Important"):
st.markdown("""
**What it measures:** Whether the price ratio returns to its mean over time

**Tests performed:**
- **ADF (Augmented Dickey-Fuller):** Tests null hypothesis of non-stationarity
- **KPSS:** Tests null hypothesis of stationarity

**Scoring logic:**
- **0.4 points:** Both tests confirm stationarity (High confidence)
- **0.2 points:** Only one test confirms stationarity (Medium confidence) 
- **0.0 points:** Both tests indicate non-stationarity

**Why it's weighted highest:** Stationarity is the foundation of mean reversion. 
Without it, the ratio will trend indefinitely rather than revert.

**Interpretation:**
- **Stationary:** Ratio tends to revert to historical mean
- **Non-stationary:** Ratio follows random walk or trend
""")

# Cointegration section
with st.expander(" Cointegration Analysis (30% Weight) - Critical"):
st.markdown("""
**What it measures:** Long-term equilibrium relationship between price series

**Test performed:**
- **Engle-Granger Test:** Tests for cointegrating relationship in both directions

**Scoring logic:**
- **0.3 points:** Pairs are cointegrated (p-value < 0.05)
- **0.0 points:** No cointegration detected

**Why it's important:** Cointegration ensures that even if prices drift apart 
temporarily, they will be pulled back together by economic forces.

**Practical meaning:**
- **Cointegrated:** Prices move together long-term, deviations are temporary
- **Not cointegrated:** No long-term relationship, pairs may drift apart permanently
""")

# Half-life section
with st.expander("⏱ Half-Life Analysis (20% Weight) - Timing"):
st.markdown("""
**What it measures:** How quickly mean reversion occurs

**Calculation:** AR(1) model: Δy = α + βy(t-1) + ε
- Half-life = -ln(2) / ln(1 + β)

**Scoring logic:**
- **0.2 points:** Significant mean reversion (β < 0, p-value < 0.05) AND half-life < 100 days
- **0.1 points:** Significant mean reversion but slower (half-life > 100 days)
- **0.0 points:** No significant mean reversion detected

**Trading implications:**
- **< 30 days:** High-frequency trading opportunity
- **30-90 days:** Medium-term rotation strategy 
- **> 90 days:** Long-term allocation strategy
- **∞ (infinite):** No mean reversion - avoid trading
""")

# Hurst exponent section
with st.expander(" Hurst Exponent (10% Weight) - Behavioral Confirmation"):
st.markdown("""
**What it measures:** Autocorrelation structure and trending behavior

**Calculation:** Rescaled range analysis over multiple time lags

**Interpretation:**
- **H < 0.5:** Mean-reverting behavior (anti-persistent)
- **H = 0.5:** Random walk (no predictable pattern)
- **H > 0.5:** Trending behavior (persistent)

**Scoring logic:**
- **0.1 points:** H < 0.5 (mean-reverting interpretation)
- **0.0 points:** H ≥ 0.5 (random walk or trending)

**Why lowest weight:** Confirms other tests but less reliable for trading signals
""")

# Score interpretation
st.subheader(" Score Interpretation & Trading Recommendations")

# Create score ranges table
score_data = {
"Score Range": ["0.90 - 1.00", "0.70 - 0.89", "0.50 - 0.69", "0.30 - 0.49", "0.00 - 0.29"],
"Classification": [
" Exceptional", 
" Strong Mean Reversion", 
" Moderate Mean Reversion", 
" Weak Mean Reversion", 
" No Mean Reversion"
],
"Trading Recommendation": [
"Highest priority - Deploy significant capital",
"Highly tradeable - Core strategy pairs", 
"Tradeable with caution - Monitor closely",
"Not recommended - High risk",
"Avoid - No statistical edge"
],
"Expected Characteristics": [
"All tests pass, fast reversion, stable relationship",
"Most tests pass, good reversion speed", 
"Mixed test results, moderate reversion",
"Few tests pass, slow/unreliable reversion",
"Tests fail, no reversion evidence"
]
}

score_df = pd.DataFrame(score_data)
st.dataframe(score_df, use_container_width=True, hide_index=True)

# Statistical significance
st.subheader(" Statistical Significance Thresholds")

col1, col2 = st.columns(2)

with col1:
st.markdown("""
**P-value Thresholds:**
- 0.05 (5%): Standard significance level
- 0.01 (1%): High confidence level

**Critical Values:**
- ADF: More negative = more stationary
- KPSS: Less than critical = stationary
""")

with col2:
st.markdown("""
**Confidence Levels:**
- **High:** Both stationarity tests agree
- **Medium:** Tests disagree, use caution 
- **Low:** Neither test supports stationarity
""")

# Practical examples
st.subheader(" Practical Examples")

example_tab1, example_tab2, example_tab3 = st.tabs(["Strong Pair", "Weak Pair", "Failed Pair"])

with example_tab1:
st.markdown("""
**Example: HYG/TLT (Score: 0.85)**

**Stationarity (0.4/0.4):** Both ADF and KPSS confirm stationarity
- ADF p-value: 0.001 → Strongly stationary
- KPSS p-value: 0.10 → Confirms stationarity

**Cointegration (0.3/0.3):** Strong cointegrating relationship 
- Engle-Granger p-value: 0.02 → Cointegrated

**Half-life (0.2/0.2):** Fast mean reversion
- Half-life: 45 days → Good for medium-term strategy
- AR(1) coefficient: -0.08 (p < 0.01)

**Hurst (0.0/0.1):** Slightly trending
- Hurst exponent: 0.52 → Mild trending behavior

**Total Score: 0.85** → **Strong Mean Reversion - Highly Tradeable**
""")

with example_tab2:
st.markdown("""
**Example: SPY/QQQ (Score: 0.45)**

**Stationarity (0.2/0.4):** Mixed signals
- ADF p-value: 0.03 → Weakly stationary 
- KPSS p-value: 0.02 → Suggests non-stationarity

**Cointegration (0.0/0.3):** No cointegration
- Engle-Granger p-value: 0.15 → Not cointegrated

**Half-life (0.1/0.2):** Slow reversion
- Half-life: 120 days → Very slow mean reversion
- AR(1) coefficient: -0.02 (p < 0.05)

**Hurst (0.1/0.1):** Mean-reverting tendency
- Hurst exponent: 0.48 → Slight mean reversion

**Total Score: 0.45** → **Weak Mean Reversion - Not Recommended**
""")

with example_tab3:
st.markdown("""
**Example: VUG/ARKK (Score: 0.15)**

**Stationarity (0.0/0.4):** Non-stationary
- ADF p-value: 0.25 → Non-stationary
- KPSS p-value: 0.01 → Confirms non-stationarity

**Cointegration (0.0/0.3):** No relationship
- Engle-Granger p-value: 0.65 → Not cointegrated

**Half-life (0.0/0.2):** No mean reversion
- Half-life: ∞ → No reversion detected
- AR(1) coefficient: +0.01 (p = 0.50)

**Hurst (0.1/0.1):** Mean-reverting tendency
- Hurst exponent: 0.45 → Some mean reversion signals

**Total Score: 0.15** → **No Mean Reversion - Avoid Trading**
""")

# Best practices
st.subheader(" Best Practices")

st.markdown("""
**For High-Quality Strategies:**
1. **Focus on scores ≥ 0.70** for core trading strategies
2. **Verify recent stability** using rolling analysis
3. **Monitor regime changes** that might break relationships
4. **Use half-life for position sizing** and holding periods
5. **Combine with fundamental analysis** for context

**Risk Management:**
- Lower scores = smaller position sizes
- Monitor cointegration stability over time 
- Set stop-losses based on historical deviation ranges
- Diversify across multiple high-scoring pairs
""")

# Footer note
st.info("""
**Remember:** Statistical tests are based on historical data and don't guarantee future performance. 
Market regimes can change, breaking previously strong mean-reverting relationships. Always combine 
statistical analysis with fundamental research and proper risk management.
""")

if __name__ == "__main__":
main()