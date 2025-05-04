import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
import io

# Page configuration
st.set_page_config(
    page_title="Stocks Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern UI styling with blue theme instead of orange
st.markdown("""
<style>
    /* Overall page styling */
    .main {
        background-color: #f8f9fa;
        padding: 2rem;
    }
    
    /* Headers */
    h1 {
        color: #0d4b9f;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
    }
    
    h2, h3 {
        color: #334155;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
    }
    
    /* Containers */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
        padding: 2rem 1rem;
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 0;
    }
    
    section[data-testid="stSidebar"] h2 {
        margin-top: 0;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1e40af;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #1e3a8a;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* DataFrames */
    .dataframe {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    
    .dataframe th {
        background-color: #f1f5f9;
        color: #334155;
        font-weight: 600;
        border: none !important;
        text-align: left !important;
    }
    
    .dataframe td {
        border-bottom: 1px solid #e2e8f0 !important;
        border-left: none !important;
        border-right: none !important;
        text-align: left !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 1.5rem;
        border: none;
        border-bottom: 2px solid transparent;
        font-weight: 500;
        color: #64748b;
        background-color: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        border-bottom: 2px solid #1e40af !important;
        color: #1e40af !important;
        background-color: transparent !important;
    }
    
    /* Radio buttons */
    div[role="radiogroup"] label {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        margin-right: 0.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    div[role="radiogroup"] label:hover {
        border-color: #cbd5e1;
        background-color: #f8fafc;
    }
    
    div[role="radiogroup"] [data-baseweb="radio"] input:checked + div {
        border-color: #2e7d32;
        background-color: #e8f5e9;
    }
    
    /* Select boxes */
    div[data-baseweb="select"] > div {
        border-radius: 6px !important;
        border-color: #e2e8f0 !important;
        background-color: white;
    }
    
    div[data-baseweb="select"] > div:hover {
        border-color: #cbd5e1 !important;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 6px;
    }
    
    /* Fix for dark mode */
    @media (prefers-color-scheme: dark) {
        .stApp, body, [data-testid="stAppViewContainer"] {
            background-color: #0e1117;
        }
        
        h1, h2, h3, p, span, div {
            color: #f8f9fa;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #f8f9fa;
        }
        
        section[data-testid="stSidebar"] {
            background-color: #262730;
            border-right: 1px solid #4b5563;
        }
        
        .dataframe th {
            background-color: #1e293b;
            color: #f8f9fa;
        }
        
        .dataframe td {
            border-bottom: 1px solid #4b5563 !important;
            color: #f8f9fa;
        }
    }
</style>
""", unsafe_allow_html=True)

# Define stock markets data
us_indices = {
    'S&P 500': '^GSPC',
    'Dow Jones': '^DJI',
    'NASDAQ': '^IXIC'
}

india_indices = {
    'NIFTY 50': '^NSEI',
    'SENSEX': '^BSESN',
    'NIFTY BANK': '^NSEBANK'
}

# Function to load stock lists
@st.cache_data(ttl=86400)
def load_stock_lists():
    # Load US Stocks from CSV
    try:
        us_stocks = pd.read_csv('data/us_stocks.csv')  # Adjust path as needed
        # Ensure the CSV has the required columns
        if not all(col in us_stocks.columns for col in ['symbol', 'name']):
            us_stocks = us_stocks.rename(columns={
                # Map your actual column names to the required ones
                'Symbol': 'symbol',  # Example mapping
                'Company': 'name'    # Example mapping
            })
    except Exception as e:
        st.warning(f"Failed to load US stocks CSV: {e}. Using default list.")
        us_stocks = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT'],
            'name': ['Apple', 'Microsoft', 'Amazon', 'Alphabet', 'Meta Platforms', 'Tesla', 'NVIDIA', 'JPMorgan Chase', 'Visa', 'Walmart']
        })
    
    # Load Indian Stocks from CSV
    try:
        india_stocks = pd.read_csv('data/india_stocks.csv')  # Adjust path as needed
        # Ensure the CSV has the required columns
        if not all(col in india_stocks.columns for col in ['symbol', 'name']):
            india_stocks = india_stocks.rename(columns={
                # Map your actual column names to the required ones
                'Symbol': 'symbol',  # Example mapping
                'Company': 'name'    # Example mapping
            })
        
        # Ensure Indian stock symbols have .NS suffix
        india_stocks['symbol'] = india_stocks['symbol'].apply(
            lambda x: x if str(x).endswith('.NS') else f"{x}.NS"
        )
    except Exception as e:
        st.warning(f"Failed to load India stocks CSV: {e}. Using default list.")
        india_stocks = pd.DataFrame({
            'symbol': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 
                     'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS'],
            'name': ['Reliance Industries', 'Tata Consultancy Services', 'HDFC Bank', 'Infosys', 
                    'ICICI Bank', 'Hindustan Unilever', 'ITC', 'State Bank of India', 
                    'Bajaj Finance', 'Bharti Airtel']
        })
    
    return us_stocks, india_stocks

# Function to process uploaded stock list
def process_uploaded_stock_list(uploaded_file, market):
    try:
        # Read CSV file
        content = uploaded_file.read()
        stocks_df = pd.read_csv(io.BytesIO(content))
        
        # Standardize column names (case-insensitive)
        column_mapping = {}
        for col in stocks_df.columns:
            if col.lower() in ['symbol', 'ticker', 'stock']:
                column_mapping[col] = 'symbol'
            elif col.lower() in ['name', 'company', 'company name', 'stock name']:
                column_mapping[col] = 'name'
        
        # Rename columns if needed
        if column_mapping:
            stocks_df = stocks_df.rename(columns=column_mapping)
        
        # Check if we have the required columns
        if 'symbol' not in stocks_df.columns:
            raise ValueError("CSV must contain a 'symbol' column")
        
        # If no name column exists, create one with symbol values
        if 'name' not in stocks_df.columns:
            stocks_df['name'] = stocks_df['symbol']
        
        # Ensure proper formatting for Indian stocks
        if market == "India":
            stocks_df['symbol'] = stocks_df['symbol'].apply(
                lambda x: x if str(x).endswith('.NS') else f"{x}.NS"
            )
        
        # Limit to 9999 stocks
        if len(stocks_df) > 9999:
            stocks_df = stocks_df.iloc[:9999]
            st.warning(f"Stock list limited to 9999 stocks")
        
        return stocks_df
        
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        return None

# Function to get stock data and calculate EMAs
@st.cache_data(ttl=3600)
def get_stock_data(symbol, interval, period='365d'):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        
        if df.empty or len(df) < 20:  # Ensure we have enough data for EMAs
            return None
        
        # Calculate multiple EMAs
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        # Calculate if price is above or below each EMA
        df['Above_EMA20'] = df['Close'] > df['EMA20']
        df['Above_EMA50'] = df['Close'] > df['EMA50']
        df['Above_EMA100'] = df['Close'] > df['EMA100']
        df['Above_EMA200'] = df['Close'] > df['EMA200']

        # Add true crossover detection columns
        df['EMA20_Crossover'] = df['Above_EMA20'].astype(int).diff()
        df['EMA50_Crossover'] = df['Above_EMA50'].astype(int).diff()
        df['EMA100_Crossover'] = df['Above_EMA100'].astype(int).diff()
        df['EMA200_Crossover'] = df['Above_EMA200'].astype(int).diff()
        
        return df
    except Exception as e:
        return None

# Function to check if stock has crossed above selected EMA and stayed above for N candles
def check_ema_bullish_confirmation(df, ema_type, lookback=10, confirmation_candles=3):
    if df is None or df.empty or len(df) <= lookback + confirmation_candles:
        return False, None, None
    
    # Make sure we have at least 20 data points for reliable EMAs
    if len(df) < 20:
        return False, None, None
    
    # Get recent data for analysis
    recent_data = df.tail(lookback + confirmation_candles)
    
    # Check for crossover in the lookback period
    crossover_column = f"{ema_type}_Crossover"
    if crossover_column not in recent_data.columns:
        return False, None, None
    
    # Look for a positive crossover (1 means crossed above) within the lookback period
    # Excluding the confirmation period
    lookback_window = recent_data.iloc[:-confirmation_candles] if confirmation_candles > 0 else recent_data
    crossover_happened = (lookback_window[crossover_column] == 1).any()
    
    if not crossover_happened:
        return False, None, None
    
    # If confirmation_candles is 0, just check the current candle's position
    if confirmation_candles == 0:
        last_candle = recent_data.iloc[-1]
        above_column = f"Above_{ema_type}"
        is_confirmed = last_candle[above_column]
        
        if not is_confirmed:
            return False, None, None
    else:
        # Ensure all confirmation candles are above the EMA
        confirmation_window = recent_data.iloc[-confirmation_candles:]
        above_column = f"Above_{ema_type}"
        is_confirmed = confirmation_window[above_column].all()
        
        if not is_confirmed:
            return False, None, None
    
    # Check position relative to all EMAs
    current_ema_status = {
        'Above_EMA20': df['Above_EMA20'].iloc[-1],
        'Above_EMA50': df['Above_EMA50'].iloc[-1],
        'Above_EMA100': df['Above_EMA100'].iloc[-1],
        'Above_EMA200': df['Above_EMA200'].iloc[-1]
    }
    
    # Calculate distance from selected EMA
    last_price = df['Close'].iloc[-1]
    ema_value = df[ema_type].iloc[-1]
    distance_pct = ((last_price - ema_value) / ema_value) * 100
    
    # Return confirmation, EMA status, and distance
    return True, current_ema_status, distance_pct

# Function to check if stock has crossed below selected EMA and stayed below for N candles
def check_ema_bearish_confirmation(df, ema_type, lookback=10, confirmation_candles=3):
    if df is None or df.empty or len(df) <= lookback + confirmation_candles:
        return False, None, None
    
    # Make sure we have at least 20 data points for reliable EMAs
    if len(df) < 20:
        return False, None, None
    
    # Get recent data for analysis
    recent_data = df.tail(lookback + confirmation_candles)
    
    # Check for crossover in the lookback period
    crossover_column = f"{ema_type}_Crossover"
    if crossover_column not in recent_data.columns:
        return False, None, None
    
    # Look for a negative crossover (-1 means crossed below) within the lookback period
    # Excluding the confirmation period
    lookback_window = recent_data.iloc[:-confirmation_candles] if confirmation_candles > 0 else recent_data
    crossover_happened = (lookback_window[crossover_column] == -1).any()
    
    if not crossover_happened:
        return False, None, None
    
    # If confirmation_candles is 0, just check the current candle's position
    if confirmation_candles == 0:
        last_candle = recent_data.iloc[-1]
        above_column = f"Above_{ema_type}"
        is_confirmed = not last_candle[above_column]
        
        if not is_confirmed:
            return False, None, None
    else:
        # Ensure all confirmation candles are below the EMA
        confirmation_window = recent_data.iloc[-confirmation_candles:]
        above_column = f"Above_{ema_type}"
        is_confirmed = ~confirmation_window[above_column].any()
        
        if not is_confirmed:
            return False, None, None
    
    # Check position relative to all EMAs
    current_ema_status = {
        'Above_EMA20': df['Above_EMA20'].iloc[-1],
        'Above_EMA50': df['Above_EMA50'].iloc[-1],
        'Above_EMA100': df['Above_EMA100'].iloc[-1],
        'Above_EMA200': df['Above_EMA200'].iloc[-1]
    }
    
    # Calculate distance from selected EMA
    last_price = df['Close'].iloc[-1]
    ema_value = df[ema_type].iloc[-1]
    distance_pct = ((last_price - ema_value) / ema_value) * 100
    
    # Return confirmation, EMA status, and distance
    return True, current_ema_status, distance_pct

# Function to scan all stocks
def scan_stocks(stock_list, timeframe, market, ema_selection, lookback, confirmation_candles):
    buy_list = []
    sell_list = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_stocks = len(stock_list)
    processed_count = 0
    
    for i, (symbol, name) in enumerate(zip(stock_list['symbol'], stock_list['name'])):
        status_text.text(f"Scanning {market} stocks: {i+1}/{total_stocks} - {name} ({symbol})")
        progress_bar.progress((i + 1) / total_stocks)
        
        # Get stock data with more history to ensure accurate EMAs
        if timeframe in ['1h', '1d']:
            period = f"{max(200, lookback + confirmation_candles + 50)}d"
        else:
            period = f"{max(200, lookback + confirmation_candles + 20)}wk"
            
        df = get_stock_data(symbol, timeframe, period=period)
        
        if df is None or df.empty:
            continue
            
        processed_count += 1
            
        # Check bullish confirmation
        bullish_confirmed, bull_ema_status, bull_distance = check_ema_bullish_confirmation(
            df, ema_selection, lookback, confirmation_candles
        )
        
        if bullish_confirmed:
            last_price = df['Close'].iloc[-1]
            ema_value = df[ema_selection].iloc[-1]
            
            # Create EMA status string
            ema_status_str = "Above: "
            ema_status_str += "20✓ " if bull_ema_status['Above_EMA20'] else "20✗ "
            ema_status_str += "50✓ " if bull_ema_status['Above_EMA50'] else "50✗ "
            ema_status_str += "100✓ " if bull_ema_status['Above_EMA100'] else "100✗ "
            ema_status_str += "200✓" if bull_ema_status['Above_EMA200'] else "200✗"
                
            buy_list.append({
                'Symbol': symbol,
                'Name': name,
                'Price': last_price,
                'Selected_EMA': ema_value,
                'EMA_Status': ema_status_str,
                'Distance': bull_distance
            })
        
        # Check bearish confirmation
        bearish_confirmed, bear_ema_status, bear_distance = check_ema_bearish_confirmation(
            df, ema_selection, lookback, confirmation_candles
        )
        
        if bearish_confirmed:
            last_price = df['Close'].iloc[-1]
            ema_value = df[ema_selection].iloc[-1]
            
            # Create EMA status string
            ema_status_str = "Above: "
            ema_status_str += "20✓ " if bear_ema_status['Above_EMA20'] else "20✗ "
            ema_status_str += "50✓ " if bear_ema_status['Above_EMA50'] else "50✗ "
            ema_status_str += "100✓ " if bear_ema_status['Above_EMA100'] else "100✗ "
            ema_status_str += "200✓" if bear_ema_status['Above_EMA200'] else "200✗"
                
            sell_list.append({
                'Symbol': symbol,
                'Name': name,
                'Price': last_price,
                'Selected_EMA': ema_value,
                'EMA_Status': ema_status_str,
                'Distance': bear_distance
            })
    
    progress_bar.empty()
    status_text.empty()
    
    # Show summary of scan results
    if processed_count < total_stocks:
        st.info(f"Note: Data for {total_stocks - processed_count} stocks could not be retrieved or processed.")
    
    # Convert to DataFrames and sort by distance from EMA
    buy_df = pd.DataFrame(buy_list) if buy_list else pd.DataFrame()
    if not buy_df.empty and 'Distance' in buy_df.columns:
        buy_df = buy_df.sort_values('Distance', ascending=False)
        
    sell_df = pd.DataFrame(sell_list) if sell_list else pd.DataFrame()
    if not sell_df.empty and 'Distance' in sell_df.columns:
        sell_df = sell_df.sort_values('Distance', ascending=True)
    
    return buy_df, sell_df

# Main application
def main():
    st.title("Stocks Scanner")
    
    # Display current market status at the top
    st.subheader("Market Status")
    col1, col2, col3 = st.columns(3)
    
    # Initialize session state for managing stock lists
    if 'using_custom_list' not in st.session_state:
        st.session_state.using_custom_list = False
    
    # Sidebar
    st.sidebar.header("Scanner Settings")
    
    # Load default stock lists
    us_stocks, india_stocks = load_stock_lists()
    
    # Custom stock list upload (must be before market selection)
    st.sidebar.subheader("Stock List")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Custom (CSV: Symbol, Name)",
        type="csv",
        help="CSV file with 'symbol' and 'name' columns (Max 50MB, 9999 stocks)"
    )
    
    # Process uploaded file if available
    custom_stocks = None
    if uploaded_file is not None:
        if uploaded_file.size > 50 * 1024 * 1024:  # 50MB limit
            st.sidebar.error("File size exceeds 50MB limit")
            st.session_state.using_custom_list = False
        else:
            # Default to US market for processing custom list if market not already selected
            market_for_processing = st.session_state.get('market', "US")
            custom_stocks = process_uploaded_stock_list(uploaded_file, market_for_processing)
            
            if custom_stocks is not None:
                st.session_state.using_custom_list = True
                st.session_state.custom_stocks = custom_stocks
                st.sidebar.success(f"Loaded {len(custom_stocks)} stocks from your CSV")
            else:
                st.session_state.using_custom_list = False
    else:
        st.session_state.using_custom_list = False
    
    # Market selection - disabled if using custom list
    if st.session_state.using_custom_list:
        market = st.sidebar.selectbox(
            "Select Market (Disabled - Using Custom List)",
            ["US", "India"],
            disabled=True,
            index=0 if st.session_state.get('market') == "US" else 1
        )
        market = st.session_state.get('market', "US")  # Keep existing market selection
    else:
        market = st.sidebar.selectbox("Select Market", ["US", "India"])
        st.session_state.market = market  # Save market selection to session state
    
    # Timeframe selection
    timeframe_options = {
        "1 Hour": "1h",
        "Daily": "1d",
        "Weekly": "1wk",
        "Monthly": "1mo"
    }
    timeframe_display = st.sidebar.selectbox("Select Timeframe", list(timeframe_options.keys()))
    timeframe = timeframe_options[timeframe_display]
    
    # EMA selection
    ema_options = {
        "EMA 20": "EMA20",
        "EMA 50": "EMA50",
        "EMA 100": "EMA100",
        "EMA 200": "EMA200"
    }
    ema_display = st.sidebar.selectbox("Select EMA", list(ema_options.keys()))
    ema_selection = ema_options[ema_display]
    
    # Look Back candles slider 
    lookback = st.sidebar.slider(
        "Look Back Candles", 
        min_value=1, 
        max_value=10, 
        value=5,
        help="Number of candles to look back for crossover event"
    )
    
    # Confirmation candles slider 
    confirmation_candles = st.sidebar.slider(
        "Confirmation Candles", 
        min_value=0, 
        max_value=4, 
        value=2,
        help="Number of consecutive candles required to confirm the new trend (0 means check only current candle)"
    )
    
    # Scan button in settings section
    scan_button = st.sidebar.button("Start Scanning", use_container_width=True)
    
    # Display current market status data
    indices = us_indices if market == "US" else india_indices
    
    index_cols = [col1, col2, col3]
    for i, (index_name, index_symbol) in enumerate(indices.items()):
        try:
            index_data = yf.Ticker(index_symbol).history(period="1d")
            if not index_data.empty:
                current = index_data['Close'].iloc[-1]
                previous = index_data['Open'].iloc[-1]
                change = current - previous
                change_percent = (change / previous) * 100
                
                color = "green" if change >= 0 else "red"
                change_icon = "▲" if change >= 0 else "▼"
                
                index_cols[i].markdown(
                    f"**{index_name}**: {current:.2f} "
                    f"<span style='color:{color}'>{change_icon} {abs(change):.2f} ({abs(change_percent):.2f}%)</span>", 
                    unsafe_allow_html=True
                )
        except:
            index_cols[i].text(f"{index_name}: Data unavailable")
    
    if scan_button:
        # Use custom stock list if uploaded, otherwise use default
        if st.session_state.using_custom_list:
            stocks_to_scan = st.session_state.custom_stocks
        else:
            stocks_to_scan = us_stocks if market == "US" else india_stocks
        
        with st.spinner(f"Scanning {market} stocks on {timeframe_display} timeframe..."):
            buy_df, sell_df = scan_stocks(stocks_to_scan, timeframe, market, 
                                         ema_selection, lookback, confirmation_candles)
        
        # Store results in session state to persist between reruns
        st.session_state.buy_df = buy_df
        st.session_state.sell_df = sell_df
        st.session_state.last_scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.market = market
        st.session_state.timeframe = timeframe_display
        st.session_state.ema_selection = ema_display
        st.session_state.confirmation = confirmation_candles
        st.session_state.lookback = lookback
    
    # Display explanation
    with st.expander("How This Scanner Works"):
        st.markdown(f"""
        ## Stocks Scanner Logic
        
        This scanner identifies stocks with confirmed trend changes:
        
        ### Bullish Stocks 🟢
        - Finds stocks that have crossed above the {ema_display} within the lookback period of {lookback} candles
        - Then stayed above for {confirmation_candles if confirmation_candles > 0 else "just the current candle"}
        - This confirms a potential trend change to bullish
        
        ### Bearish Stocks 🔴
        - Finds stocks that have crossed below the {ema_display} within the lookback period of {lookback} candles
        - Then stayed below for {confirmation_candles if confirmation_candles > 0 else "just the current candle"}
        - This confirms a potential trend change to bearish
        
        ### EMA Status Explained
        - The scanner shows the current position relative to all EMAs (20, 50, 100, 200)
        - ✓ means price is above that EMA
        - ✗ means price is below that EMA
        
        ### Using Custom Stock Lists
        - You can upload your own CSV file with stock symbols
        - The CSV must have columns for 'symbol' and 'name' (or similar naming)
        - For Indian stocks, '.NS' suffix will be automatically added if missing
        - Maximum 9999 stocks per list and 50MB file size
        
        Adjust the lookback and confirmation candles to fine-tune your scan results.
        """)
    
    # Display results in tabs
    tab1, tab2 = st.tabs(["Bullish Stocks 🟢", "Bearish Stocks 🔴"])
    
    # Show last scan info if available
    if 'last_scan_time' in st.session_state:
        scan_info = f"""Last scan: {st.session_state.last_scan_time} | Market: {st.session_state.market} | 
        Timeframe: {st.session_state.timeframe} | {st.session_state.ema_selection} | 
        Lookback: {st.session_state.lookback} candles | Confirmation: {st.session_state.confirmation} candles"""
        st.info(scan_info)
    
    with tab1:
        if 'buy_df' in st.session_state and not st.session_state.buy_df.empty:
            st.subheader(f"Bullish Stocks - {st.session_state.ema_selection} Confirmation")
            
            # Format the dataframe
            df_display = st.session_state.buy_df.copy()
            if 'Price' in df_display.columns:
                df_display['Price'] = df_display['Price'].apply(lambda x: f"{x:.2f}" if x is not None else "N/A")
            if 'Selected_EMA' in df_display.columns:
                df_display['Selected_EMA'] = df_display['Selected_EMA'].apply(lambda x: f"{x:.2f}" if x is not None else "N/A")
            if 'Distance' in df_display.columns:
                df_display['Distance'] = df_display['Distance'].apply(lambda x: f"+{x:.2f}%" if x is not None else "N/A")
            
            # Rename columns for better display
            df_display = df_display.rename(columns={
                'Symbol': 'Symbol',
                'Name': 'Company Name',
                'Price': 'Current Price',
                'Selected_EMA': f'{st.session_state.ema_selection} Value',
                'EMA_Status': 'EMA Status',
                'Distance': 'Above EMA By'
            })
            
            st.dataframe(df_display, use_container_width=True)
            
            # Download button
            csv = df_display.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"bullish_stocks_{st.session_state.market}_{st.session_state.timeframe}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info(f"No bullish stocks found with {ema_display} confirmation. Run a scan to see results.")
    
    with tab2:
        if 'sell_df' in st.session_state and not st.session_state.sell_df.empty:
            st.subheader(f"Bearish Stocks - {st.session_state.ema_selection} Confirmation")
            
            # Format the dataframe
            df_display = st.session_state.sell_df.copy()
            if 'Price' in df_display.columns:
                df_display['Price'] = df_display['Price'].apply(lambda x: f"{x:.2f}" if x is not None else "N/A")
            if 'Selected_EMA' in df_display.columns:
                df_display['Selected_EMA'] = df_display['Selected_EMA'].apply(lambda x: f"{x:.2f}" if x is not None else "N/A")
            if 'Distance' in df_display.columns:
                df_display['Distance'] = df_display['Distance'].apply(lambda x: f"{x:.2f}%" if x is not None else "N/A")
            
            # Rename columns for better display
            df_display = df_display.rename(columns={
                'Symbol': 'Symbol',
                'Name': 'Company Name',
                'Price': 'Current Price',
                'Selected_EMA': f'{st.session_state.ema_selection} Value',
                'EMA_Status': 'EMA Status',
                'Distance': 'Below EMA By'
            })
            
            st.dataframe(df_display, use_container_width=True)
            
            # Download button
            csv = df_display.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"bearish_stocks_{st.session_state.market}_{st.session_state.timeframe}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info(f"No bearish stocks found with {ema_display} confirmation. Run a scan to see results.")

# Run the application
if __name__ == "__main__":
    main()