import streamlit as st
import yfinance as yf
import pandas as pd
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf

# OpenAI API Setup
openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI API Key

# Add Custom CSS
def add_custom_css():
    st.markdown(
        """
        <style>
        body {
            background-color: #f4f4f4;
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            background: linear-gradient(to bottom right, #1f4e79, #f4f4f4);
        }
        .main-title {
            color: #ffffff;
            text-align: center;
            font-size: 2.5em;
            margin-top: 20px;
            margin-bottom: 20px;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }
        .sub-title {
            color: #add8e6;
            font-size: 1.5em;
            font-weight: bold;
            margin-top: 10px;
        }
        .sidebar-title {
            color: #ffffff;
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Load dataset
@st.cache_data
def load_dataset():
    file_path = "market_indicators.csv"  # Path to your Kaggle dataset
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])  # Ensure 'Date' is in datetime format
    return data

# Fetch live stock data
def get_stock_data(ticker, period="1mo", interval="1d"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    return hist

# Add moving averages
def add_moving_averages(data):
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    return data

# Plot candlestick chart
def plot_candlestick(data, ticker):
    data.index = pd.to_datetime(data.index)
    mpf_data = data[['Open', 'High', 'Low', 'Close']]
    st.subheader(f"Candlestick Chart for {ticker}")
    fig, ax = mpf.plot(
        mpf_data,
        type="candle",
        style="charles",
        title=f"{ticker} Candlestick Chart",
        ylabel="Price",
        returnfig=True
    )
    st.pyplot(fig)

# Time-Series Visualization
def plot_time_series(data, columns):
    st.subheader("Time-Series Visualization")
    st.write("Compare trends over time for selected indicators.")
    
    fig, ax = plt.subplots()
    for column in columns:
        ax.plot(data['Date'], data[column], label=column)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title("Time-Series Trends")
    ax.legend()
    st.pyplot(fig)

# Correlation Heatmap
def plot_correlation_heatmap(data):
    st.subheader("Correlation Between Indicators")
    st.write("Analyze the relationship between different indicators.")
    
    correlation_data = data.iloc[:, 1:].corr()  # Exclude 'Date' column
    fig, ax = plt.subplots()
    sns.heatmap(correlation_data, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

# Streamlit UI
def main():
    # Add custom CSS
    add_custom_css()

    # Main Title
    st.markdown('<div class="main-title">📈 Advanced Stock Market Dashboard with OpenAI Chatbot</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown('<div class="sidebar-title">Options</div>', unsafe_allow_html=True)

    # Load dataset
    dataset = load_dataset()

    # Tabs for Dashboard and Chatbot
    tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 Chatbot"])

    # Dashboard Tab
    with tab1:
        st.markdown('<div class="sub-title">Market Indicators</div>', unsafe_allow_html=True)
        st.dataframe(dataset.head())

        st.markdown('<div class="sub-title">Live Stock Prices</div>', unsafe_allow_html=True)
        ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", "AAPL")
        period = st.selectbox("Select Time Period:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "5y"])
        interval = st.selectbox("Select Interval:", ["1m", "5m", "15m", "1h", "1d", "1wk", "1mo"])
        
        if st.button("Fetch Data"):
            try:
                data = get_stock_data(ticker, period=period, interval=interval)
                st.line_chart(data["Close"])
                st.write(data.tail())

                # Add advanced visualizations
                data = add_moving_averages(data)
                plot_candlestick(data, ticker)
            except Exception as e:
                st.error(f"Error fetching data: {e}")

        # Time-Series Visualization
        st.markdown('<div class="sub-title">Explore Time-Series Trends</div>', unsafe_allow_html=True)
        available_columns = dataset.columns[1:]  # Exclude 'Date' column
        selected_columns = st.multiselect(
            "Select indicators to visualize:", available_columns, default=["dxy", "gold"]
        )
        if selected_columns:
            plot_time_series(dataset, selected_columns)
        else:
            st.warning("Please select at least one indicator to visualize.")

        # Correlation Heatmap
        st.markdown('<div class="sub-title">Correlation Analysis</div>', unsafe_allow_html=True)
        if st.button("Show Correlation Heatmap"):
            plot_correlation_heatmap(dataset)

    # Chatbot Tab
    with tab2:
        st.markdown('<div class="sub-title">Ask the OpenAI Chatbot</div>', unsafe_allow_html=True)
        user_input = st.text_input("Ask me anything about the stock market:")
        if st.button("Ask"):
            if user_input:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": user_input}],
                        max_tokens=150
                    )
                    st.write(response['choices'][0]['message']['content'])
                except Exception as e:
                    st.error(f"Error with OpenAI API: {e}")
            else:
                st.error("Please enter a question or statement.")

if __name__ == "__main__":
    main()
