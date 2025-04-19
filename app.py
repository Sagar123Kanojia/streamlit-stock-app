import streamlit as st
import yfinance as yf
from prophet import Prophet
import plotly.graph_objs as go
from datetime import date
import pandas as pd
import requests  # ✅ for news integration

# Page configuration
st.set_page_config(page_title="Trade Trends", page_icon="📈", layout="wide")
st.title('📊 Trade Trends: Stock Prediction & Analysis')

# Sidebar for user input
st.sidebar.header("User  Input Parameters")

# Text input for manual entry
selected_stock = st.sidebar.text_input(
    'Enter Stock Symbol (e.g., AAPL for Apple or RELIANCE.NS for Reliance India)',
    'RELIANCE.NS'
)

# Dropdown for popular Indian stocks
indian_stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services (TCS)": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Wipro": "WIPRO.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Mphasis": "MPHASIS.NS"
}

selected_from_list = st.sidebar.selectbox("Or select from popular Indian stocks:", options=["None"] + list(indian_stocks.keys()))

if selected_from_list != "None":
    selected_stock = indian_stocks[selected_from_list]

n_years = st.sidebar.slider('Years of prediction:', 1, 4, 1)
period = n_years * 365

# Instructions
st.sidebar.write("""
### Instructions:
1. Enter the stock symbol (e.g., AAPL or RELIANCE.NS).
2. Or select from the dropdown.
3. Select the number of years for prediction.
4. Click on 'Load Data' to view historical data and predictions.
""")

# Load data button
if st.sidebar.button('Load Data'):
    if selected_stock:
        @st.cache_data
        def load_data(ticker):
            try:
                end_date = date.today().strftime("%Y-%m-%d")
                data = yf.download(ticker, start="2018-01-01", end=end_date)
                if data.empty:
                    raise ValueError(f"No data found for stock symbol {ticker}")
                return data
            except Exception as e:
                st.error(f"Error loading data for symbol {ticker}: {e}")
                return pd.DataFrame()

        # Quick test for validity
        test_data = yf.download(selected_stock, period="1d")
        if test_data.empty:
            st.error("Invalid or unsupported stock symbol. Try using '.NS' for Indian stocks.")
            st.stop()

        data = load_data(selected_stock)

        if not data.empty:
            col1, col2 = st.columns(2)
            start_date = col1.date_input("Select start date", min_value=data.index.min(), max_value=data.index.max(), value=data.index.min())
            end_date = col2.date_input("Select end date", min_value=data.index.min(), max_value=data.index.max(), value=data.index.max())

            if start_date > end_date:
                st.error("Start date must be before end date.")
            else:
                filtered_data = data.loc[start_date:end_date]

                if filtered_data.empty:
                    st.warning("No data available for the selected date range.")
                else:
                    st.subheader(f'Historical Data for {selected_stock}')
                    st.write(filtered_data)

                    st.subheader('📈 Real-Time Data')
                    real_time_data = yf.Ticker(selected_stock).history(period="1d")
                    st.write(real_time_data)

                    # ✅ News Section Start
                    st.subheader("📰 Latest News Headlines")

                    def get_news(query):
                        api_key = "a227f972c065477b9dfcd00d80a06759"  # 🔑 Replace this!
                        url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&pageSize=5&apiKey={api_key}"
                        try:
                            response = requests.get(url)
                            articles = response.json().get("articles", [])
                            return articles
                        except Exception as e:
                            st.error(f"Error fetching news: {e}")
                            return []

                    news_articles = get_news(selected_stock.split('.')[0])  # "RELIANCE.NS" → "RELIANCE"

                    if news_articles:
                        for article in news_articles:
                            st.markdown(f"""
                            <div style="background-color:#1e1e1e;padding:1rem;border-radius:10px;border:1px solid #1ABC9C;margin-bottom:1rem;">
                                <strong><a href="{article['url']}" target="_blank" style="color:#1ABC9C;">{article['title']}</a></strong><br>
                                <span style="font-size: 0.8rem; color: gray;">{article['source']['name']} | {article['publishedAt'][:10]}</span>
                                <p style="margin-top: 0.5rem;">{article['description']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.write("No news articles found.")
                    # ✅ News Section End

                    df_train = data[['Close']].reset_index()
                    df_train.columns = ["ds", "y"]
                    df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
                    df_train = df_train.dropna(subset=['y'])

                    if df_train.shape[0] < 2:
                        st.error("Not enough data for Prophet model to fit.")
                    else:
                        m = Prophet()
                        try:
                            m.fit(df_train)
                            future = m.make_future_dataframe(periods=period)
                            forecast = m.predict(future)

                            st.subheader(f'📅 Predicted Data for {selected_stock}')
                            st.write(forecast.tail())

                            st.write(f'📈 Enhanced Forecast Plot for {n_years} Year(s)')
                            fig = go.Figure()

                            fig.add_trace(go.Scatter(
                                x=df_train['ds'], y=df_train['y'],
                                mode='lines', name='Historical',
                                line=dict(color='blue')
                            ))

                            fig.add_trace(go.Scatter(
                                x=forecast['ds'], y=forecast['yhat'],
                                mode='lines', name='Forecast',
                                line=dict(color='green')
                            ))

                            fig.add_trace(go.Scatter(
                                x=forecast['ds'], y=forecast['yhat_upper'],
                                mode='lines', name='Upper Bound',
                                line=dict(width=0),
                                showlegend=False
                            ))

                            fig.add_trace(go.Scatter(
                                x=forecast['ds'], y=forecast['yhat_lower'],
                                mode='lines', name='Lower Bound',
                                line=dict(width=0),
                                fill='tonexty',
                                fillcolor='rgba(0, 255, 0, 0.2)',
                                showlegend=True
                            ))

                            fig.update_layout(
                                title=f"{selected_stock} Forecast for {n_years} Year(s)",
                                xaxis_title='Date',
                                yaxis_title='Stock Price',
                                template='plotly_dark',
                                hovermode='x unified',
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                margin=dict(l=20, r=20, t=60, b=20)
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            st.write("Prediction Components")
                            st.pyplot(m.plot_components(forecast))

                        except Exception as e:
                            st.error(f"Error in fitting the Prophet model: {e}")
        else:
            st.write("⚠ No data available for the selected stock symbol.")
    else:
        st.write("⚠ Please enter a valid stock symbol in the sidebar.")

# Custom styling
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }

    .block-container {
        background-color: black;
        padding: 2rem;
        border-radius: 15px;
    }

    .sidebar .sidebar-content {
        background-color: black;
        color: orange;
    }

    .sidebar .sidebar-content .stButton button {
        background-color: #1ABC9C;
        color: white;
        border: None;
        border-radius: 8px;
    }

    .sidebar .sidebar-content .stButton button:hover {
        background-color: #16A085;
    }

    .css-18e3th9 {
        font-size: 2rem;
        font-family: 'Verdana', sans-serif;
        font-weight: 600;
        color: #1ABC9C;
    }

    .css-1d391kg {
        color: #3498DB;
    }

    .stTextInput > div > input {
        background-color: #F2F3F4;
        border: 1px solid #BDC3C7;
        border-radius: 8px;
    }

    .stSlider > div > div {
        color: #1ABC9C;
    }

    .dataframe {
        border: 2px solid #1ABC9C;
        border-radius: 8px;
    }

    .stPlotlyChart {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 8px;
    }

    .css-15tx938 {
        background-color: #3498DB;
        color: #FFFFFF;
        padding: 1rem;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
