import streamlit as st
import requests
from datetime import datetime

st.set_page_config(page_title="Stock News", layout="wide")

# Finnhub API (hardcoded)
API_KEY = "d3443fhr01qqt8snegsgd3443fhr01qqt8snegt0"
BASE_URL = "https://finnhub.io/api/v1/news?category=general&token=" + API_KEY

def fetch_news():
    response = requests.get(BASE_URL)
    if response.status_code == 200:
        return response.json()
    else:
        return []

def main():
    st.title("📰 Real-time Stock Market News")

    news_data = fetch_news()
    if not news_data:
        st.error("Failed to fetch news. Try again later.")
        return

    for article in news_data[:10]:  # Show top 10
        st.subheader(article["headline"])
        st.write(article["summary"])
        st.caption(f"Source: {article['source']} | {datetime.utcfromtimestamp(article['datetime']).strftime('%Y-%m-%d %H:%M')}")
        st.markdown(f"[View More]({article['url']})", unsafe_allow_html=True)
        st.markdown("---")

if __name__ == "__main__":
    main()
