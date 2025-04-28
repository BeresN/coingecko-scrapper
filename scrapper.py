import requests
import time
import pandas as pd
import random

headers = {
    "accept": "application/json"
}

# Get top 50 tokens
def get_top_tokens():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 50,
        "page": 1,
        "sparkline": False
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    return [coin['id'] for coin in data]

# Get Daily OHLCV data properly
def get_token_ohlcv(token_id):
    url = f"https://api.coingecko.com/api/v3/coins/{token_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "365",
        "interval": "daily"
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()

    prices = data.get('prices', [])
    volumes = data.get('total_volumes', [])

    final_data = []
    for i in range(len(prices)):
        timestamp = int(prices[i][0])
        price = prices[i][1]
        volume = volumes[i][1] if i < len(volumes) else None

        final_data.append({
            "token_id": token_id,
            "timestamp": timestamp,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": volume
        })
    
    return final_data

def main():
    top_tokens = get_top_tokens()
    all_rows = []

    for token in top_tokens:
        print(f"Fetching data for: {token}")
        try:
            data = get_token_ohlcv(token)
            all_rows.extend(data)
            time.sleep(random.uniform(3,6))
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print("Rate limit hit. Waiting 60 seconds...")
                time.sleep(60)
            else:
                print(f"HTTP error for {token}: {e}")
        except Exception as e:
            print(f"Other error for {token}: {e}")
    

    df = pd.DataFrame(all_rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.to_csv("all_tokens_ohlcv_daily.csv", index=False)

if __name__ == "__main__":
    main()
