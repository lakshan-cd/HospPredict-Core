trends_df.to_csv('/content/drive/MyDrive/FYP/global_google_trends_2014_2019.csv', index=False)
import pandas as pd
import numpy as np

# Load your trends CSV
df = pd.read_csv("global_google_trends_2014_2019.csv", index_col=0, parse_dates=True)

# ---- Hotel alias and symbol map ---- #
hotel_aliases = {
    "AITKEN SPENCE": ["aitken", "spence"],
    "ASIAN HOTELS": ["asian hotels", "cinnamon red", "cinnamon grand"],
    "BROWNS BEACH": ["browns beach"],
    "CEYLON HOTELS": ["ceylon hotels", "queens hotel", "galle face"],
    "DOLPHIN HOTELS": ["dolphin"],
    "EDEN HOTEL": ["eden"],
    "GALADARI HOTELS": ["galadari"],
    "HOTEL SIGIRIYA": ["hotel sigiriya", "sigiriya plc"],
    "JOHN KEELLS": ["john keells", "jetwing", "cinnamon", "hikka tranz"],
    "MAHAWELI REACH": ["mahaweli reach"],
    "PALM GARDEN": ["palm garden"],
    "PEGASUS HOTELS": ["pegasus"],
    "RENUKA CITY": ["renuka city"],
    "RENUKA HOTELS": ["renuka hotels"],
    "ROYAL PALMS": ["royal palms"],
    "SERENDIB HOTELS": ["serendib"],
    "SIGIRIYA VILLAGE": ["sigiriya village"],
    "TAL LANKA": ["tal lanka"],
    "TANGERINE BEACH": ["tangerine beach"],
    "LIGHTHOUSE HOTEL": ["lighthouse"],
    "KANDY HOTELS": ["kandy hotels", "queens hotel"],
    "TRANS ASIA": ["trans asia"],
}

hotel_symbol_map = {
    "AITKEN SPENCE": "AHUN.N0000",
    "ASIAN HOTELS": "AHPL.N0000",
    "BROWNS BEACH": "BBH.N0000",
    "CEYLON HOTELS": "CHOT.N0000",
    "DOLPHIN HOTELS": "STAF.N0000",
    "EDEN HOTEL": "EDEN.N0000",
    "GALADARI HOTELS": "GHLL.N0000",
    "HOTEL SIGIRIYA": "HSIG.N0000",
    "JOHN KEELLS": "KHL.N0000",
    "MAHAWELI REACH": "MRH.N0000",
    "PALM GARDEN": "PALM.N0000",
    "PEGASUS HOTELS": "PEG.N0000",
    "RENUKA CITY": "RENU.N0000",
    "RENUKA HOTELS": "RCH.N0000",
    "ROYAL PALMS": "RPBH.N0000",
    "SERENDIB HOTELS": "SHOT.N0000",
    "SIGIRIYA VILLAGE": "SIGV.N0000",
    "TAL LANKA": "TAJ.N0000",
    "TANGERINE BEACH": "TANG.N0000",
    "LIGHTHOUSE HOTEL": "LHL.N0000",
    "KANDY HOTELS": "KHC.N0000",
    "TRANS ASIA": "TRAN.N0000"
}

# ---- Detect spikes ---- #
threshold_std = 2  # 2 standard deviations
spike_labels = []
spike_types = []

for date, row in df.iterrows():
    active_hotels = []
    active_symbols = []

    for col in df.columns:
        series = df[col]
        mean = series.mean()
        std = series.std()
        if row[col] >= mean + threshold_std * std:
            # Match alias to hotel name
            for hotel_group, aliases in hotel_aliases.items():
                if any(alias.lower() in col.lower() for alias in aliases):
                    active_hotels.append(hotel_group)
                    if hotel_group in hotel_symbol_map:
                        active_symbols.append(hotel_symbol_map[hotel_group])
                    break

    # Remove duplicates and join
    if active_hotels:
        spike_labels.append("; ".join(sorted(set(active_hotels))))
        spike_types.append("; ".join(sorted(set(active_symbols))))
    else:
        spike_labels.append(None)
        spike_types.append(None)

# Add to DataFrame
df['spike_label'] = spike_labels
df['spike_type'] = spike_types

# Save final file
df.to_csv("google_trends_with_spikes.csv")
print("✅ Done! File saved as google_trends_with_spikes.csv")
import pandas as pd
import numpy as np

# Read CSV again just to reset
df = pd.read_csv("global_google_trends_2014_2019.csv", index_col=0, parse_dates=True)

# Tourism keywords
tourism_keywords = [
    "Sri Lanka", "Sri Lanka tourism", "visit Sri Lanka",
    "Kandy tourism", "sri lanka beaches", "Galle", "Colombo travel"
]

# Use only numeric keyword columns
keyword_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Compute mean and std safely
means = df[keyword_cols].mean()
stds = df[keyword_cols].std()

# Threshold multiplier
std_threshold = 2.0

spike_labels = []
spike_types = []

for idx, row in df.iterrows():
    tourism_spike = False
    active_hotels = []
    active_symbols = []

    for col in keyword_cols:
        value = row[col]
        if pd.isna(value):
            continue

        mean = means[col]
        std = stds[col]

        # Skip columns with zero or NaN std
        if std == 0 or pd.isna(std):
            continue

        is_spike = value >= mean + std_threshold * std

        if is_spike:
            if col in tourism_keywords:
                tourism_spike = True
            else:
                for hotel_group, aliases in hotel_aliases.items():
                    if any(alias.lower() in col.lower() for alias in aliases):
                        active_hotels.append(hotel_group)
                        if hotel_group in hotel_symbol_map:
                            active_symbols.append(hotel_symbol_map[hotel_group])
                        break

    # Assign labels
    if tourism_spike and active_hotels:
        spike_labels.append("Tourism; " + "; ".join(sorted(set(active_hotels))))
        spike_types.append("tourism; " + "; ".join(sorted(set(active_symbols))))
    elif tourism_spike:
        spike_labels.append("Tourism")
        spike_types.append("tourism")
    elif active_hotels:
        spike_labels.append("; ".join(sorted(set(active_hotels))))
        spike_types.append("; ".join(sorted(set(active_symbols))))
    else:
        spike_labels.append(None)
        spike_types.append(None)

# Add to DataFrame
df["spike_label"] = spike_labels
df["spike_type"] = spike_types

# Save
df.to_csv("google_trends_with_spikes_fixed.csv")

# Optional: count how many weeks had each type
print(df['spike_type'].value_counts(dropna=False))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from google.colab import drive
!pip install transformers
drive.mount('/content/drive')
from google.colab import files
import pandas as pd

# Load the CSV
df1 = pd.read_csv('/content/drive/MyDrive/FYP/NEW/preprocessed_news_articles_SL.csv')

# Convert published_date to datetime
df1['date'] = pd.to_datetime(df1['published_date'], errors='coerce')

df1 = df1[(df1['date'].dt.year >= 2014) & (df1['date'].dt.year <= 2019)]

# Remove original heading/content columns if they exist
df1 = df1.drop(columns=[col for col in ['heading', 'content'] if col in df1.columns])

# Remove rows where both cleaned_heading and cleaned_content are empty or NaN
df1 = df1[~((df1['cleaned_heading'].isna() | df1['cleaned_heading'].str.strip().eq('')) &
            (df1['cleaned_content'].isna() | df1['cleaned_content'].str.strip().eq('')))]

# Keep only the required columns in desired order
df1 = df1[['date', 'cleaned_heading', 'cleaned_content','source']]
df1 = df1.rename(columns={'cleaned_heading': 'heading', 'cleaned_content': 'content'})

# Save to CSV
df1.to_csv('/content/drive/MyDrive/FYP/preprocessed_news_articles_SL.csv', index=False)
# Hotel alias mapping — add more keywords as needed
hotel_aliases = {
    "AITKEN SPENCE": ["aitken", "spence"],
    "ASIAN HOTELS": ["asian hotels", "cinnamon red", "cinnamon grand", "cinnamon lakeside"],
    "BROWNS BEACH": ["browns beach"],
    "CEYLON HOTELS": ["ceylon hotels", "queens hotel", "galle face"],
    "DOLPHIN HOTELS": ["dolphin"],
    "EDEN HOTEL": ["eden"],
    "GALADARI HOTELS": ["galadari"],
    "HOTEL SIGIRIYA": ["hotel sigiriya", "sigiriya plc"],
    "JOHN KEELLS": ["john keells", "cinnamon", "hikka tranz"],
    "MAHAWELI REACH": ["mahaweli reach"],
    "PALM GARDEN": ["palm garden"],
    "PEGASUS HOTELS": ["pegasus"],
    "RENUKA CITY": ["renuka city"],
    "RENUKA HOTELS": ["renuka hotels"],
    "ROYAL PALMS": ["royal palms"],
    "SERENDIB HOTELS": ["serendib"],
    "SIGIRIYA VILLAGE": ["sigiriya village"],
    "TAL LANKA": ["tal lanka"],
    "TANGERINE BEACH": ["tangerine beach"],
    "LIGHTHOUSE HOTEL": ["lighthouse"],
    "KANDY HOTELS": ["kandy hotels", "queens hotel"],
    "TRANS ASIA": ["trans asia"],
}

def classify_news_type_and_hotel(row):
    heading = str(row['heading']).lower()
    content = str(row['content']).lower()

    for hotel, keywords in hotel_aliases.items():
        for keyword in keywords:
            if keyword in heading or keyword in content:
                return pd.Series(["hotel", hotel])
    return pd.Series(["general", "Unknown"])

# Apply classification
df1[['type', 'hotel']] = df1.apply(classify_news_type_and_hotel, axis=1)
# Tourism keywords
tourism_keywords = [
    "tourism", "travel", "holiday", "resort", "hotel industry", "beach", "sri lanka tourism",
    "tourist", "vacation", "hospitality", "wildlife park", "ayurveda", "unawatuna", "ella",
    "sigiriya", "yala", "kandy", "galle fort", "ecotourism", "foreign arrivals"
]

def classify_news_type_and_hotel(row):
    heading = str(row['heading']).lower()
    content = str(row['content']).lower()

    # HOTEL DETECTION
    for hotel, keywords in hotel_aliases.items():
        for keyword in keywords:
            if keyword in heading or keyword in content:
                return pd.Series(["hotel", hotel])

    # TOURISM DETECTION
    for keyword in tourism_keywords:
        if keyword in heading or keyword in content:
            return pd.Series(["tourism", "Unknown"])

    # GENERAL
    return pd.Series(["general", "Unknown"])

# Apply to dataframe
df1[['type', 'hotel']] =df1.apply(classify_news_type_and_hotel, axis=1)
df1.head()
!pip install vaderSentiment
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load your dataset
df = df1

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment from text
def get_sentiment(text):
    if pd.isna(text) or text.strip() == '':
        return 'neutral'
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis: Use 'content' if available, otherwise use 'heading'
df['sentiment'] = df.apply(
    lambda row: get_sentiment(row['content']) if 'content' in df.columns and not pd.isna(row['content']) and row['content'].strip() != ''
    else get_sentiment(row['heading']),
    axis=1
)

# Save with sentiment
df.to_csv('/content/drive/MyDrive/FYP/news_with_sentiment.csv', index=False)
df.to_csv('/content/drive/MyDrive/FYP/NEW/news_with_sentiment.csv', index=False)

import pandas as pd
from scipy.stats import zscore
from google.colab import drive
drive.mount('/content/drive')

# Load your dataset
file_path = '/content/SriLanka.csv'  # Update path if needed
df = pd.read_csv(file_path)

# Set up spike label columns
df['spike_label'] = ''
df['spike_type'] = ''

# Company mappings (hotel name as in column -> list of CSE symbols)
hotel_to_company = {
    "Mahaweli Reach Hotel": ["MRH.N0000"],
    "Hotel Sigiriya": ["HSIG.N0000"],
    "Serendib Hotels": ["SHOT.N0000", "SHOT.X0000"],
    "Renuka City Hotel": ["RENU.N0000"],
    "Trans Asia Hotels": ["TRAN.N0000"],
    "Tangerine Beach Hotel": ["TANG.N0000"],
    "John Keells Holdings": ["KHL.N0000"],
    "The Galadari Hotel": ["GHLL.N0000"],
    "Cinnamon Hotels": ["AHPL.N0000", "TRAN.N0000"],
    "Brown Beach hotel": ["BBH.N0000"],
    "Aitken Spence": ["AHUN.N0000"],
    "Habarana Village by Cinnamon": ["KHL.N0000"],
    "Cinnamon Bentota Beach": ["AHPL.N0000"],
    "Cinnamon Bey Beruwala": ["AHPL.N0000"],
    "Cinnamon Red Colombo": ["AHPL.N0000"],
    "Cinnamon Wild Yala": ["AHPL.N0000"],
    "Cinnamon Grand Colombo": ["AHPL.N0000"],
    "Cinnamon Lakeside Colombo": ["TRAN.N0000"],
    "Taj Samudra, Colombo": ["TAJ.N0000"],
    "Club Hotel Dolphin": ["STAF.N0000"],
    "Heritance Kandalama": ["AHUN.N0000"],
    "Heritance Ahungalla": ["AHUN.N0000"]
}

# Tourism-related columns
tourism_columns = ["Sri Lanka", "Sri Lanka tourism", "visit Sri Lanka", "Kandy tourism", "sri lanka beaches", "Galle", "Colombo travel"]
print(df.columns.tolist())

import pandas as pd

# Load dataset
df = pd.read_csv('/content/drive/MyDrive/FYP/search_volume_with_spikes.csv')

# Convert Week column to datetime
df['Week'] = pd.to_datetime(df['Week'])

# Define hotel → company symbol mapping
hotel_symbol_map = {
    'Mahaweli Reach Hotel': 'MRH.N0000',
    'Hotel Sigiriya': 'HSIG.N0000',
    'Serendib Hotels': 'SHOT.N0000',
    'Renuka City Hotel': 'RENU.N0000',
    'Trans Asia Hotels': 'TRAN.N0000',
    'Tangerine Beach Hotel': 'TANG.N0000',
    'John Keells Holdings': 'KHL.N0000',
    'The Galadari Hotel': 'GHLL.N0000',
    'Cinnamon Hotels': 'AHPL.N0000',
    'Brown Beach hotel': 'BBH.N0000',
    'Aitken Spence': 'AHUN.N0000',
    'Habarana Village by Cinnamon': 'KHL.N0000',
    'Cinnamon Bentota Beach': 'KHL.N0000',
    'Cinnamon Bey Beruwala': 'KHL.N0000',
    'Cinnamon Red Colombo': 'AHPL.N0000',
    'Cinnamon Wild Yala': 'KHL.N0000',
    'Cinnamon Grand Colombo': 'AHPL.N0000',
    'Cinnamon Lakeside Colombo': 'AHPL.N0000',
    'Taj Samudra, Colombo': 'TAJ.N0000',
    'Club Hotel Dolphin': 'STAF.N0000',
    'Heritance Kandalama': 'AHUN.N0000',
    'Heritance Ahungalla': 'AHUN.N0000'
}

# General tourism terms
tourism_terms = ['Sri Lanka', 'Sri Lanka tourism', 'visit Sri Lanka', 'Kandy tourism', 'sri lanka beaches', 'Galle', 'Colombo travel']

# Define parameters
threshold_multiplier = 1.4
rolling_window = 4

# Initialize spike_type and label columns
df['spike_type'] = 'None'
df['spike_label'] = 'None'

# Loop through each column (except Week)
for column in df.columns[1:-2]:  # exclude spike_type & label
    # Calculate rolling mean (excluding current week)
    df[f'{column}_rolling'] = df[column].shift(1).rolling(window=rolling_window).mean()

    # Identify spikes
    spikes = df[column] > df[f'{column}_rolling'] * threshold_multiplier

    for idx in df[spikes].index:
        # Label addition
        if df.loc[idx, 'spike_label'] == 'None':
            df.at[idx, 'spike_label'] = column
        else:
            df.at[idx, 'spike_label'] += f"; {column}"

        # Type addition
        if column in tourism_terms:
            if df.loc[idx, 'spike_type'] == 'None':
                df.at[idx, 'spike_type'] = 'tourism'
            elif 'tourism' not in df.loc[idx, 'spike_type']:
                df.at[idx, 'spike_type'] += "; tourism"
        elif column in hotel_symbol_map:
            symbol = hotel_symbol_map[column]
            if df.loc[idx, 'spike_type'] == 'None':
                df.at[idx, 'spike_type'] = symbol
            elif symbol not in df.loc[idx, 'spike_type']:
                df.at[idx, 'spike_type'] += f"; {symbol}"

# Drop intermediate rolling columns
df = df.drop(columns=[col for col in df.columns if '_rolling' in col])

# Fill empty values with "None" just in case
df[['spike_label', 'spike_type']] = df[['spike_label', 'spike_type']].fillna('None')

# Save to file (optional)
df.to_csv('/content/drive/MyDrive/FYP/search_volume_spikes_final.csv', index=False)

# Show rows with spikes
df.head(100)
import pandas as pd

# Define hotel group alias mapping
hotel_aliases = {
    "AITKEN SPENCE": ["aitken", "spence"],
    "ASIAN HOTELS": ["asian hotels", "cinnamon red", "cinnamon grand"],
    "BROWNS BEACH": ["browns beach"],
    "CEYLON HOTELS": ["ceylon hotels", "queens hotel", "galle face"],
    "DOLPHIN HOTELS": ["dolphin"],
    "EDEN HOTEL": ["eden"],
    "GALADARI HOTELS": ["galadari"],
    "HOTEL SIGIRIYA": ["hotel sigiriya", "sigiriya plc"],
    "JOHN KEELLS": ["john keells", "jetwing", "cinnamon", "hikka tranz"],
    "MAHAWELI REACH": ["mahaweli reach"],
    "PALM GARDEN": ["palm garden"],
    "PEGASUS HOTELS": ["pegasus"],
    "RENUKA CITY": ["renuka city"],
    "RENUKA HOTELS": ["renuka hotels"],
    "ROYAL PALMS": ["royal palms"],
    "SERENDIB HOTELS": ["serendib"],
    "SIGIRIYA VILLAGE": ["sigiriya village"],
    "TAL LANKA": ["tal lanka"],
    "TANGERINE BEACH": ["tangerine beach"],
    "LIGHTHOUSE HOTEL": ["lighthouse"],
    "KANDY HOTELS": ["kandy hotels", "queens hotel"],
    "TRANS ASIA": ["trans asia"],
}

# Define symbol mapping
hotel_symbol_map = {
    "AITKEN SPENCE": "AHUN.N0000",
    "ASIAN HOTELS": "AHPL.N0000",
    "BROWNS BEACH": "BBH.N0000",
    "CEYLON HOTELS": "CHOT.N0000",
    "DOLPHIN HOTELS": "STAF.N0000",
    "EDEN HOTEL": "EDEN.N0000",
    "GALADARI HOTELS": "GHLL.N0000",
    "HOTEL SIGIRIYA": "HSIG.N0000",
    "JOHN KEELLS": "KJL.N0000",
    "MAHAWELI REACH": "MRH.N0000",
    "PALM GARDEN": "PALM.N0000",
    "PEGASUS HOTELS": "PEG.N0000",
    "RENUKA CITY": "RENU.N0000",
    "RENUKA HOTELS": "RCH.N0000",
    "ROYAL PALMS": "RPBH.N0000",
    "SERENDIB HOTELS": "SHOT.N0000",
    "SIGIRIYA VILLAGE": "SIGV.N0000",
    "TAL LANKA": "TAJ.N0000",
    "TANGERINE BEACH": "TANG.N0000",
    "LIGHTHOUSE HOTEL": "LHL.N0000",
    "KANDY HOTELS": "KHC.N0000",
    "TRANS ASIA": "TRAN.N0000"
}

# Load your dataset
merged_news = pd.read_csv('/content/drive/MyDrive/FYP/NEW/final_merged_news.csv', parse_dates=['date'])

# Function to find group
def get_hotel_group(hotel_name):
    hotel_name = str(hotel_name).lower()
    for group, keywords in hotel_aliases.items():
        for kw in keywords:
            if kw in hotel_name:
                return group
    return "None"

# Apply function
merged_news['hotel_group'] = merged_news['hotel'].apply(get_hotel_group)
merged_news['hotel_symbol'] = merged_news['hotel_group'].map(hotel_symbol_map).fillna("None")

# Reorder columns
cols = merged_news.columns.tolist()
hotel_idx = cols.index('hotel')
group_idx = cols.index('hotel_group')
symbol_idx = cols.index('hotel_symbol')

# Reorder: hotel, hotel_group, hotel_symbol, ...
ordered_cols = (
    cols[:hotel_idx+1] +
    [cols[group_idx]] +
    [cols[symbol_idx]] +
    [c for c in cols if c not in [cols[group_idx], cols[symbol_idx]]]
)
merged_news = merged_news[ordered_cols]

# Save updated file
merged_news.head(50)
import os
import pandas as pd
import glob
import numpy as np

path = '/content/drive/MyDrive/FYP/Stock_Prices/'
all_files = glob.glob(os.path.join(path, "*.csv"))

dfs = []

for file in all_files:
    try:
        df = pd.read_csv(file, parse_dates=['Trade Date'])

        # Extract filename without path and extension
        filename = os.path.basename(file).replace('.csv', '')

        # Split into parts and get symbol and name
        parts = filename.split('_')
        hotel_symbol = parts[-1].strip()
        hotel_name = '_'.join(parts[:-1]).strip()

        # Add to dataframe
        df['hotel_symbol'] = hotel_symbol
        df['hotel_name'] = hotel_name

        dfs.append(df)
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")

if not dfs:
    raise ValueError("No valid CSV files found or processed")

combined_df = pd.concat(dfs, ignore_index=True)
import pandas as pd

# Remove currency symbols and commas, and convert to float
def clean_numeric_column(col):
    return pd.to_numeric(col.replace({',': '', 'Rs.': '', '(': '-', ')': ''}, regex=True), errors='coerce')

# Convert Trade Date to datetime
combined_df['Trade Date'] = pd.to_datetime(combined_df['Trade Date'], errors='coerce')

# Clean and convert numeric columns
numeric_columns = ['Open (Rs.)', 'High (Rs.)', 'Low (Rs.)', 'Close (Rs.)',
                   'TradeVolume', 'ShareVolume', 'Turnover (Rs.)', 'price_change_pct',
                   'moving_avg_7d', 'volatility']

for col in numeric_columns:
    combined_df[col] = clean_numeric_column(combined_df[col])

# Ensure symbol and name columns are strings
combined_df['hotel_symbol'] = combined_df['hotel_symbol'].astype(str).str.strip()
combined_df['hotel_name'] = combined_df['hotel_name'].astype(str).str.strip()

# Optional: convert volume columns to int if they don't have decimals
for col in ['TradeVolume', 'ShareVolume']:
    if pd.api.types.is_float_dtype(combined_df[col]) and combined_df[col].dropna().mod(1).eq(0).all():
        combined_df[col] = combined_df[col].astype('Int64')  # nullable int

# Final check
print("📊 Final column types:")
print(combined_df.dtypes)
# Calculate daily price changes and moving averages
combined_df = combined_df.sort_values(['hotel_symbol', 'Trade Date'])

# Daily % change in closing price
combined_df['price_change_pct'] = combined_df.groupby('hotel_symbol')['Close (Rs.)'].pct_change() * 100

# 7-day moving average (short-term trend)
combined_df['moving_avg_7d'] = combined_df.groupby('hotel_symbol')['Close (Rs.)'].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)

# Volatility (rolling 7-day std dev)
combined_df['volatility'] = combined_df.groupby('hotel_symbol')['Close (Rs.)'].transform(
    lambda x: x.rolling(7, min_periods=1).std()
)

# Relative Strength Index (RSI) - momentum indicator
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

combined_df['RSI'] = combined_df.groupby('hotel_symbol')['Close (Rs.)'].transform(calculate_rsi)



def detect_spikes(df, window=10, threshold=2):
    df = df.sort_values('Trade Date').copy()
    df['rolling_mean'] = df['Close (Rs.)'].rolling(window=window, min_periods=1).mean()
    df['rolling_std'] = df['Close (Rs.)'].rolling(window=window, min_periods=1).std()

    # Avoid division by zero
    df['z_score'] = np.where(df['rolling_std'] > 0,
                           (df['Close (Rs.)'] - df['rolling_mean']) / df['rolling_std'],
                           0)

    df['event_type'] = np.where(df['z_score'] > threshold, 'stock_spike_up',
                               np.where(df['z_score'] < -threshold, 'stock_spike_down', None))

    return df[df['event_type'].notna()][['Trade Date', 'hotel_symbol', 'hotel_name', 'event_type']]\
           .rename(columns={'Trade Date': 'event_date'})

# Process each hotel separately
spike_dfs = []
for (hotel_symbol, hotel_name), group in combined_df.groupby(['hotel_symbol', 'hotel_name']):
    print(f"\nProcessing {hotel_name} ({hotel_symbol}) with {len(group)} records...")
    spikes = detect_spikes(group)
    if not spikes.empty:
        spike_dfs.append(spikes)
    else:
        print("No spikes detected")

if spike_dfs:
    stock_spike_df = pd.concat(spike_dfs, ignore_index=True)
    print("\nFinal spike detection results:")
    print(stock_spike_df['hotel_name'].value_counts())
else:
    stock_spike_df = pd.DataFrame(columns=['event_date', 'hotel_symbol', 'hotel_name', 'event_type'])
    print("No spike events detected in any hotel")
stock_spike_df.to_csv('/content/drive/MyDrive/FYP/Stock_spikes.CSV', index=False)
file_path1 = '/content/drive/MyDrive/FYP/Stock_spikes.CSV'
file_path2 = '/content/drive/MyDrive/FYP/NEW/Search_2014_2024_combined.csv'
file_path3 = '/content/drive/MyDrive/FYP/NEW/merged_news_dataset_Symbols.csv'
search_spike_df = pd.read_csv(file_path2)
stock_spike_df= pd.read_csv(file_path1)
news_df=pd.read_csv(file_path3)
cols = search_spike_df.columns.tolist()

# Remove 'Week' from the list and insert it at position 0
cols.insert(0, cols.pop(cols.index('Week')))

# Reorder DataFrame columns
search_spike_df = search_spike_df[cols]
search_spike_df.to_csv('/content/drive/MyDrive/FYP/NEW/Search_2014_2024_combined.csv', index=False)
news_df['date'] = pd.to_datetime(news_df['date'])
stock_spike_df['event_date'] = pd.to_datetime(stock_spike_df['event_date'])
search_spike_df['week_start'] = pd.to_datetime(search_spike_df['Week'])
def is_news_linked_to_event(news_row):
    date = news_row['date']
    sentiment = news_row['sentiment']
    n_type = news_row['type']
    symbol = news_row['hotel_symbol']

    # ---- STOCK SPIKES ----
    if symbol and sentiment == 'positive':
        # Check for stock spike UP
        spikes = stock_spike_df[
            (stock_spike_df['hotel_symbol'] == symbol) &
            (stock_spike_df['event_type'].str.contains("spike_up"))
        ]
        if any((date >= s - timedelta(days=2)) & (date <= s + timedelta(days=2)) for s in spikes['event_date']):
            return True

    if symbol and sentiment == 'negative':
        # Check for stock spike DOWN
        spikes = stock_spike_df[
            (stock_spike_df['hotel_symbol'] == symbol) &
            (stock_spike_df['event_type'].str.contains("spike_down"))
        ]
        if any((date >= s - timedelta(days=2)) & (date <= s + timedelta(days=2)) for s in spikes['event_date']):
            return True

    # ---- SEARCH SPIKES ----
    news_week = date - timedelta(days=date.weekday())  # get Monday
    spikes = search_spike_df[search_spike_df['week_start'] == news_week]['spike_type'].tolist()

    # For hotel-specific search spikes (e.g., MRH.N0000)
    if symbol in spikes and sentiment in ['positive', 'negative']:
        return True

    # For tourism search spikes (general or tourism news types)
    if 'tourism' in spikes and n_type in ['tourism', 'general'] and sentiment in ['positive', 'negative']:
        return True

    return False


from datetime import timedelta

def is_stock_spike_related(news_row):
    symbol = news_row['hotel_symbol']
    if pd.isna(symbol):
        return False
    date = news_row['date']
    spikes = stock_spike_df[stock_spike_df['hotel_symbol'] == symbol]
    return any((date >= spike - timedelta(days=2)) & (date <= spike + timedelta(days=2)) for spike in spikes['event_date'])

news_df['linked_to_stock_spike'] = news_df.apply(is_stock_spike_related, axis=1)
def is_search_spike_related(news_row):
    news_week = news_row['date'] - timedelta(days=news_row['date'].weekday())  # start of the week
    spike_keywords = search_spike_df[search_spike_df['week_start'] == news_week]['spike_type'].tolist()

    if news_row['type'] == 'hotel' and news_row['hotel_symbol'] in spike_keywords:
        return True
    elif news_row['type'] in ['tourism', 'general'] and 'tourism' in spike_keywords:
        return True
    return False

news_df['linked_to_search_spike'] = news_df.apply(is_search_spike_related, axis=1)
def calculate_weight(row):
    base = 0.2

    if row['linked_to_stock_spike'] and row['linked_to_search_spike']:
        base += 0.7
    elif row['linked_to_stock_spike']:
        base += 0.4
    elif row['linked_to_search_spike']:
        base += 0.3

    if row['type'] == 'hotel':
        base += 0.2
    elif row['type'] == 'tourism':
        base += 0.1

    if row['sentiment'] in ['positive', 'negative']:
        base += 0.05

    return round(min(base, 1.0), 2)

news_df['weight'] = news_df.apply(calculate_weight, axis=1)
news_df.to_csv('/content/drive/MyDrive/FYP/Structured_df_with_hotelChange.csv', index=False)

# ✅ Step 0: Install required libraries if needed
# !pip install pandas numpy matplotlib

import pandas as pd
import os
from datetime import timedelta

# ✅ Step 1: Load all hotel stock files into a dictionary
# Place all your hotel CSVs in a folder named "stock_data"
# Each file should be named like "KHL.N0000.csv", "AHUN.N0000.csv", etc.

folder_path = "/content/drive/MyDrive/FYP/Stock_Prices"  # adjust to your Drive location
stock_df_dict = {}

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        symbol = filename.split("_")[-1].replace(".csv", "")
        df = pd.read_csv(os.path.join(folder_path, filename))
        df['Trade Date'] = pd.to_datetime(df['Trade Date'])
        stock_df_dict[symbol] = df

print(f"✅ Loaded {len(stock_df_dict)} hotel stock files")
from datetime import timedelta

def get_price_context(symbol, date, stock_df_dict):
    if pd.isna(symbol) or symbol not in stock_df_dict:
        return [None] * 6

    df = stock_df_dict[symbol]
    df['Trade Date'] = pd.to_datetime(df['Trade Date'])
    df = df.sort_values('Trade Date')

    try:
        close_T0 = df[df['Trade Date'] == date]['Close (Rs.)'].values[0]
        close_Tn1 = df[df['Trade Date'] == (date - timedelta(days=1))]['Close (Rs.)'].values[0]
        close_Tp1 = df[df['Trade Date'] == (date + timedelta(days=1))]['Close (Rs.)'].values[0]
        close_Tp2 = df[df['Trade Date'] == (date + timedelta(days=2))]['Close (Rs.)'].values[0]

        high_max = df[
            df['Trade Date'].between(date, date + timedelta(days=2))
        ]['High (Rs.)'].max()

        # Calculate %
        price_change_0d = round((close_T0 - close_Tn1) / close_Tn1 * 100, 2)
        price_change_1d = round((close_Tp1 - close_T0) / close_T0 * 100, 2)
        price_change_3d = round((close_Tp2 - close_Tn1) / close_Tn1 * 100, 2)
        max_surge = round((high_max - close_T0) / close_T0 * 100, 2)

        return close_T0, close_Tn1, price_change_0d, price_change_1d, price_change_3d, max_surge

    except IndexError:
        return [None] * 6

from datetime import timedelta

def add_all_hotels_change_only(news_df, stock_df_dict):
    hotel_symbols = list(stock_df_dict.keys())
    news_df['Week'] = pd.to_datetime(news_df['Week'])

    for symbol in hotel_symbols:
        print(f"Processing {symbol}...")
        df_stock = stock_df_dict[symbol]
        df_stock['Trade Date'] = pd.to_datetime(df_stock['Trade Date'])
        df_stock = df_stock.sort_values('Trade Date').reset_index(drop=True)

        # Lookup dictionary for fast access
        date_to_close = dict(zip(df_stock['Trade Date'], df_stock['Close (Rs.)']))

        price_changes = []
        for date in news_df['Week']:
            T0 = date_to_close.get(date, None)
            Tp1 = date_to_close.get(date + timedelta(days=1), None)

            if T0 is not None and Tp1 is not None:
                pct_change = round((Tp1 - T0) / T0 * 100, 2)
            else:
                pct_change = None

            price_changes.append(pct_change)

        news_df[f'{symbol}_change_1d'] = price_changes

    return news_df

from datetime import timedelta

# Step 1: List of hotel symbols
hotel_symbols = list(stock_df_dict.keys())

# Step 2: Add empty columns for price changes
for symbol in hotel_symbols:
    news_df[f'{symbol}_change_1d'] = None  # 1-day change
    # Optional: Add more timeframes (e.g., 3d, 5d)
    # news_df[f'{symbol}_change_3d'] = None

# Step 3: Loop over each news article
for idx, row in news_df.iterrows():
    news_date = pd.to_datetime(row['date'])

    for symbol in hotel_symbols:
        df = stock_df_dict[symbol].copy()
        df['Trade Date'] = pd.to_datetime(df['Trade Date'])
        df = df.sort_values('Trade Date')

        # Get T0 price (news date)
        T0_row = df[df['Trade Date'] == news_date]

        if T0_row.empty:
            news_df.at[idx, f'{symbol}_change_1d'] = None
            continue  # Skip if no data on news date

        T0_price = T0_row['Close (Rs.)'].values[0]

        # Get T+1 price (next trading day, not necessarily +1 calendar day)
        next_day_row = df[df['Trade Date'] > news_date].head(1)

        if next_day_row.empty:
            news_df.at[idx, f'{symbol}_change_1d'] = None
        else:
            T1_price = next_day_row['Close (Rs.)'].values[0]
            pct_change = round((T1_price - T0_price) / T0_price * 100, 2)
            news_df.at[idx, f'{symbol}_change_1d'] = pct_change

# Save results
news_df.to_csv('news_with_stock_changes.csv', index=False)
cols = ['stock_price', 'prev_close_price', 'price_change_0d',
        'price_change_1d', 'price_change_3d', 'max_surge_after_news']

for col in cols:
    news_df[col] = None

for idx, row in news_df.iterrows():
    symbol, date = row['hotel_symbol'], row['date']
    values = get_price_context(symbol, date, stock_df_dict)
    for i, col in enumerate(cols):
        news_df.at[idx, col] = values[i]

cols_to_drop = [
    'price_change', 'price_change_label', 'stock_price', 'prev_close_price',
    'price_change_0d', 'price_change_1d', 'price_change_3d', 'max_surge_after_news',
    'date.1', 'hotel.1'
]

news_df = news_df.drop(columns=cols_to_drop)

