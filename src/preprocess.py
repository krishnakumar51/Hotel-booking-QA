import pandas as pd
import sqlite3

# Load and clean data
df = pd.read_csv('../data/hotel_bookings.csv')
df['country'].fillna('Unknown', inplace=True)
df.dropna(subset=['adr'], inplace=True)
df = df[df['adr'] >= 0]
df['arrival_date'] = pd.to_datetime(
    df['arrival_date_year'].astype(str) + '-' + 
    df['arrival_date_month'] + '-' + 
    df['arrival_date_day_of_month'].astype(str), 
    errors='coerce'
)
df['revenue'] = df['adr'] * (df['stays_in_weekend_nights'] + df['stays_in_week_nights'])
df.dropna(subset=['arrival_date'], inplace=True)

# Store in SQLite
conn = sqlite3.connect('../db/bookings.db')
df.to_sql('bookings', conn, if_exists='replace', index=False)
conn.close()
print("Data preprocessed and saved to db/bookings.db")