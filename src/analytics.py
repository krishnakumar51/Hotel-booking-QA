import sqlite3
import pandas as pd

def get_revenue_trends():
    conn = sqlite3.connect('../db/bookings.db')
    df = pd.read_sql_query(
        "SELECT strftime('%Y-%m', arrival_date) AS month, SUM(revenue) AS total_revenue "
        "FROM bookings GROUP BY month ORDER BY month", conn
    )
    conn.close()
    return df.to_dict(orient='records')

def get_cancellation_rate():
    conn = sqlite3.connect('../db/bookings.db')
    df = pd.read_sql_query(
        "SELECT strftime('%Y-%m', arrival_date) AS month, "
        "100.0 * SUM(CASE WHEN is_canceled = 1 THEN 1 ELSE 0 END) / COUNT(*) AS cancel_rate "
        "FROM bookings GROUP BY month ORDER BY month", conn
    )
    conn.close()
    return df.to_dict(orient='records')

def get_geo_distribution():
    conn = sqlite3.connect('../db/bookings.db')
    df = pd.read_sql_query(
        "SELECT country, COUNT(*) AS booking_count FROM bookings "
        "GROUP BY country ORDER BY booking_count DESC LIMIT 10", conn
    )
    conn.close()
    return df.to_dict(orient='records')

def get_lead_time_dist():
    conn = sqlite3.connect('../db/bookings.db')
    df = pd.read_sql_query("SELECT lead_time FROM bookings", conn)
    conn.close()
    return df['lead_time'].tolist()