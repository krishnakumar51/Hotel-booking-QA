# Hotel-booking-QA/src/rag.py
import pandas as pd
import sqlite3
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load data
conn = sqlite3.connect('../db/bookings.db')  # Connect to the database
df = pd.read_sql_query("SELECT * FROM bookings", conn, parse_dates=['arrival_date'])
conn.close()

# Generate descriptions
descriptions = []
for idx, row in df.iterrows():
    desc = (f"Booking ID {idx}: Customer from {row['country']} booked a room "
            f"for {row['arrival_date'].strftime('%B %Y')}, revenue ${row['revenue']:.2f}, "
            f"{'canceled' if row['is_canceled'] else 'not canceled'}, lead time {row['lead_time']} days.")
    descriptions.append(desc)

# Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(descriptions, show_progress_bar=True)

# FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype(np.float32))
faiss.write_index(index, '../models/bookings_index.faiss')  # Write index after creating it
print("FAISS index created at ../models/bookings_index.faiss")