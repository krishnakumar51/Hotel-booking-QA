# Hotel-booking-QA/src/api.py
from fastapi import FastAPI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
import pandas as pd
from analytics import get_revenue_trends, get_cancellation_rate, get_geo_distribution, get_lead_time_dist
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('../models/bookings_index.faiss')
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)  # Temperature 0 for consistency

# Load descriptions from the database for RAG fallback
conn = sqlite3.connect('../db/bookings.db')
df = pd.read_sql_query("SELECT * FROM bookings", conn, parse_dates=['arrival_date'])
conn.close()
descriptions = [
    f"Booking ID {idx}: Customer from {row['country']} booked a room "
    f"for {row['arrival_date'].strftime('%B %Y')}, revenue ${row['revenue']:.2f}, "
    f"{'canceled' if row['is_canceled'] else 'not canceled'}, lead time {row['lead_time']} days."
    for idx, row in df.iterrows()
]

# Define Pydantic models for request bodies
class AnalyticsRequest(BaseModel):
    analytic: str

class QuestionRequest(BaseModel):
    question: str

# Optimized SQL generation prompt
sql_prompt = ChatPromptTemplate.from_template(
    """
    You are an expert SQL query generator with deep knowledge of the hotel booking dataset. Your task is to generate a precise, repeatable SQLite query to answer the user's question consistently. Output ONLY the raw SQL query (no Markdown, no explanations, no extra text) or "NO_SQL_QUERY" if the question cannot be answered with SQL. Here’s the schema and insights:

    Table: bookings
    - booking_id: INTEGER (unique identifier for each booking)
    - arrival_date: DATETIME (arrival date, e.g., '2015-07-01', stored as YYYY-MM-DD)
    - country: TEXT (customer country code, e.g., 'PRT', 'USA', can be 'Unknown')
    - adr: REAL (average daily rate in dollars)
    - is_canceled: INTEGER (0 = not canceled, 1 = canceled)
    - lead_time: INTEGER (days between booking and arrival)
    - revenue: REAL (total revenue, adr * total nights)

    Data Insights:
    - Spans July 2015 to August 2017.
    - Approx. totals: 2015 (~22,000 bookings), 2016 (~56,000), 2017 (~40,000).
    - Use strftime('%Y', arrival_date) for year extraction.

    User Question: {user_question}

    Output ONLY the SQLite query or "NO_SQL_QUERY":
    """
)

# Answer generation prompt for /ask endpoint
answer_prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant providing insights about hotel bookings. Based on the exact database query result, provide a concise answer followed by a brief, data-driven insight. Do NOT restate the user’s question—just give the answer and insight directly.

    Query Result: {query_result}

    Provide the answer and concise insight:
    """
)

# Analytics summary generation prompt
analytics_summary_prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant summarizing hotel booking analytics. Based on the provided data, generate a concise paragraph summarizing the key findings in a natural, readable format. Focus on trends, patterns, or notable points relevant to the analytic type. Do not include the raw data in the paragraph—just interpret it.

    Analytic Type: {analytic_type}
    Data: {data}

    Provide a summary paragraph:
    """
)

# Create the SQL generation chain
sql_chain = RunnableSequence(sql_prompt | llm | StrOutputParser())

# Create the answer generation chain for /ask
answer_chain = RunnableSequence(answer_prompt | llm | StrOutputParser())

# Create the analytics summary generation chain
analytics_summary_chain = RunnableSequence(analytics_summary_prompt | llm | StrOutputParser())

# Function to clean SQL query output
def clean_sql_query(query: str) -> str:
    cleaned = re.sub(r'```sql|```|[\r\n].*', '', query, flags=re.DOTALL).strip()
    return cleaned if cleaned else "NO_SQL_QUERY"

@app.post("/analytics")
async def analytics(request: AnalyticsRequest):
    analytic = request.analytic
    if analytic == "revenue_trends":
        table_data = get_revenue_trends()
        summary = analytics_summary_chain.invoke({
            "analytic_type": "revenue trends",
            "data": str(table_data)
        })
        return {"table": table_data, "summary": summary}
    elif analytic == "cancellation_rate":
        table_data = get_cancellation_rate()
        summary = analytics_summary_chain.invoke({
            "analytic_type": "cancellation rate",
            "data": str(table_data)
        })
        return {"table": table_data, "summary": summary}
    elif analytic == "geo_distribution":
        table_data = get_geo_distribution()
        summary = analytics_summary_chain.invoke({
            "analytic_type": "geographical distribution",
            "data": str(table_data)
        })
        return {"table": table_data, "summary": summary}
    elif analytic == "lead_time_dist":
        table_data = get_lead_time_dist()
        summary = analytics_summary_chain.invoke({
            "analytic_type": "lead time distribution",
            "data": str([min(table_data), max(table_data), sum(table_data)/len(table_data)])
        })
        return {"table": table_data, "summary": summary}
    else:
        return {"error": "Invalid analytic type"}

@app.post("/ask")
async def ask(request: QuestionRequest):
    question = request.question
    logger.info(f"Received question: {question}")
    
    # Generate SQL query
    raw_sql_query = sql_chain.invoke({"user_question": question})
    logger.info(f"Raw generated SQL query: {raw_sql_query}")
    
    # Clean the SQL query
    sql_query = clean_sql_query(raw_sql_query)
    logger.info(f"Cleaned SQL query: {sql_query}")
    
    # Check if the LLM generated a valid SQL query
    if sql_query != "NO_SQL_QUERY":
        conn = sqlite3.connect('../db/bookings.db')
        cursor = conn.cursor()
        try:
            cursor.execute(sql_query)
            result = cursor.fetchall()
            logger.info(f"Query result: {result}")
            if not result:
                return {"answer": "No data found for this query."}
            
            if len(result) == 1 and len(result[0]) == 1:
                query_result = str(result[0][0])
                if "how many" in question.lower() and float(query_result) < 0:
                    raise ValueError("Invalid count result")
            else:
                columns = [description[0] for description in cursor.description]
                query_result = str([dict(zip(columns, row)) for row in result])
            
            # Generate a contextual answer
            contextual_answer = answer_chain.invoke({
                "query_result": query_result
            })
            logger.info(f"Generated answer: {contextual_answer}")
            return {"answer": contextual_answer}
        
        except (sqlite3.Error, ValueError) as e:
            logger.error(f"SQL execution failed: {e}, falling back to RAG")
        finally:
            conn.close()
    
    # Fallback to RAG
    logger.info("Falling back to RAG")
    query_embedding = model.encode([question])
    distances, indices = index.search(query_embedding.astype(np.float32), k=5)
    retrieved_docs = [descriptions[idx] for idx in indices[0]]
    context = "\n".join(retrieved_docs)
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    logger.info(f"RAG fallback answer: {response.content}")
    return {"answer": response.content}