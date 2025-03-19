import streamlit as st
import requests
import pandas as pd

st.title("Booking Analytics & QA System")

option = st.sidebar.selectbox("Choose an option", ["Analytics", "Ask a Question"])

if option == "Analytics":
    analytic_type = st.selectbox("Select Analytic", ["revenue_trends", "cancellation_rate", "geo_distribution", "lead_time_dist"])
    if st.button("Get Analytics"):
        try:
            response = requests.post("http://localhost:8000/analytics", json={"analytic": analytic_type})
            response.raise_for_status()
            data = response.json()
            
            # Extract table and summary
            table_data = data.get("table", [])
            summary = data.get("summary", "No summary available")
            
            # Display table data
            if analytic_type == "lead_time_dist":
                # For lead_time_dist, show as a list or basic stats
                st.subheader("Lead Time Distribution")
                st.write("Sample Lead Times:", table_data[:10])  # Show first 10 for brevity
                st.write(f"Min: {min(table_data)}, Max: {max(table_data)}, Avg: {sum(table_data)/len(table_data):.1f}")
            else:
                # For table-based analytics, use DataFrame
                st.subheader(f"{analytic_type.replace('_', ' ').title()}")
                st.dataframe(pd.DataFrame(table_data))
            
            # Display summary
            st.subheader("Summary")
            st.write(summary)
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
            if 'response' in locals():
                st.error(f"Response: {response.text}")
        except ValueError as e:
            st.error(f"Error processing response: {e}")
elif option == "Ask a Question":
    question = st.text_input("Enter your question")
    if st.button("Ask"):
        try:
            response = requests.post("http://localhost:8000/ask", json={"question": question})
            response.raise_for_status()
            st.write(response.json().get("answer", "No answer provided"))
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
            if 'response' in locals():
                st.error(f"Response: {response.text}") 