
import psycopg
import streamlit as st


# Open a secure connection to our Supabase PostgreSQL database.
def get_connection():
    settings = st.secrets["database"]

    return psycopg.connect(
        host=settings["host"],
        port=settings["port"],
        dbname=settings["dbname"],
        user=settings["user"],
        password=settings["password"],
        sslmode=settings["sslmode"],
        connect_timeout=10,
    )


# Test whether the application can reach the database.
def test_connection():
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_user, current_database();")
            return cursor.fetchone()
