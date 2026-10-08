import sqlite3
import pandas as pd
from utils.database import DB_PATH

def init_db(db_name=DB_PATH):
    """Initialise la base de données et les tables nécessaires."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS races (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            reunion TEXT,
            course TEXT,
            data TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(df, table_name="races", db_name=DB_PATH):
    """Sauvegarde un DataFrame dans SQLite."""
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists="append", index=False)
    conn.close()