"""
SQLite Database Management Module for Diwali Sales Analytics Platform.
Handles schema initialization, transaction persistence, indexing, dynamic SQL queries,
and real-time ML prediction logging.
"""

import os
import sys
import sqlite3
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DB_PATH = os.path.join("data", "diwali_sales.db")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "diwali_sales_cleaned.csv")
SEGMENTS_DATA_PATH = os.path.join("data", "processed", "customer_segments.csv")


def get_db_connection(db_path=DB_PATH) -> sqlite3.Connection:
    """Create and return a SQLite database connection with row factory."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database(db_path=DB_PATH, force_reload=False) -> dict:
    """
    Initializes database schema, creates performance indexes,
    and populates data from processed CSVs if empty or force_reload is True.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # 1. Create Predictions Log Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        model_used TEXT,
        gender TEXT,
        age INTEGER,
        age_group TEXT,
        marital_status TEXT,
        state TEXT,
        zone TEXT,
        occupation TEXT,
        product_category TEXT,
        orders INTEGER,
        predicted_amount REAL,
        lower_bound REAL,
        upper_bound REAL
    );
    """)

    # Check if sales_transactions table already exists and has records
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sales_transactions';")
    table_exists = cursor.fetchone() is not None

    record_count = 0
    if table_exists:
        cursor.execute("SELECT COUNT(*) FROM sales_transactions;")
        record_count = cursor.fetchone()[0]

    stats = {"sales_records": record_count, "segments_records": 0, "status": "existing"}

    if not table_exists or record_count == 0 or force_reload:
        print(f"Ingesting datasets into SQLite database ({db_path})...")

        # Load cleaned sales data
        if os.path.exists(PROCESSED_DATA_PATH):
            df_sales = pd.read_csv(PROCESSED_DATA_PATH)
            # Rename column "Age Group" to "Age_Group" for clean SQL column naming
            df_sales_sql = df_sales.rename(columns={"Age Group": "Age_Group"})
            df_sales_sql.to_sql("sales_transactions", conn, if_exists="replace", index=False)
            stats["sales_records"] = len(df_sales_sql)
            print(f"  -> Inserted {len(df_sales_sql):,} records into 'sales_transactions'")

            # Create Performance Indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_user ON sales_transactions(User_ID);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_state ON sales_transactions(State);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_cat ON sales_transactions(Product_Category);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_occ ON sales_transactions(Occupation);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_gender ON sales_transactions(Gender);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_amount ON sales_transactions(Amount);")

        # Load customer segments
        if os.path.exists(SEGMENTS_DATA_PATH):
            df_seg = pd.read_csv(SEGMENTS_DATA_PATH)
            df_seg.to_sql("customer_segments", conn, if_exists="replace", index=False)
            stats["segments_records"] = len(df_seg)
            print(f"  -> Inserted {len(df_seg):,} records into 'customer_segments'")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_seg_user ON customer_segments(User_ID);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_seg_persona ON customer_segments(Segment_Name);")

        conn.commit()
        stats["status"] = "freshly_populated"

    conn.close()
    return stats


def log_prediction(
    model_used: str,
    gender: str,
    age: int,
    age_group: str,
    marital_status: str,
    state: str,
    zone: str,
    occupation: str,
    product_category: str,
    orders: int,
    predicted_amount: float,
    lower_bound: float,
    upper_bound: float,
    db_path=DB_PATH
):
    """Log an inference event into SQLite predictions_log."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO predictions_log (
        model_used, gender, age, age_group, marital_status, state, zone,
        occupation, product_category, orders, predicted_amount, lower_bound, upper_bound
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        model_used, gender, age, age_group, marital_status, state, zone,
        occupation, product_category, orders, predicted_amount, lower_bound, upper_bound
    ))
    conn.commit()
    conn.close()


def get_prediction_history(limit: int = 50, db_path=DB_PATH) -> pd.DataFrame:
    """Fetch recent model prediction logs from SQLite."""
    conn = get_db_connection(db_path)
    query = f"""
    SELECT id, timestamp, model_used, gender, age, state, occupation, 
           product_category, orders, predicted_amount, lower_bound, upper_bound
    FROM predictions_log
    ORDER BY id DESC
    LIMIT {limit};
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def execute_custom_query(query: str, db_path=DB_PATH) -> tuple[pd.DataFrame, str]:
    """Execute arbitrary read-only SQL query on the SQLite database."""
    # Basic safety check to prevent unintentional data drops
    forbidden = ["DROP", "DELETE", "TRUNCATE", "UPDATE", "ALTER", "INSERT"]
    first_word = query.strip().split()[0].upper() if query.strip() else ""
    if first_word in forbidden:
        return pd.DataFrame(), f"Error: Modifying statement '{first_word}' is restricted. Only SELECT queries are permitted in the analytics explorer."

    conn = get_db_connection(db_path)
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df, ""
    except Exception as e:
        conn.close()
        return pd.DataFrame(), str(e)


def main():
    print(f"Initializing SQLite database at: {DB_PATH}")
    stats = init_database(force_reload=True)
    print("\nDatabase initialization complete!")
    print(f"  Sales Records: {stats['sales_records']:,}")
    print(f"  Customer Segments: {stats['segments_records']:,}")
    print(f"  Database file size: {os.path.getsize(DB_PATH):,} bytes")


if __name__ == "__main__":
    main()

