import sqlite3

# ---------- CONNECT ----------
def connect_db():
    return sqlite3.connect("users.db")


# ---------- CREATE TABLES ----------
def create_tables():
    conn = connect_db()
    cursor = conn.cursor()

    # USERS TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT
    )
    """)

    # PREDICTIONS TABLE (FULL FEATURES)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        no_of_dependents INTEGER,
        education INTEGER,
        self_employed INTEGER,
        income_annum REAL,
        loan_amount REAL,
        loan_term REAL,
        cibil_score REAL,
        residential_assets_value REAL,
        commercial_assets_value REAL,
        luxury_assets_value REAL,
        bank_asset_value REAL,
        prediction INTEGER,
        probability REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("PRAGMA table_info(predictions)")
    prediction_columns = [row[1] for row in cursor.fetchall()]

    if "income_annum" not in prediction_columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN income_annum REAL")

    if "income" in prediction_columns:
        cursor.execute("""
        UPDATE predictions
        SET income_annum = income
        WHERE income_annum IS NULL
        """)

    conn.commit()
    conn.close()


# ---------- REGISTER ----------
def register_user(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return "exists"
    finally:
        conn.close()


# ---------- LOGIN ----------
def login_user(username, password):
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    )

    result = cursor.fetchone()
    conn.close()

    return result is not None


# ---------- SAVE FULL PREDICTION ----------
def save_prediction(username, income_annum, loan_amount, prediction, probability):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (username, income_annum, loan_amount, prediction, probability)
        VALUES (?, ?, ?, ?, ?)
        """, (username, income_annum, loan_amount, prediction, probability))

    conn.commit()
    conn.close()


# ---------- GET USER HISTORY ----------
def get_user_history(username):
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM predictions
    WHERE username=?
    ORDER BY created_at DESC
    """, (username,))

    rows = cursor.fetchall()
    conn.close()

    return rows
