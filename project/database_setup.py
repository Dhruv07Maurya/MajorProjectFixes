# This script is run once to create the database and its tables.
import sqlite3

DB_FILE = "greenhouse.db"

def setup_database():
    """Creates the database tables if they don't already exist."""
    print("Setting up database...")
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Create table for historical sensor readings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                temperature REAL,
                humidity REAL,
                light_intensity REAL,
                soil_moisture REAL,
                air_quality REAL
            )
        ''')

        # Create table for logging system actions and alerts
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS action_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source TEXT,
                action TEXT
            )
        ''')

        conn.commit()
        conn.close()
        print(f"Database '{DB_FILE}' and tables created successfully.")
    except Exception as e:
        print(f"Error setting up database: {e}")

if __name__ == '__main__':
    setup_database()


