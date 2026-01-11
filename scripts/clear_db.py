import psycopg

DB_DSN = "postgresql://cvuser:cvpass@localhost:5432/cvdb"
TABLE = "cv_profiles"

def clear_table():
    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {TABLE};")
        conn.commit()
    print(f"✅ Table '{TABLE}' cleared (TRUNCATE).")

if __name__ == "__main__":
    clear_table()
