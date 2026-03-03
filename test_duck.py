import duckdb

print("Trying single file:")
q1 = duckdb.query("SELECT count(*) FROM read_csv_auto('data/raw/Anand Vihar, New Delhi - DPCC_235/*.csv', ignore_errors=True)").fetchone()
print(q1)

print("Trying glob:")
q2 = duckdb.query("SELECT count(*) FROM read_csv_auto('data/raw/*/*.csv', ignore_errors=True)").fetchone()
print(q2)

print("Columns:")
df = duckdb.query("SELECT * FROM read_csv_auto('data/raw/*/*.csv', ignore_errors=True) LIMIT 5").to_df()
print(df.columns)
print(df.head())
