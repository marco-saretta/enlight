import duckdb

con = duckdb.connect('onshore_wind.duckdb')

#con.execute("""
#            CREATE TABLE wind_capacity AS
#            SELECT * FROM read_csv_auto('00. Onshore wind.csv')
#            """)

# 3. (Optional) Check it worked
df = con.execute("SELECT * FROM wind_capacity LIMIT 8000").fetchdf()
print(df)


# Run a query (e.g., filter or aggregation)
df = con.execute("""
    SELECT * FROM wind_capacity WHERE BE > 100
""").fetchdf()

print(df.head())

# List all tables
tables = con.execute("SHOW TABLES").fetchdf()
print(tables)

# See schema of a table
schema = con.execute("DESCRIBE wind_capacity").fetchdf()
print(schema)
