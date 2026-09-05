"""Create the DuckDB database and execute the SQL layer in order."""
import duckdb

from carebridge.config import DB_PATH, RAW, SQL


def run_sql_dir(con: duckdb.DuckDBPyConnection, folder: str) -> None:
    paths = sorted((SQL / folder).glob("*.sql"))
    if not paths:
        print(f"[warn] no .sql files in {folder}")
    for path in paths:
        print(f"[sql ] {folder}/{path.name}")
        sql = path.read_text().replace("{{raw_dir}}", RAW.as_posix())
        con.execute(sql)


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    for folder in ("01_bronze", "02_silver", "03_gold"):
        run_sql_dir(con, folder)
    print("\n[done] tables:")
    print(con.execute("SHOW ALL TABLES").fetchdf()[["schema", "name"]].to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
