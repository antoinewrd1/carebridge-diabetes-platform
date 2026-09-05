"""Execute every SQL file in dependency order, stopping at the first failure."""
import duckdb
from carebridge.config import DB_PATH, RAW, SQL


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    for folder in ("01_bronze", "02_silver", "03_gold"):
        for path in sorted((SQL / folder).glob("*.sql")):
            sql = path.read_text().replace("{{raw_dir}}", RAW.as_posix())
            try:
                con.execute(sql)
                print(f"  ok   {folder}/{path.name}")
            except Exception as e:
                print(f"  FAIL {folder}/{path.name}")
                print(f"       {type(e).__name__}: {e}")
                return
    print("\nall SQL parsed and executed")


if __name__ == "__main__":
    main()
