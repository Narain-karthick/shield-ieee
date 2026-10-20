from pathlib import Path

import requests


URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"
DESTINATION = Path(__file__).resolve().parents[1] / "data" / "yellow_tripdata_2025-01.parquet"


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.exists():
        print(f"Dataset already exists: {DESTINATION}")
        return
    with requests.get(URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        with DESTINATION.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                output.write(chunk)
    print(f"Downloaded dataset to {DESTINATION}")


if __name__ == "__main__":
    main()

