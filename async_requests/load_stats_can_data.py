import warnings
import asyncio
import re
import time
import pandas as pd
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Suppress FutureWarnings from stats_can and pandas
warnings.filterwarnings('ignore', category=FutureWarning)


SC_URL = "https://www150.statcan.gc.ca/t1/wds/rest/"
ENDPOINT = SC_URL + "getDataFromVectorsAndLatestNPeriods"


def normalize_vector_id(v):
    """Accept 32164132, '32164132', or 'v32164132' -> int 32164132."""
    if isinstance(v, str):
        m = re.search(r'(\d+)$', v)
        if not m:
            raise ValueError(f"Unrecognized vectorId format: {v!r}")
        v = m.group(1)
    return int(v)

def chunked(seq, n):
    """Yield successive n-sized chunks from seq."""
    seq = list(seq) if not isinstance(seq, list) else seq
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def make_session():
    s = requests.Session()
    retry = Retry(
        total=5, connect=5, read=5,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["POST"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
    })
    return s


def get_data_from_vectors_and_latest_n_periods(vectors, periods, *, chunk_size=75, pause=0.2, timeout=(5, 60)):
    """
    vectors: int | str | list[int|str]  (accepts 'v12345' too)
    periods: int  (latest N)
    chunk_size: keep modest to avoid timeouts; 50–100 is usually safe
    returns: list of 'object' dicts (one per vector)
    """
    # normalize inputs
    if isinstance(vectors, (str, int)):
        vectors = [vectors]
    norm_vectors = [normalize_vector_id(v) for v in vectors]
    if not isinstance(periods, int) or periods <= 0:
        raise ValueError("periods must be a positive integer")

    session = make_session()
    results = []
    for chunk in chunked(norm_vectors, chunk_size):
        payload = [{"vectorId": v, "latestN": periods} for v in chunk]
        try:
            resp = session.post(ENDPOINT, json=payload, timeout=timeout)
            # If StatCan returns 406, print the offending payload for quick debugging
            if resp.status_code == 406:
                raise requests.HTTPError(f"406 Not Acceptable. Payload likely malformed. Example item: {payload[0]}", response=resp)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ReadTimeout:
            # one retry with a longer read window for this batch
            resp = session.post(ENDPOINT, json=payload, timeout=(timeout[0], max(90, timeout[1]*2)))
            resp.raise_for_status()
            data = resp.json()

        # collect objects (WDS returns list of {object, status, ...})
        for row in data:
            if "object" in row:
                results.append(row["object"])
            else:
                # keep some context if something odd returns for a vector
                status = row.get("status", {})
                raise RuntimeError(f"Missing 'object' in response row. status={status}")

        time.sleep(pause)  # be polite / avoid spikes

    return results


def vectors_to_df(vectors, periods=1, start_release_date=None, end_release_date=None):
    """Get DataFrame of vectors with n periods data.

    Wrapper on get_data_from_vectors_and_latest_n_periods function to turn the 
    resulting list of JSONs into a DataFrame

    Parameters
    ----------
    vectors: str or list of str
        vector numbers to get info for
    periods: int
        number of periods to retrieve data for
    start_release_date: datetime.date (not supported in this version)
        start release date for the data
    end_release_date: datetime.date (not supported in this version)
        end release date for the data

    Returns
    -------
    df: DataFrame
        vectors as columns and ref_date as the index (not release date)
    """
    df = pd.DataFrame()
    
    # Only support latest N periods mode
    if (start_release_date is not None) or (end_release_date is not None):
        raise NotImplementedError("Date range filtering is not supported in this version")
    
    start_list = get_data_from_vectors_and_latest_n_periods(vectors, periods)
    
    for vec in start_list:
        name = "v" + str(vec["vectorId"])
        # If there's no data for the series just skip it
        if not vec["vectorDataPoint"]:
            continue
        ser = (
            pd.DataFrame(vec["vectorDataPoint"])
            .assign(refPer=lambda x: pd.to_datetime(x["refPer"], errors="ignore"))
            .set_index("refPer")
            .rename(columns={"value": name})
            .filter([name])
        )
        df = pd.concat([df, ser], axis=1, sort=True)
    return df

async def import_stat_can_vector(vector_province_map, latestN, value_name, max_retries=5, retry_delay=2):
    """
    Asynchronously fetch StatsCan vector data for multiple provinces in parallel.
    
    Args:
        vector_province_map: Dictionary mapping vector_id to province name
        latestN: Number of latest periods to fetch
        value_name: Name to assign to the value column
        max_retries: Maximum number of retry attempts for failed requests
        retry_delay: Delay in seconds between retries
    
    Returns:
        DataFrame with combined data from all vectors
    """
    
    async def fetch_vector(vector_id, province):
        """Helper function to fetch a single vector asynchronously with retry logic"""
        for attempt in range(max_retries):
            try:
                # Create a new StatsCan instance for each request to avoid connection sharing
                # Run the synchronous I/O operation in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                vc = await loop.run_in_executor(
                    None, 
                    # lambda: StatsCan().vectors_to_df_remote(vector_id, latestN, None, None)
                    lambda: vectors_to_df(vector_id, latestN, None, None)
                )
                break  # Exit the retry loop if successful
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed for vector {vector_id} ({province}): {str(e)}")
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    print(f"Failed to fetch vector {vector_id} ({province}) after {max_retries} attempts: {str(e)}")
                    raise
            
        # Add metadata columns
        vc["Province"] = province
        vc["VectorID"] = vector_id
        
        # Rename column named after vector_id to 'Value' if needed
        if vector_id in vc.columns:
            vc = vc.rename(columns={vector_id: "Value"})
        
        # Reset index to convert index to a column if necessary
        vc.reset_index(inplace=True)
        
        # Ensure consistent column order before returning
        desired_columns = ['refPer', 'Value', 'Province', 'VectorID']
        vc = vc[desired_columns]
        
        return vc

    
    # Create tasks for all vector fetches
    tasks = [
        fetch_vector(vector_id, province) 
        for vector_id, province in vector_province_map.items()
    ]
    
    # Execute all tasks concurrently
    vcs = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and collect successful results
    successful_vcs = [vc for vc in vcs if not isinstance(vc, Exception)]
    failed_count = len(vcs) - len(successful_vcs)
    
    if failed_count > 0:
        raise ValueError(f"Warning: {failed_count} out of {len(vcs)} vectors failed to fetch")

    
    # Concatenate all individual vector DataFrames into one
    all_vcs = pd.concat(successful_vcs, ignore_index=True)
    all_vcs["Value_name"] = value_name
    return all_vcs


#details on stat can-> https://www150.statcan.gc.ca/t1/tbl1/en/cv!recreate.action?pid=3410028501&selectedNodeIds=1D2,1D3,1D4,1D5,1D6,1D7,1D8,1D9,1D10,1D11,1D12,1D13,1D14,2D4,3D6,4D5,5D1&checkedLevels=&refPeriods=20220101,20250301&dimensionLayouts=layout3,layout3,layout3,layout3,layout3,layout2&vectorDisplay=true
construction_vector_province_map = {'v1578996785':'NL', 
                       'v1579028720':'PE', 
                       'v1579049632':'NS', 
                       'v1579014410':'NB', 
                       'v1579053282':'QC', 
                       'v1578994708':'ON', 
                       'v1579057398':'AB', 
                       'v1579014981':'BC'
}

#details on stat can-> https://www150.statcan.gc.ca/t1/tbl1/en/cv!recreate.action?pid=1810000101&selectedNodeIds=1D2,1D3,1D4,1D5,1D6,1D7,1D8,1D9,1D10,1D14,1D15,1D16,1D17,2D2&checkedLevels=&refPeriods=20220101,20250401&dimensionLayouts=layout3,layout3,layout2&vectorDisplay=true
gasoline_vector_province_map = {'v735082':'NL', 
                       'v735092':'PE', 
                       'v735093':'NS', 
                       'v735094':'NB', 
                       'v735096':'QC', 
                       'v735098':'ON', 
                       'v735087':'AB',
                       'v735088':'BC'
}

#details on stat can-> https://www150.statcan.gc.ca/t1/tbl1/en/cv!recreate.action?pid=2010008501&selectedNodeIds=2D2,5D1,6D1&checkedLevels=0D2,2D1,3D1&refPeriods=20220101,20250301&dimensionLayouts=layout3,layout3,layout3,layout3,layout3,layout3,layout2&vectorDisplay=true
car_vector_province_map = {'v1617815582':'NL', 
                       'v1617815592':'PE', 
                       'v1617815602':'NS', 
                       'v1617815612':'NB', 
                       'v1617815622':'QC', 
                       'v1617815628':'ON', 
                       'v1617815654':'AB',
                       'v1617815664':'BC'
}
#details on stat can-> https://www150.statcan.gc.ca/t1/tbl1/en/cv!recreate.action?pid=1810000401&selectedNodeIds=1D3,1D5,1D7,1D9,1D11,1D14,1D23,1D26,2D2,2D184&checkedLevels=&refPeriods=20220101,20250401&dimensionLayouts=layout3,layout3,layout2&vectorDisplay=true
CPI_vector_province_map = {'v41691244':'NL', 
                       'v41691379':'PE', 
                       'v41691513':'NS', 
                       'v41691648':'NB', 
                       'v41691783':'QC', 
                       'v41691919':'ON', 
                       'v41692327':'AB',
                       'v41692462':'BC'
}
#details on stat can-> https://www150.statcan.gc.ca/t1/tbl1/en/cv.action?pid=1710000901
population_vector_province_map = {'v2':'NL', 
                       'v8':'PE', 
                       'v9':'NS', 
                       'v10':'NB', 
                       'v11':'QC', 
                       'v12':'ON', 
                       'v15':'AB',
                       'v3':'BC'
}

def save_statcan_dataframes(population_df, construction_df, gasoline_df, car_df, CPI_df, data_folder="data"):
    """
    Save StatsCan dataframes to CSV files in the specified folder.
    
    Args:
        population_df: Population data DataFrame
        construction_df: Construction permits data DataFrame  
        gasoline_df: Gasoline price data DataFrame
        car_df: New vehicle sales data DataFrame
        CPI_df: Consumer Price Index data DataFrame
        data_folder: Folder to save CSV files (default: "data")
    """
    # Create data folder if it doesn't exist
    Path(data_folder).mkdir(parents=True, exist_ok=True)
    
    # Define dataframes and their corresponding filenames
    dataframes = {
        'population.csv': population_df,
        'construction_permits.csv': construction_df,
        'gasoline_prices.csv': gasoline_df,
        'vehicle_sales.csv': car_df,
        'consumer_price_index.csv': CPI_df
    }
    
    # Save each dataframe to CSV
    for filename, df in dataframes.items():
        filepath = Path(data_folder) / filename
        df.to_csv(filepath, index=False)
        print(f"Saved {filename} to {filepath}")

async def main():
    """Main async function to fetch all StatsCan data concurrently"""
    # Run all data fetches in parallel
    population_df, construction_df, gasoline_df, car_df, CPI_df = await asyncio.gather(
        import_stat_can_vector(population_vector_province_map, 100, "population"),
        import_stat_can_vector(construction_vector_province_map, 100, "construction_permits_units"),
        import_stat_can_vector(gasoline_vector_province_map, 100, "gasoline_price_self_service_station"),
        import_stat_can_vector(car_vector_province_map, 100, "new_vehicles_sales_units"),
        import_stat_can_vector(CPI_vector_province_map, 100, "consumer_price_index_all")
    )
    
    return population_df, construction_df, gasoline_df, car_df, CPI_df

# Run the async main function
if __name__ == "__main__":
    population_df, construction_df, gasoline_df, car_df, CPI_df = asyncio.run(main())
    # Save all dataframes to CSV files
    save_statcan_dataframes(population_df, construction_df, gasoline_df, car_df, CPI_df)
    