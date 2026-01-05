import os
from pandas import DataFrame, concat, read_parquet


FOLDER_PATH = 'app/projects/pltv/data/cache'


def get_file_path(file_name: str, csv: bool = False) -> str:
    os.makedirs(FOLDER_PATH, exist_ok=True)
    if csv:
        return os.path.join(FOLDER_PATH, file_name + '.csv')
    else:
        return os.path.join(FOLDER_PATH, file_name + '.parquet')
    
def overwrite_or_append_parquet(file_name: str, df: DataFrame, overwrite: bool = False, csv: bool = False) -> None:
    file_path = get_file_path(file_name, csv)

    if os.path.exists(file_path) and not overwrite:
        existing_df = read_parquet(file_path)
        combined_df = concat([existing_df, df], ignore_index=True)
        if csv:
            combined_df.to_csv(file_path, index=False)
        else:
            combined_df.to_parquet(file_path)
    else:
        if csv:
            df.to_csv(file_path, index=False)
        else:
            df.to_parquet(file_path)
