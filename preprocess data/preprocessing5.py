import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
import os

def load_and_process_data(data_path, datafolder, site):
    with open(data_path, 'rb') as file:
        data = pickle.load(file)

    file_path_covid = f"{datafolder}{site}/p0_covid_status_{site}.parquet"
    file_path_onset = f"{datafolder}{site}/p0_onset_{site}.parquet"

    p0_covid_data = pd.read_parquet(file_path_covid)
    p0_onset_data = pd.read_parquet(file_path_onset)

    data['PATID'] = data['PATID'].astype(str)
    data['ENCOUNTERID'] = data['ENCOUNTERID'].astype(str)

    p0_onset_data['PATID'] = p0_onset_data['PATID'].astype(str)
    p0_onset_data['ENCOUNTERID'] = p0_onset_data['ENCOUNTERID'].astype(str)

    p0_covid_data['PATID'] = p0_covid_data['PATID'].astype(str)
    p0_covid_data['ENCOUNTERID'] = p0_covid_data['ENCOUNTERID'].astype(str)

    data = data.merge(
        p0_onset_data[['PATID', 'ENCOUNTERID', 'ADMIT_DATE']], 
        on=['PATID', 'ENCOUNTERID'], 
        how='left'
    )

    data = data.merge(
        p0_covid_data[['PATID', 'ENCOUNTERID', 'BCCOVID']], 
        on=['PATID', 'ENCOUNTERID'], 
        how='left'
    )

    return data


def rename_columns_with_mapping(data, mapping_path="covid_column_mapping_dict.pkl"):
    all_columns = set(data.columns)

    try:
        with open(mapping_path, "rb") as f:
            mapping_dict = pickle.load(f)
    except FileNotFoundError:
        mapping_dict = {}

    prefixes = ['LAB', 'DX', 'PX', 'MED']
    counter = {prefix: 1 for prefix in prefixes}

    for existing_mapping in mapping_dict.values():
        for prefix in prefixes:
            if existing_mapping.startswith(prefix):
                num = int(existing_mapping.split("_")[1])
                counter[prefix] = max(counter[prefix], num + 1)

    new_columns_to_map = all_columns.difference(mapping_dict.keys())

    for col in new_columns_to_map:
        for prefix in prefixes:
            if col.startswith(prefix):
                new_name = f"{prefix}_{counter[prefix]}"
                mapping_dict[col] = new_name
                counter[prefix] += 1
                break

    data.rename(columns=mapping_dict, inplace=True)

    with open(mapping_path, "wb") as f:
        pickle.dump(mapping_dict, f)

    return data

def apply_mice_imputation(data_by_year, excluded_columns=None, max_iter=80, random_state=19):
    if excluded_columns is None:
        excluded_columns = ['ENCOUNTERID', 'SINCE_ADMIT', 'PATID', 'ADMIT_DATE', 'BCCOVID', 'FLAG']

    imputed_data_by_year = {}

    for year, data in tqdm(data_by_year.items(), desc="MICE Processing per Year"):
        print(f"Processing year {year} for MICE imputation.")

        to_impute = data.drop(columns=excluded_columns, errors='ignore')
        excluded_data = data[excluded_columns] if all(col in data.columns for col in excluded_columns) else data[data.columns.intersection(excluded_columns)]

        nan_columns = to_impute.columns[to_impute.isnull().any()]
        nan_columns_filtered = [col for col in nan_columns if to_impute[col].notna().sum() > 0]

        if not nan_columns_filtered:
            print(f"No columns with NaN in year {year}.")
            imputed_data_by_year[year] = data
            continue

        imputer = IterativeImputer(max_iter=max_iter, verbose=2, random_state=random_state)
        to_impute[nan_columns_filtered] = imputer.fit_transform(to_impute[nan_columns_filtered])

        imputed_full_data = pd.concat([to_impute, excluded_data], axis=1)
        imputed_data_by_year[year] = imputed_full_data

        print(f"Completed MICE imputation for year {year}.")

    return imputed_data_by_year

def save_imputed_data_by_year(imputed_data_by_year, save_path="./"):
    os.makedirs(save_path, exist_ok=True)

    for year, imputed_data in imputed_data_by_year.items():
        file_name = f"imputed_data_{year}.csv"
        file_path = os.path.join(save_path, file_name)
        imputed_data.to_csv(file_path, index=False)
        print(f"Imputed data for year {year} saved to {file_path}.")