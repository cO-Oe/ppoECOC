import os
import pandas as pd

# Define the path to the main folder
main_folder_path = "./results"  # Update with your path

def get_selected_columns_of_last_row(file_path, columns):
    """
    Read the CSV and return the selected columns of the last row as a dictionary.
    """
    try:
        df = pd.read_csv(file_path)

        # Check if all columns exist in the DataFrame
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns in {file_path}: {missing_columns}")
            return None
        
        # Get only the required columns for the last row
        return df[columns].iloc[-1].to_dict()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def process_folders(dataset_name, main_folder_path, columns):
    """
    Loop through subfolders in the main folder and print the selected columns of the last row of results.csv.
    """
    results = []
    for subfolder_name in os.listdir(main_folder_path):
        subfolder_path = os.path.join(main_folder_path, subfolder_name)
        if dataset_name not in subfolder_name:
            continue
        # Check if it is a directory
        if os.path.isdir(subfolder_path):
            result_csv_path = os.path.join(subfolder_path, "results.csv")

            # Check if the results.csv file exists in the subfolder
            if os.path.exists(result_csv_path):
                last_row = get_selected_columns_of_last_row(result_csv_path, columns)
                if last_row:
                    result = {'name': subfolder_name}
                    result.update(last_row)
                    
                    results.append(result)
            else:
                print(f"No results.csv found in {subfolder_name}")

    return results

if __name__ == "__main__":
    dataset_name = str(input("Dataset name:"))
    selected_columns = ['f2', 'g_mean', 'mauc']
    results = process_folders(dataset_name, main_folder_path, selected_columns)
    print(results)
