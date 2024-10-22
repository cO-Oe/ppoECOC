import logging
import os
import numpy as np

from utils.dataset import TabData
from utils.extract import process_folders
from process import run
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)

def log_result(dataset_name: str):
    """
    Log experiment results to a results log file.
    Results are stored in the runs directory with experiment details.
    """
    log_file = "results.log"
    
    selected_columns = ['f2', 'f1', 'g_mean', 'mauc']
    results = process_folders(dataset_name, "./results", selected_columns)

    # Find the entry with maximum total
    best_result = max(results, key=lambda x: (x['f2'] + x['g_mean'] + x['mauc']))

    # Write results to log file
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Dataset: {dataset_name}\n")
        
        f.write("Best Metrics:\n")
        metric_names = ['g_mean', 'f2', 'mauc', 'f1']
        for metric in metric_names:
            f.write(f"{metric}: {best_result[metric]:.4f}\n")
    
        f.write(f"{'='*80}\n")

def main():
    td = TabData("./dataset")
    specify_list = [] # add dataset_name into if you want to run a specify dataset ["golub.csv", "breast-cancer.csv"]

    for file in td.file_list:
        dataset_name = os.path.basename(file)
        print(dataset_name)
        num_classes = td.count_class(file)

        if len(specify_list) != 0 and dataset_name not in specify_list:
            continue

        logging.info(f"Starting dataset {dataset_name} with num_class {num_classes}")

        try:
            run(f"{dataset_name}", num_classes)
            logging.info(f"Completed dataset {dataset_name}")
            log_result(dataset_name)
        except Exception as e:
            logging.error(f"Error processing dataset {dataset_name}: {str(e)}")
            continue
  
    logging.info("All datasets processed")

if __name__ == "__main__":
    main()