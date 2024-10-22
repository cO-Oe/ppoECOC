# ppoECOC

A novel approach to Error-Correcting Output Codes (ECOC) matrix optimization using reinforcement learning, specifically the Proximal Policy Optimization (PPO) algorithm. This method addresses the challenging problem of multi-class imbalanced data classification.

## Installation

### Prerequisites
- Python 3.10 
- Conda (recommended) or pip

### Setup Instructions

1. Clone the Repository
```bash
git clone https://github.com/cO-Oe/ppoECOC.git
cd ppoECOC-main
```
2. Dataset Preparation

Extract (Unzip) the contents of dataset.zip under the project directory
Make sure all dataset files are properly placed in their respective directories

3. Environment Setup

Option 1: Using conda (Recommended)
```bash 
conda env create -f environment.yml
conda activate ppoECOC
```
Option 2: Using pip
```bash
conda create --name ppoECOC
conda activate ppoECOC
pip install -r requirements.txt
```
Running the Program

```bash
python main.py
```

4. Results

The results will be recorded under results/ directory, saving each single run with it's seed as results.csv in individual subdirectories.
For each entry(row) in results.csv, it will create another subdirectory named with the same unique id as in the results.csv, and the ECOC matrix and SVM classifiers are stored under it.
After all the seeds are run, the extractor will extract the best result and log it in results.log for direct readability.

5. Using with your own custom datasets

First put your dataset (.data/.arff/.csv) under the dataset/ directory and navigate to the **tabdata_parser** function at **utils/dataset.py**.
Simply ensure the labels are located at the last column before below lines:
```python
# separate label and features
label = new_df.columns[new_df.shape[1]-1]
features = new_df.columns[:new_df.shape[1]-1]

return new_df[features], new_df[label]
```


## Project Structure
ppoECOC/
├── agent/                  # For ppo agent .pth storage
├── runs/                   # For tensorboard event file storage
├── results/               # Execution result
├── dataset/               # Dataset files
├── utils/
│   ├── dataset.py         # Dataset loader and preprocessings
│   ├── extract.py         # Result extractor helper functions
│   └── GSE.py             # Script to query GSE dataset
├── environment.yml        # Conda environment configuration
├── requirements.txt       # Python package requirements
├── main.py               # Main execution script
├── process.py            # Single dataset execution for main
├── agent.py              # ppo agent class with transformer
├── env.py                # RL environment implementation with ECOC
├── decode.py             # ECOC decoding methods
├── ppo.py                # PPO algorithm
├── results.log           # logging results, auto generate if not exist
├── experiment.log        # For status and error logging
└── README.md             # This file

License
MIT