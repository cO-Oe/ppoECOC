import pandas as pd
import GEOparse
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
import re

logging.basicConfig(level=logging.INFO)

def classify_sample(title):
    """
    Classify samples based on their title patterns into 5 specific classes:
    1. Normal_Breast
    2. Cancer_Cell_Line
    3. Other_Cell_Line
    4. Primary_BC
    5. other
    """
    title = title.upper()
    
    # Standard cancer cell lines
    cell_lines = ['MCF7', 'SK-BR-3', 'BT-474', 'HS578T', 'BT-549', 
                 'MDA-MB-231', 'T47D', 'RPMI-8226', 'MOLT4', 'SW872']
    
    # Cell line markers
    cell_markers = ['HMEC', 'HMVEC', 'HUVEC', '184']
    
    # Check for normal breast tissue
    if 'NORMAL BREAST' in title:
        return 'Normal_Breast'
    
    # Check for standard cancer cell lines
    for cell_line in cell_lines:
        if cell_line in title:
            return 'Cancer_Cell_Line'
    
    # Check for other cell lines
    if any(marker in title for marker in cell_markers):
        return 'Other_Cell_Line'
    
    # Check for primary breast cancer samples (both BE and AF)
    if title.startswith('BC') and any(x in title for x in ['-BE', '-AF']):
        return 'Primary_BC'
    
    # Everything else goes to other category
    return 'other'

def process_geo_dataset(gse_id, output_file):
    """
    Process a GEO dataset into a CSV file with features and 5 class labels based on sample titles.
    """
    try:
        logging.info(f"Downloading dataset {gse_id}...")
        gse = GEOparse.get_GEO(geo=gse_id, destdir="./")
        
        logging.info("Processing samples...")
        expression_data = pd.DataFrame()
        metadata = pd.DataFrame()
        
        # First pass: collect all titles and their classifications
        for gsm_name, gsm in gse.gsms.items():
            title = gsm.metadata['title'][0]
            category = classify_sample(title)
            metadata.loc[gsm_name, 'title'] = title
            metadata.loc[gsm_name, 'category'] = category
            
            if 'VALUE' in gsm.table.columns:
                expression_data[gsm_name] = gsm.table['VALUE']
        
        # Log the classifications found
        logging.info("\nSample classifications:")
        for category in ['Normal_Breast', 'Cancer_Cell_Line', 'Other_Cell_Line', 
                        'Primary_BC', 'other']:
            samples = metadata[metadata['category'] == category]
            logging.info(f"\n{category} ({len(samples)} samples):")
            for title in sorted(samples['title']):
                logging.info(f"  - {title}")
        
        # Transpose expression data
        expression_data = expression_data.T
        
        # Create labels
        le = LabelEncoder()
        labels = pd.Series(le.fit_transform(metadata['category']), index=metadata.index)
        
        # Log label information
        label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
        logging.info(f"\nLabel mapping: {label_mapping}")
        logging.info(f"Number of classes: {len(le.classes_)}")
        
        # Show distribution of samples
        class_distribution = pd.value_counts(metadata['category'])
        logging.info(f"\nClass distribution:\n{class_distribution}")
        
        # Add labels to expression data
        expression_data['label'] = labels
        expression_data['category'] = metadata['category']
        
        # Save to CSV
        expression_data.to_csv(output_file)
        logging.info(f"\nData saved to {output_file}")
        
        # Save category mapping to a separate file
        mapping_df = pd.DataFrame({
            'Category': le.classes_,
            'Label': range(len(le.classes_)),
            'Sample_Count': class_distribution[le.classes_].values
        })
        mapping_df.to_csv('category_mapping.csv', index=False)
        logging.info("Category mapping saved to category_mapping.csv")
        
        return expression_data.drop(['label', 'category'], axis=1), labels
        
    except Exception as e:
        logging.error(f"Error processing dataset: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    gse_id = "GSE61"
    output_file = "breast-cancer.csv"
    
    try:
        features, labels = process_geo_dataset(gse_id, output_file)
        
        print("\nDataset processed successfully!")
        print(f"Number of samples: {features.shape[0]}")
        print(f"Number of features: {features.shape[1]}")
        
    except Exception as e:
        print(f"Error: {str(e)}")