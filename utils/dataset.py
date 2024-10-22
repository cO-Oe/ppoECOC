import os
import sys
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

from typing import (
	List
)
from scipy.io import arff
from pmlb import fetch_data

class customDataset(Dataset):
	def __init__(self, X, y):
		self.X = X
		self.y = y

		self.feature_size = len(X[0])
		self.num_classes = len(set(y.tolist()))
		
	def __len__(self):
		return len(self.y)

	def __getitem__(self, idx):
		return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

class TabData():
	"""
	Class helper to handle datasets, including parsing, reading, and preproccessing
	"""
	def __init__(self, data_path='./dataset'):
		
		# path guard
		if not os.path.exists(data_path):
			print("Folder not found. Please reconfirm your data_path.")
			sys.exit(0)

		self.data_path = data_path
		self.file_list = []

		self.get_files()

		self.X = None 
		self.y = None
		self.feature = None
	
	def __len__(self):
		return len(self.y)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]

	def get_files(self) -> List[str]:
		"""
		Return list of all file's relative path recursively under data_path
		"""
		for file in os.listdir(self.data_path):
			self.file_list.append(os.path.join(self.data_path, file))

		return self.file_list

	def count_class(self, file):
		"""
		Returns the count of unique ys in the dataset, i.e. the number of classes
		"""
		dataset_name = os.path.basename(file)
		# pmlb data
		if os.path.isdir(file): 
			X, y = fetch_data(dataset_name, local_cache_dir=os.path.dirname(file), return_X_y=True)
		# .arff, .data, .csv data
		else:
			X, y = self.tabdata_parser(file)

		unique_values, counts = np.unique(y, return_counts=True)
		return len(unique_values)

	def load_data(self, load_name="", split=0.0, seed=1):
		"""
		Returns processed 2 dataframes: X (features) and y (labels) given dataset name
		if split ratio is given, return stratified train_test_split X_train, X_test, y_train, y_test
		"""

		for file in self.file_list:
			dataset_name = os.path.basename(file)
			if load_name != dataset_name:
				continue

			# pmlb data
			if os.path.isdir(file): 
				X, y = fetch_data(dataset_name, local_cache_dir=os.path.dirname(file), return_X_y=True)
			# .arff, .data, .csv data
			else:
				X, y = self.tabdata_parser(file)

			feature = None
			X, y, feature = self.preprocess(X, y)

			self.X, self.y = X, y
			self.feature = feature


			if split != 0:
				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, stratify=y, random_state=seed)
				# Check for missing classes in the test set
				train_classes = set(y_train)
				test_classes = set(y_test)
				missing_classes = train_classes - test_classes

				if missing_classes:
					print(f"Warning: {len(missing_classes)} classes are missing from the test set. Adding one sample of each.")
					for cls in missing_classes:
						# Find indices of the missing class in the training set
						class_indices = np.where(y_train == cls)[0]
						if len(class_indices) > 0:
							# Select one random sample
							index_to_move = np.random.choice(class_indices)
							
							# Move the sample from train to test
							X_test = np.vstack((X_test, X_train[index_to_move]))
							y_test = np.append(y_test, y_train[index_to_move])
							
							# Remove the sample from train
							X_train = np.delete(X_train, index_to_move, axis=0)
							y_train = np.delete(y_train, index_to_move)
						else:
							print(f"Warning: No samples of class {cls} found in the training set.")

				return X_train, X_test, y_train, y_test
			else:
				return X, y, None, None
	

	def tabdata_parser(self, file) -> tuple[pd.DataFrame, pd.DataFrame]:
		"""
		Input .arff, .csv, .data file path and returns 2 dataframes: X (features) and y (labels)
		"""
		dataset_name = os.path.basename(file)
		
		# arff data
		if file.endswith('.arff'):
			data, meta = arff.loadarff(file)
			df = pd.DataFrame(data)
		
		# tab-delimited data
		elif file.endswith('.tab'):
			df = pd.read_csv(file, sep='\t')

		else:
			if ',' in open(file).read(): # separate by comma (,)
				df = pd.read_csv(file, header=None)
			elif ';' in open(file).read(): # separate by ;
				df = pd.read_csv(file, sep=';') 
			else: # separate by space
				df = pd.read_csv(file, header=None, delim_whitespace=True)
		
		# dataset specific dataframe processing
		new_df = df.copy()
		if dataset_name == "Faults.NNA":
			target = new_df[new_df.columns[27:]].values.argmax(1)
			new_df = new_df.drop(new_df.columns[27:], axis=1)
			new_df[27] = target
		elif dataset_name in ["yeast.data", "ecoli.data", "zoo.data", "hayes-roth.data"]:
			new_df = new_df.drop(new_df.columns[0], axis=1)
		elif dataset_name in ["wine.data", "lymphography.data", "balance-scale.data", "new-thyroid.data", "abalone.data", "segmentation.data"]:
			new_df = new_df[list(new_df.columns[1:]) + [new_df.columns[0]]]
		elif dataset_name == "breast-cancer.csv":
			# fill the nan values with 0
			new_df = new_df.fillna(0)
		elif dataset_name == "golub.csv":
			# encode catagorical features
			pass
		
		# separate label and features
		label = new_df.columns[new_df.shape[1]-1]
		features = new_df.columns[:new_df.shape[1]-1]

		return new_df[features], new_df[label]
	
	def preprocess(self, feature, label, method="standard") -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
		"""
		Perform
		1. label and catagorical feature encoding
		2. data standardize

		And return the preprocessed feature and label dataframes
		"""
		# encode label and categorical features
		encoder = LabelEncoder()
		y = encoder.fit_transform(label)

		# Encode categorical features
		if isinstance(feature, pd.DataFrame):
			for column in feature.select_dtypes(include=['object']):
				feature[column] = encoder.fit_transform(feature[column])
		else:
			feature = pd.DataFrame(feature)  # Ensure X is a DataFrame for encoding

		# standardize data with scikit-learn
		if method == "standard":
			scaler = StandardScaler()
			X = scaler.fit_transform(feature)

		elif method == "minmax":
			scaler = MinMaxScaler()
			X = scaler.fit_transform(feature)
		else:
			X = feature.values

		return X, y, feature
	
	def save_csv(self, name, path='./csv'):
		# feature_df = pd.DataFrame(self.X, columns=self.feature.columns, index=self.feature.index)
		# label_df = pd.DataFrame(self.y, columns=['Target'], index=self.feature.index)

		feature_df = pd.DataFrame(self.X)
		label_df = pd.DataFrame(self.y)

		feature_df.to_csv(f'{path}/{name}_feature.csv')
		label_df.to_csv(f'{path}/{name}_label.csv')