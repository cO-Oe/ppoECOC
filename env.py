import numpy as np
import gymnasium as gym

from collections import Counter
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import fbeta_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from imblearn.metrics import geometric_mean_score

from utils.dataset import TabData
from decode import decode_ecoc

DEBUG_PRINT = False

class ECOCEnv(gym.Env):
    def __init__(self, num_classes, max_columns, dataset_name, seed, decode_method):
        super(ECOCEnv, self).__init__()
        self.num_classes = num_classes
        self.max_columns = max_columns

        # define action space for reinforcement learning
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(num_classes,), dtype=np.int8)

        confusion_matrix_size = num_classes * num_classes

        # define state space (observation space) for reinforcement learning
        self.observation_space = gym.spaces.Box(
            low=-1, 
            high=1, 
            shape=(confusion_matrix_size + num_classes * max_columns,), 
            dtype=np.float32
        )
        
        # retrieve the dataset from dataset helper class
        self.td = TabData('./dataset')
        self.dataset_name = dataset_name
        print(f"dataset inited with seed {seed}")

        # train-validation-test split for datasets
        if self.dataset_name in ["golub.csv", "breast-cancer.csv", "SRBCT.tab.csv"]:
            self.X_train, self.X_test, self.y_train, self.y_test = self.td.load_data(self.dataset_name, split=0.25, seed=seed)
            self.X_vali = self.X_train
            self.y_vali = self.y_train
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = self.td.load_data(self.dataset_name, split=0.25, seed=seed)
            self.X_train, self.X_vali, self.y_train, self.y_vali = self.validation_split(val_ratio=1/3, seed=seed)


        # print detailed information for each split
        self.print_split_info("Training", self.X_train, self.y_train)
        self.print_split_info("Validation", self.X_vali, self.y_vali)
        self.print_split_info("Test", self.X_test, self.y_test)

        # calculate and print overall ratios
        total_samples = len(self.y_train) + len(self.y_vali) + len(self.y_test)
        print("\nOverall Split Ratios:")
        print(f"Training: {len(self.y_train) / total_samples:.2%}")
        print(f"Validation: {len(self.y_vali) / total_samples:.2%}")
        print(f"Test: {len(self.y_test) / total_samples:.2%}")

        # list to store the SVM classifiers
        self.classifiers = []

        # confusion matrix and metric recorder
        self.confusion_matrix = np.zeros((num_classes, num_classes))
        self.previous_metrics = {'f1': 0, 'weighted_f1': 0, 'f2': 0, 'weighted_f2': 0, 'g_mean': 0, 'mauc': 0, 'conf_matrix': None,
                            'f1_vali': 0, 'weighted_f1_vali': 0, 'f2_vali': 0, 'weighted_f2_vali': 0, 'g_mean_vali': 0, 'mauc_vali': 0, 'conf_matrix_vali': None,
                            'f1_train': 0, 'weighted_f1_train': 0, 'f2_train': 0, 'weighted_f2_train': 0, 'g_mean_train': 0, 'mauc_train': 0, 'conf_matrix_train': None}

        # parameters
        self.sample_thres = 512
        self.decode_method = decode_method

        # for early termination
        self.plateau_counter = 0
        self.max_plateau = 3  # TBD should add to args maximum number of steps without improvement

        # best metric recorder
        self.best_score = float('-inf')
        self.best_metrics = None
        self.best_matrix = None
        self.best_classifiers = None
      
        self.reset(seed=seed)
    
    def validation_split(self, val_ratio=1/3, seed=None):
        # performs train-validation split for the training data
        class_counts = Counter(self.y_train)
        min_class_count = min(class_counts.values())
        min_required = max(3, int(1 / (1 - val_ratio)) + 1)  # Ensure at least 3 samples per class

        # duplicate sample if a class of data if not enough to perform the split
        if min_class_count < min_required:
            print(f"Insufficient samples for some classes. Duplicating underrepresented classes.")
            X_augmented, y_augmented = [], []
            
            for class_label, count in class_counts.items():
                class_mask = (self.y_train == class_label)
                X_class = self.X_train[class_mask]
                y_class = self.y_train[class_mask]
                
                if count < min_required:
                    multiplier = int(np.ceil(min_required / count))
                    X_class = np.tile(X_class, (multiplier, 1))
                    y_class = np.tile(y_class, multiplier)
                
                X_augmented.append(X_class)
                y_augmented.append(y_class)
            
            X_augmented = np.vstack(X_augmented)
            y_augmented = np.concatenate(y_augmented)
        else:
            X_augmented, y_augmented = self.X_train, self.y_train

        # perform the split
        X_train, X_val, y_train, y_val = train_test_split(
            X_augmented, y_augmented, 
            test_size=val_ratio, 
            stratify=y_augmented, 
            random_state=seed
        )

        return X_train, X_val, y_train, y_val
    
    def print_split_info(self, split_name, X, y):
        # information helper function
        total_samples = len(y)
        class_distribution = Counter(y)
        print(f"\n{split_name} Split:")
        print(f"Total samples: {total_samples}")
        print("Class distribution:")
        for class_label, count in class_distribution.items():
            ratio = count / total_samples
            print(f"  Class {class_label}: {count} samples ({ratio:.2%})")

    def reset(self, seed=None, options=None):
        # implementation of reset function in standard reinforcement learning environments
        super().reset(seed=seed)
        self.matrix = np.zeros((self.num_classes, self.max_columns), dtype=np.int8)
        self.current_column = 0
        self.classifiers = []
        
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.previous_metrics = {'f1': 0, 'weighted_f1': 0, 'f2': 0, 'weighted_f2': 0, 'g_mean': 0, 'mauc': 0, 'conf_matrix': None,
                                 'f1_vali': 0, 'weighted_f1_vali': 0, 'f2_vali': 0, 'weighted_f2_vali': 0, 'g_mean_vali': 0, 'mauc_vali': 0, 'conf_matrix_vali': None,
                                 'f1_train': 0, 'weighted_f1_train': 0, 'f2_train': 0, 'weighted_f2_train': 0, 'g_mean_train': 0, 'mauc_train': 0, 'conf_matrix_train': None}

        # for early termination
        self.plateau_counter = 0
        self.max_plateau = 3  # maximum number of steps without improvement
        
        # best metric
        self.best_score = float('-inf')
        self.best_metrics = None
        self.best_matrix = None
        self.best_classifiers = None

        return self._get_state(), {}

    def step(self, action):
        # implementation of step function in standard reinforcement learning environments

        # terminate if maximum column is reached
        if self.current_column >= self.max_columns:
            return self._get_state(), 0, True, False, {}

        # update the ECOC column with action from agent
        self.matrix[:, self.current_column] = action - 1
        self.current_column += 1

        """
        Reward Column Penalty
        """ 
        # check the number of unique non-zero classes in the new column
        unique_classes = np.unique(self.matrix[:, self.current_column-1][self.matrix[:, self.current_column-1] != 0])
        if len(unique_classes) <= 1:
            # if there are 1 or fewer unique non-zero classes, terminate the episode with a negative reward -1
            if self.best_score <= 0.0: # No any results
                return self._get_state(), -1.0, True, False, {"termination_reason": "insufficient_classes"}
            else:
                return self._get_state(), -1.0, True, False, {"termination_reason": "insufficient_classes", 
                                                                                "best_score": self.best_score, 
                                                                                "best_metrics": self.best_metrics, 
                                                                                "best_matrix": self.best_matrix, 
                                                                                "best_classifiers": self.best_classifiers
                                                                            }
            
        # train a base classifier (SVM) for the new column
        self._train_base_classifier()
        
        if DEBUG_PRINT:
            print(f"action: {action-1}")

        # evaluate current ECOC performance
        current_metrics = self._evaluate_ecoc_performance()
        current_score = current_metrics['f2_vali'] + current_metrics['g_mean_vali'] + current_metrics['mauc_vali']

        reward = self._calculate_reward(current_metrics)

        # best metric recorder
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_metrics = current_metrics
            self.best_matrix = self.matrix.copy()
            self.best_classifiers = self.classifiers.copy()

            self.plateau_counter = 0
        else:
            self.plateau_counter += 1

        # check for episodic early termination
        done = (self.current_column == self.max_columns) or (self.plateau_counter >= self.max_plateau) 

        if done:
            # If we're done, return the best matrix we've seen
            info = {"best_score": self.best_score, "best_metrics": self.best_metrics, "best_matrix": self.best_matrix, "best_classifiers": self.best_classifiers}
            reward += (self.best_metrics['f2_vali'] + self.best_metrics['g_mean_vali'] + self.best_metrics['mauc_vali'])
        else:
            info = {}

        return self._get_state(), reward, done, False, info

    def sample_data(self, X, y, threshold=512, random_state=None):  # None random_state for randomness
        # sample data helper function for training

        if len(X) <= threshold:
            return X, y

        unique_classes, class_counts = np.unique(y, return_counts=True)
        rng = np.random.default_rng(random_state)

        if np.any(class_counts == 1):
            # handle the case where some classes have only one sample
            X_sampled = []
            y_sampled = []
            for label in unique_classes:
                class_indices = np.where(y == label)[0]
                n_samples = min(int(threshold * class_counts[unique_classes == label] / len(y)), len(class_indices))
                sampled_indices = rng.choice(class_indices, size=max(n_samples, 1), replace=False)
                X_sampled.append(X[sampled_indices])
                y_sampled.extend([label] * len(sampled_indices))
            
            X_sampled = np.vstack(X_sampled)
            y_sampled = np.array(y_sampled)
        else:
            # use StratifiedShuffleSplit requires all classes have at least 2 samples
            sss = StratifiedShuffleSplit(n_splits=1, train_size=threshold, random_state=random_state)
            for train_index, _ in sss.split(X, y):
                X_sampled = X[train_index]
                y_sampled = y[train_index]

            # handle classes not present in the sampled dataset due to class ratio to small regard to sampling ratio
            for label in unique_classes:
                if label not in y_sampled:
                    label_indices = np.where(y == label)[0]
                    random_idx = rng.choice(label_indices)
                    X_sampled = np.vstack((X_sampled, X[random_idx]))
                    y_sampled = np.append(y_sampled, label)

        return X_sampled, y_sampled

    def _train_base_classifier(self):
        # Create binary labels for the current column, ignoring classes labeled as 0
        column = self.matrix[:, self.current_column-1]
        mask = column != 0

        binary_labels = np.zeros_like(self.y_train)
        for i, label in enumerate(self.y_train):
            if column[label] != 0:
                binary_labels[i] = 1 if column[label] == 1 else -1
            else:
                binary_labels[i] = 0  # This sample will be ignored in training
        
        # Filter out samples corresponding to classes labeled as 0
        train_mask = binary_labels != 0
        X_train_filtered = self.X_train[train_mask]
        binary_labels_filtered = binary_labels[train_mask]

        
        unique_classes = np.unique(binary_labels_filtered)
        if len(unique_classes) >= 2: # Note: this condition should always hold since case <= 1 will be terminated with negative reward in advance
            # Sample the data
            X_train_sampled, binary_labels_sampled = self.sample_data(X_train_filtered, binary_labels_filtered, threshold=512) 

            # Train SVM classifier
            clf = SVC(probability=True)
            clf.fit(X_train_sampled, binary_labels_sampled)
            self.classifiers.append((clf, mask))

    def _calculate_reward(self, current_metrics):
        # calculate improvements for all metrics
        f1_train_improvement = current_metrics['f1_train'] - self.previous_metrics.get('f1_train', 0)
        weighted_f1_train_improvement = current_metrics['weighted_f1_train'] - self.previous_metrics.get('weighted_f1_train', 0)
        f2_train_improvement = current_metrics['f2_train'] - self.previous_metrics.get('f2_train', 0)
        weighted_f2_train_improvement = current_metrics['weighted_f2_train'] - self.previous_metrics.get('weighted_f2_train', 0)
        g_mean_train_improvement = current_metrics['g_mean_train'] - self.previous_metrics.get('g_mean_train', 0)
        mauc_train_improvement = current_metrics['mauc_train'] - self.previous_metrics.get('mauc_train', 0)

        # Update previous metrics
        self.previous_metrics = current_metrics

        reward = self._calculate_confusion_reward(self.confusion_matrix, current_metrics['conf_matrix_train'], self.matrix[:, self.current_column-1])

        # update confusion matrix
        self.confusion_matrix = current_metrics['conf_matrix_train']

        if DEBUG_PRINT:
            print(f"Total Reward: {reward:.4f}")
            print("-" * 50)  # Separator for readability

        return reward
    
    def _calculate_confusion_reward(self, prev_conf_matrix, new_conf_matrix, new_column):
        # calculate the reward based on confusion matrix improvement
        if np.sum(prev_conf_matrix) == 0 or np.sum(new_conf_matrix) == 0:
            return 0

        # create copies before modifying
        prev_conf_copy = prev_conf_matrix.copy()
        new_conf_copy = new_conf_matrix.copy()

        # remove diagonal elements (self-confusion)
        np.fill_diagonal(prev_conf_copy, 0)
        np.fill_diagonal(new_conf_copy, 0)
        
        # calculate overall confusion reduction
        overall_reduction = np.sum(prev_conf_copy) - np.sum(new_conf_copy)
        
        # calculate targeted confusion reduction based on the new column
        targeted_reduction = 0
        for i in range(len(new_column)):
            for j in range(i+1, len(new_column)):
                if new_column[i] != new_column[j] and new_column[i] != 0 and new_column[j] != 0:
                    targeted_reduction += prev_conf_copy[i, j] - new_conf_copy[i, j]
                    targeted_reduction += prev_conf_copy[j, i] - new_conf_copy[j, i]
        
        # combine metrics for final confusion-based reward
        confusion_reward = (overall_reduction * 0.5 + 
                            targeted_reduction * 0.5)
        
        return confusion_reward
    
    def _predict(self, X, y):
        # output helper function for the classifiers

        predictions = []
        probabilities = []
        decision_functions = []

        for clf, mask in self.classifiers:
            pred = clf.predict(X)
            predictions.append(pred)

            if self.decode_method == "prob_loss_based":
                prob = clf.predict_proba(X)[:, 1]
                prob[~mask[y]] = 0.5 
                probabilities.append(prob)

            if self.decode_method == "loss_based":
                df = clf.decision_function(X)
                df[~mask[y]] = 0
                decision_functions.append(df)
            
        predictions = np.array(predictions).T
        probabilities = np.array(probabilities).T
        decision_functions = np.array(decision_functions).T

        return predictions, probabilities, decision_functions
    
    def _calculate_imbalance_ratios(self):
        # helper function to get class imbalance ratios
        class_counts = np.bincount(self.y_train)
        max_count = np.max(class_counts)
        imbalance_ratios = max_count / class_counts

        return imbalance_ratios

    def _evaluate_ecoc_performance(self):
        # test, vali, and train results
        test_predictions, test_probabilities, test_decision_functions = self._predict(self.X_test, self.y_test)
        vali_predictions, vali_probabilities, vali_decision_functions = self._predict(self.X_vali, self.y_vali)
        train_predictions, train_probabilities, train_decision_functions = self._predict(self.X_train, self.y_train)

        current_codewords = self.matrix[:, :self.current_column]

        # Decode ECOC predictions
        if self.decode_method == "hamming":
            decoded_test_predictions = decode_ecoc(current_codewords, test_predictions, method='hamming')
            decoded_vali_predictions = decode_ecoc(current_codewords, vali_predictions, method='hamming')
            decoded_train_predictions = decode_ecoc(current_codewords, train_predictions, method='hamming')
        elif self.decode_method == "euclidean":
            decoded_test_predictions = decode_ecoc(current_codewords, test_predictions, method='euclidean')
            decoded_vali_predictions = decode_ecoc(current_codewords, vali_predictions, method='euclidean')
            decoded_train_predictions = decode_ecoc(current_codewords, train_predictions, method='euclidean')
        elif self.decode_method == "prob_loss_based":
            decoded_test_predictions = decode_ecoc(current_codewords, test_probabilities, method='prob_loss_based')
            decoded_vali_predictions = decode_ecoc(current_codewords, vali_probabilities, method='prob_loss_based')
            decoded_train_predictions = decode_ecoc(current_codewords, train_probabilities, method='prob_loss_based')
        elif self.decode_method == "loss_based":
            decoded_test_predictions = decode_ecoc(current_codewords, test_decision_functions, method='loss_based')
            decoded_vali_predictions = decode_ecoc(current_codewords, vali_decision_functions, method='loss_based')
            decoded_train_predictions = decode_ecoc(current_codewords, train_decision_functions, method='loss_based')

        # calculate imbalance ratios
        imbalance_ratios = self._calculate_imbalance_ratios()

        # calculate metrics for test set
        f1 = fbeta_score(self.y_test, decoded_test_predictions, beta=1, average=None)
        f2 = fbeta_score(self.y_test, decoded_test_predictions, beta=2, average=None)
        weighted_f1 = np.average(f1, weights=imbalance_ratios)
        weighted_f2 = np.average(f2, weights=imbalance_ratios)
        g_mean = geometric_mean_score(self.y_test, decoded_test_predictions, correction=0.001)
        mauc = self._calculate_mauc(decoded_test_predictions, self.y_test)
        conf_matrix = confusion_matrix(self.y_test, decoded_test_predictions, normalize="true")

         # calculate metrics for validation set
        f1_vali = fbeta_score(self.y_vali, decoded_vali_predictions, beta=1, average=None)
        f2_vali = fbeta_score(self.y_vali, decoded_vali_predictions, beta=2, average=None)
        weighted_f1_vali = np.average(f1_vali, weights=imbalance_ratios)
        weighted_f2_vali = np.average(f2_vali, weights=imbalance_ratios)
        g_mean_vali = geometric_mean_score(self.y_vali, decoded_vali_predictions, correction=0.001)
        mauc_vali = self._calculate_mauc(decoded_vali_predictions, self.y_vali)
        conf_matrix_vali = confusion_matrix(self.y_vali, decoded_vali_predictions, normalize="true")

        # calculate metrics for train set
        f1_train = fbeta_score(self.y_train, decoded_train_predictions, beta=1, average=None)
        f2_train = fbeta_score(self.y_train, decoded_train_predictions, beta=2, average=None)
        weighted_f1_train = np.average(f1_train, weights=imbalance_ratios)
        weighted_f2_train = np.average(f2_train, weights=imbalance_ratios)
        g_mean_train = geometric_mean_score(self.y_train, decoded_train_predictions, correction=0.001)
        mauc_train = self._calculate_mauc(decoded_train_predictions, self.y_train)
        conf_matrix_train = confusion_matrix(self.y_train, decoded_train_predictions, normalize="true")

        if DEBUG_PRINT:
            print(f"\tCurrent column: {self.current_column}")
            print(f"\tCurrent matrix: {self.matrix}")

            print(f"\t\tTest metrics:")
            print(f"\t\t\tg_mean : {g_mean}")
            print(f"\t\t\tf1-score : {f1.mean() * 100:.3f}%  ({' '.join([f'{f1[i] * 100:2.3f}' for i in range(len(f1))])})")
            print(f"\t\t\tf2-score : {f2.mean() * 100:.3f}%  ({' '.join([f'{f2[i] * 100:2.3f}' for i in range(len(f2))])})")
            print(f"\t\t\tweighted f1-score : {weighted_f1})")
            print(f"\t\t\tweighted f2-score : {weighted_f2})")
            print(f"\t\t\tauc : {mauc}")
            print(f"\t\t\tconf_matrix : {conf_matrix}")

            print(f"\t\tValidation metrics:")
            print(f"\t\t\tg_mean : {g_mean_vali}")
            print(f"\t\t\tf1-score : {f1_vali.mean() * 100:.3f}%  ({' '.join([f'{f1_vali[i] * 100:2.3f}' for i in range(len(f1_vali))])})")
            print(f"\t\t\tf2-score : {f2_vali.mean() * 100:.3f}%  ({' '.join([f'{f2_vali[i] * 100:2.3f}' for i in range(len(f2_vali))])})")
            print(f"\t\t\tweighted f1-score : {weighted_f1_vali})")
            print(f"\t\t\tweighted f2-score : {weighted_f2_vali})")
            print(f"\t\t\tauc : {mauc_vali}")
            print(f"\t\t\tconf_matrix : {conf_matrix_vali}")

            
            print(f"\t\tTrain metrics:")
            print(f"\t\t\tg_mean : {g_mean_train}")
            print(f"\t\t\tf1-score : {f1_train.mean() * 100:.3f}%  ({' '.join([f'{f1_train[i] * 100:2.3f}' for i in range(len(f1_train))])})")
            print(f"\t\t\tf2-score : {f2_train.mean() * 100:.3f}%  ({' '.join([f'{f2_train[i] * 100:2.3f}' for i in range(len(f2_train))])})")
            print(f"\t\t\tweighted f1-score : {weighted_f1_train})")
            print(f"\t\t\tweighted f2-score : {weighted_f2_train})")
            print(f"\t\t\tauc : {mauc_train}")
            print(f"\t\t\tconf_matrix : {conf_matrix_train}")

            print(f"\t\tImbalance ratios : {' '.join([f'{ratio:.2f}' for ratio in imbalance_ratios])}")
            

        # return the metrics 
        return {
            'f1': f1.mean(), 
            'f2': f2.mean(), 
            'weighted_f1': weighted_f1,
            'weighted_f2': weighted_f2,
            'g_mean': g_mean, 
            'mauc': mauc,
            'conf_matrix': conf_matrix,
            'f1_vali': f1_vali.mean(),
            'f2_vali': f2_vali.mean(),
            'weighted_f1_vali': weighted_f1_vali,
            'weighted_f2_vali': weighted_f2_vali,
            'g_mean_vali': g_mean_vali,
            'mauc_vali': mauc_vali,
            'conf_matrix_vali': conf_matrix_vali,
            'f1_train': f1_train.mean(),
            'f2_train': f2_train.mean(),
            'weighted_f1_train': weighted_f1_train,
            'weighted_f2_train': weighted_f2_train,
            'g_mean_train': g_mean_train,
            'mauc_train': mauc_train,
            'conf_matrix_train': conf_matrix_train,
        }

    def _calculate_mauc(self, y_pred, y_ref):
        # helper function to calculate MAUC
        y_true = y_ref
        unique_classes = np.unique(y_true)
        
        # create binary predictions for each class
        y_pred_bin = [[1 if c == p else 0 for c in unique_classes] for p in y_pred]
        
        try:
            mauc = roc_auc_score(y_true, y_pred_bin, multi_class="ovo")
            return mauc
        except ValueError as e:
            print(f"ValueError in AUC calculation: {e}")
            return 0.5  # return neutral AUC if calculation fails

    def render(self):
        print("Current ECOC Matrix:")
        print(self.matrix[:, :self.current_column])

    def _get_state(self):
        # returns current state

        # flatten the confusion matrix and concatenate with the flattened ECOC matrix
        flattened_confusion_matrix = self.confusion_matrix.flatten()
        flattened_ecoc_matrix = self.matrix.flatten()

        return np.concatenate([flattened_confusion_matrix, flattened_ecoc_matrix])
    