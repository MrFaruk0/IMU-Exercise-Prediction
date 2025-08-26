import numpy as np
import os
from iteration_utilities import flatten
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from scipy.stats import mode

class RecGym_DATA(Dataset):
    def __init__(self, root_path, window_size, overlap_size, transform=None):
        """
        root_path : Root directory of the data set
        window_size : Size of the window in seconds
        overlap_size : Size of the overlap in seconds
        transform : Optional transform to be applied on a sample
        """
        self.root_path = root_path
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.transform = transform
        self.used_cols = ["A_x", "A_y", "A_z", "G_x", "G_y", "G_z", "C_1", "Workout", "Subject", "Session"]
        self.label_map = {
            "Adductor": 1, "ArmCurl": 2, "BenchPress": 3, "LegCurl": 4, "LegPress": 5, "Null": 6, 
            "Riding": 7, "RopeSkipping": 8, "Running": 9, "Squat": 10, "StairClimber": 11, "Walking": 12
        }
        self.labelToId = {v: k for v, k in self.label_map.items()}
        self.data_x, self.data_y, self.subjects, self.sessions = self.load_all_the_data()

    def load_all_the_data(self):
        print(" ----------------------- load all the data -------------------")
        df_all = pd.read_csv(os.path.join(self.root_path, "RecGym.csv"))
        df_all = df_all[self.used_cols]
        df_all.dropna(inplace=True)
        df_all["Workout"] = df_all["Workout"].map(self.labelToId)
        data_y = df_all["Workout"].values
        data_x = df_all.drop(columns=["Workout", "Subject", "Session"]).values
        subjects = df_all["Subject"].values
        sessions = df_all["Session"].values

        # Segment the data into windows
        window_size_samples = int(self.window_size * 20)  # 20Hz sampling rate
        overlap_size_samples = int(self.overlap_size * 20)
        step_size = window_size_samples - overlap_size_samples

        segmented_data_x = []
        segmented_data_y = []
        segmented_subjects = []
        segmented_sessions = []

        for start in range(0, len(data_x) - window_size_samples + 1, step_size):
            end = start + window_size_samples
            window_x = data_x[start:end]
            window_y = data_y[start:end]
            window_subjects = subjects[start:end]
            window_sessions = sessions[start:end]
            segmented_data_x.append(window_x)
            segmented_data_y.append(mode(window_y)[0])  # Most common label in the window
            segmented_subjects.append(mode(window_subjects)[0])  # Most common subject in the window
            segmented_sessions.append(mode(window_sessions)[0])  # Most common session in the window


        self.data_x = np.array(segmented_data_x)
        self.data_y = np.array(segmented_data_y)
        self.subjects = np.array(segmented_subjects)
        self.sessions = np.array(segmented_sessions)

        print(self.data_x.shape)
        print(self.data_y.shape)
        return self.data_x, self.data_y, self.subjects, self.sessions

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        sample = self.data_x[idx]
        label = self.data_y[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

def load_data(root_path, batch_size, window_size, overlap_size, training_strategy, fold):
    dataset = RecGym_DATA(root_path, window_size, overlap_size)
    
    if training_strategy == "LOUO":
        if fold < 1 or fold > 10:
            raise ValueError("Fold number for LOUO must be between 1 and 10.")
        train_idx = np.where(dataset.subjects != fold)[0]
        test_idx = np.where(dataset.subjects == fold)[0]
    elif training_strategy == "LOSO":
        if fold < 1 or fold > 5:
            raise ValueError("Fold number for LOSO must be between 1 and 5.")
        train_idx = np.where(dataset.sessions != fold)[0]
        test_idx = np.where(dataset.sessions == fold)[0]
    else:
        raise ValueError("Invalid training strategy. Choose either 'LOUO' or 'LOSO'.")

    train_x, train_y = dataset.data_x[train_idx], dataset.data_y[train_idx]
    test_x, test_y = dataset.data_x[test_idx], dataset.data_y[test_idx]

    train_dataset = CustomDataset(train_x, train_y, dataset.transform)
    test_dataset = CustomDataset(test_x, test_y, dataset.transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

class CustomDataset(Dataset):
    def __init__(self, data_x, data_y, transform=None):
        self.data_x = data_x
        self.data_y = data_y
        self.transform = transform

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        sample = self.data_x[idx]
        label = self.data_y[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

def main():
    root_path = "datasets"
    batch_size = 64
    window_size = 4  # Window size in seconds
    overlap_size = 2  # Overlap size in seconds
    training_strategy = "LOUO"  # Training strategy: "LOUO"(Leave-One-User-Out) or "LOSO"(Leave-One-Session-Out)
    fold = 1  # Fold number for cross-validation  (1-10 for LOUO, 1-5 for LOSO)

    train_loader, test_loader = load_data(root_path, batch_size, window_size, overlap_size, training_strategy, fold)

    for i, (samples, labels) in enumerate(train_loader):
        print(f"Train Batch {i+1}")
        print("Samples:", samples)
        print("Labels:", labels)
        if i == 0:  # Print only the first batch for brevity
            break

    for i, (samples, labels) in enumerate(test_loader):
        print(f"Test Batch {i+1}")
        print("Samples:", samples)
        print("Labels:", labels)
        if i == 0:  # Print only the first batch for brevity
            break

if __name__ == '__main__':
    main()