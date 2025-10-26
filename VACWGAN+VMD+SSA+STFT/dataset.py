from torch.utils.data import Dataset
import pandas as pd


class BearingDataset(Dataset):
    """
    读取轴承故障检测数据集
    """

    def __init__(self, f_paths, labels, transform=None):
        self.f_paths = f_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.f_paths)

    def __getitem__(self, index):
        path = self.f_paths[index]
        label = self.labels[index]

        # 读取样本，只要第二列
        data_series = pd.read_excel(path, header=None, usecols=[1]).squeeze("columns")
        signal = data_series.to_numpy()

        if self.transform:
            signal = self.transform(signal)

        return signal, label
