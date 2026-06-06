import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size, step_size, ):
        self.data = data
        self.num_entity = len(data)
        self.window_size = window_size
        self.step_size = step_size

        self.windows = self.create_windows()

    def create_windows(self):
        windows = {}
        for key in self.data:
            windows[key] = []
            for entity_data in self.data[key]:
                num_windows = (entity_data.shape[0] - self.window_size) // self.step_size + 1
                for i in range(num_windows):
                    start_idx = i * self.step_size
                    end_idx = start_idx + self.window_size
                    window = entity_data[start_idx:end_idx]
                    windows[key].append(window)
        return windows

    def __getitem__(self, index):
        return {key: self.windows[key][index] for key in self.windows}

    def __len__(self):
        return len(self.windows['data'])
