import torch
import torch.nn as nn

class Normalizer(nn.Module):
    def __init__(self, size, std_epsilon = 1e-8, name = 'Normalizer', device = 'cuda'):
        super(Normalizer, self).__init__()
        self.device = device
        self.name = name
        self._std_epsilon = torch.tensor(std_epsilon, dtype = torch.float32, requires_grad = False, device = device)
        self._acc_count = torch.tensor(0, dtype = torch.float32, requires_grad = False, device = device)
        self._num_accumulations = torch.tensor(0, dtype = torch.float32, requires_grad = False, device = device)
        self._acc_sum = torch.zeros((size), dtype = torch.float32, requires_grad = False, device = device)
        self._acc_sum_squared = torch.zeros((size), dtype = torch.float32, requires_grad = False, device = device)

    def forward(self, batched_data, accumulate = False):
        if accumulate:
            self._accumulate(batched_data)
        return (batched_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data):
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batched_data):
        batch_size, point_size = batched_data.shape[0 : 2]
        data_sum = torch.sum(batched_data, dim = (0, 1))
        squared_data_sum = torch.sum(batched_data ** 2, dim = (0, 1))

        self._acc_sum += data_sum
        self._acc_sum_squared += squared_data_sum
        self._acc_count += batch_size * point_size
        self._num_accumulations += batch_size

    def _mean(self):
        safe_count = torch.maximum(self._acc_count, torch.tensor(1.0, dtype = torch.float32, device = self._acc_count.device))
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = torch.maximum(self._acc_count, torch.tensor(1.0, dtype = torch.float32, device = self._acc_count.device))
        std = torch.sqrt(torch.clamp(self._acc_sum_squared / safe_count - self._mean() ** 2, min = 0.))
        return torch.maximum(std, self._std_epsilon)

    def get_variable(self):
        
        dict = {
            '_std_epsilon': self._std_epsilon,
            '_acc_count': self._acc_count,
            '_num_accumulations': self._num_accumulations,
            '_acc_sum': self._acc_sum,
            '_acc_sum_squared': self._acc_sum_squared,
            'name': self.name
        }

        return dict
    
    def save_variable(self, path):
        dict = self.get_variable()
        torch.save(dict, path)
    
    def load_variable(self, path):
        dict = torch.load(path, map_location = self.device, weights_only = True)
        self._std_epsilon = dict['_std_epsilon']
        self._acc_count = dict['_acc_count']
        self._num_accumulations = dict['_num_accumulations']
        self._acc_sum = dict['_acc_sum']
        self._acc_sum_squared = dict['_acc_sum_squared']
        self.name = dict['name']