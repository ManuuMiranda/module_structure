import torch
import logging


__all__ = ['TensorCalculator']


class TensorCalculator:

    def __int__(self):
        pass

    @staticmethod
    def tensor_zeros(dim_x, dim_y, dim_z):
        zeros = torch.zeros(dim_x, dim_y, dim_z)
        return zeros

    @staticmethod
    def tensor_ones(dim_x, dim_y, dim_z):
        ones = torch.ones([dim_x, dim_y, dim_z])
        return ones

    @staticmethod
    def tensor_random(dim_x, dim_y, dim_z):
        rands = torch.rand(dim_x, dim_y, dim_z)
        return rands

    @staticmethod
    def tensor_sum(tensor_1, tensor_2):
        if not isinstance(tensor_1, torch.Tensor) or not isinstance(tensor_1, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor")

        if tensor_1.size() == tensor_2.size():
            return tensor_1 + tensor_2
        elif tensor_1.size() != tensor_2.size():
            print('\nCareful! The tensors do not have the same size even if you can add them\n')
            return tensor_1 + tensor_2
        else:
            raise ValueError('An unexpected error happened, try converting your elements into tensors or adding the 2 parameters')

    @staticmethod
    def tensor_diff(tensor_1, tensor_2):
        if not isinstance(tensor_1, torch.Tensor) or not isinstance(tensor_1, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor")

        if tensor_1.size() == tensor_2.size():
            return tensor_1 - tensor_2
        elif tensor_1.size() != tensor_2.size():
            print('\nCareful! The tensors do not have the same size even if you can add them\n')
            return tensor_1 - tensor_2
        else:
            raise ValueError('An unexpected error happened, try converting your elements into tensors or adding the 2 parameters')

    @staticmethod
    def tensor_mult(tensor_1, tensor_2):
        if not isinstance(tensor_1, torch.Tensor) or not isinstance(tensor_1, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor")

        if tensor_1.size() == tensor_2.size():
            return tensor_1 * tensor_2
        elif tensor_1.size() != tensor_2.size():
            print(f'\nCareful! The tensors do not have the same size even if you can add them\n')
            return tensor_1 * tensor_2
        else:
            raise ValueError('An unexpected error happened, try converting your elements into tensors or adding the 2 parameters')

    @staticmethod
    def tensor_mean(tensor_1):
        if not isinstance(tensor_1, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor")

        return torch.mean(tensor_1)

    @staticmethod
    def tensor_stdev(tensor_1):
        if not isinstance(tensor_1, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor")

        return torch.std(tensor_1)

    @staticmethod
    def tensor_min(tensor_1):
        if not isinstance(tensor_1, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor")

        return torch.min(tensor_1)

    @staticmethod
    def tensor_max(tensor_1):
        if not isinstance(tensor_1, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor")

        return torch.max(tensor_1)


TensorCalculator.tensor_mult(tensor_1=torch.rand(2,3,4), tensor_2=torch.rand(2,1,4))


# Example 1: Create tensors filled with zeros
zeros_tensor_1 = TensorCalculator.tensor_zeros(2, 3, 4)
zeros_tensor_2 = TensorCalculator.tensor_zeros(2, 2, 2)

# Example 2: Create tensors filled with ones
ones_tensor_1 = TensorCalculator.tensor_ones(3, 2, 1)
ones_tensor_2 = TensorCalculator.tensor_ones(2, 2, 2)

# Example 3: Create random tensors
random_tensor_1 = TensorCalculator.tensor_random(2, 3, 4)
random_tensor_2 = TensorCalculator.tensor_random(1, 1, 1)

# Example 4: Calculate the sum of two tensors
sum_result_1 = TensorCalculator.tensor_sum(zeros_tensor_2, ones_tensor_2)
sum_result_2 = TensorCalculator.tensor_sum(random_tensor_1, random_tensor_2)

# Example 5: Calculate the difference of two tensors
diff_result_1 = TensorCalculator.tensor_diff(ones_tensor_2, zeros_tensor_2)
diff_result_2 = TensorCalculator.tensor_diff(random_tensor_1, random_tensor_2)

# Example 6: Calculate the element-wise multiplication of two tensors
mult_result_1 = TensorCalculator.tensor_mult(ones_tensor_2, zeros_tensor_2)
mult_result_2 = TensorCalculator.tensor_mult(random_tensor_1, random_tensor_2)

# Example 7: Calculate the mean of a tensor
mean_result_1 = TensorCalculator.tensor_mean(random_tensor_1)
mean_result_2 = TensorCalculator.tensor_mean(random_tensor_2)

# Example 8: Calculate the standard deviation of a tensor
stdev_result_1 = TensorCalculator.tensor_stdev(random_tensor_1)
stdev_result_2 = TensorCalculator.tensor_stdev(random_tensor_2)

# Example 9: Find the minimum value in a tensor
min_result_1 = TensorCalculator.tensor_min(random_tensor_1)
min_result_2 = TensorCalculator.tensor_min(random_tensor_2)

# Example 10: Find the maximum value in a tensor
max_result_1 = TensorCalculator.tensor_max(random_tensor_1)
max_result_2 = TensorCalculator.tensor_max(random_tensor_2)