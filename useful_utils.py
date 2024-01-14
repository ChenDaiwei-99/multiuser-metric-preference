import numpy as np

def generate_random_tensor(num_users, num_groups):
    # Create a zero-filled matrix of shape (num_users, num_groups)
    tensor = np.zeros((num_users, num_groups), dtype=int)

    # Randomly choose a column index for each row and set that element to 1
    for i in range(num_users):
        random_col = np.random.randint(num_groups)
        tensor[i, random_col] = 1

    return tensor


if __name__ == '__main__':
    # Example usage
    num_users = 5
    num_groups = 3
    tensor = generate_random_tensor(num_users, num_groups)
    print(tensor)
