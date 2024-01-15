import numpy as np
import random

def generate_random_tensor(num_users, num_groups):
    # Create a zero-filled matrix of shape (num_users, num_groups)
    tensor = np.zeros((num_users, num_groups), dtype=int)

    # Randomly choose a column index for each row and set that element to 1
    for i in range(num_users):
        random_col = np.random.randint(num_groups)
        tensor[i, random_col] = 1

    return tensor

def generate_random_tensor_even_allocation(num_users, num_groups):
    if num_users < num_groups:
        return "Number of users should be greater than or equal to the number of groups."

    # Determine the minimum number of users per group
    min_users_per_group = num_users // num_groups

    # Determine the number of remaining users
    remaining_users = num_users % num_groups

    # Create a list to store the number of users allocated to each group
    group_allocation = [min_users_per_group] * num_groups

    # Distribute the remaining users evenly among the groups
    for i in range(remaining_users):
        group_allocation[i] += 1

    # Create an empty matrix of shape (num_users, num_groups)
    tensor = np.zeros((num_users, num_groups), dtype=int)

    # Assign users to groups based on the calculated allocation
    current_user_index = 0
    for group_index, num_users_in_group in enumerate(group_allocation):
        for _ in range(num_users_in_group):
            tensor[current_user_index, group_index] = 1
            current_user_index += 1

    return tensor




def randomly_and_evenly_assign_items(N, m):
    if N < m:
        return "N should be greater than or equal to m for random and even distribution."

    # Create a list of N items
    items = list(range(N))

    # Shuffle the list randomly
    random.shuffle(items)

    # Calculate the minimum number of items in each group
    min_items_per_group = N // m

    # Calculate the number of remaining items
    remaining_items = N % m

    # Initialize an empty list to store the groups
    groups = [[] for _ in range(m)]
    
    # Initialize a list to store group indicators for each item
    group_indicators = [None] * N

    # Assign items to groups
    item_count = 0
    for i in range(m):
        group_indicator = i  # Assign a group indicator to the current group
        # Add the minimum number of items to the current group
        groups[i].extend(items[item_count:item_count + min_items_per_group])
        for item in items[item_count:item_count + min_items_per_group]:
            group_indicators[item] = group_indicator
        item_count += min_items_per_group

        # If there are remaining items, distribute them among the groups
        if remaining_items > 0:
            group_indicator = i  # Assign a group indicator to the current group
            groups[i].append(items[item_count])
            group_indicators[items[item_count]] = group_indicator
            item_count += 1
            remaining_items -= 1

    return group_indicators

if __name__ == '__main__':
    # Example usage
    num_users = 5
    num_groups = 3
    tensor = generate_random_tensor(num_users, num_groups)
    print(tensor)

    # Example usage
    N = 10  # Total number of items
    m = 2   # Number of groups
    result = randomly_and_evenly_assign_items(N, m)
    print(result)