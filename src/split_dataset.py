from random import gauss, shuffle, seed, sample
seed(42)

'''
indices_per_class: list of indices for each class
n_server: how many data server
stdev: how unequal you want the split to be? must not be less than 0

return indices_per_server
'''

def split_dataset_indices(indices_per_class, n_server, stdev):
    indices_per_server = [[] for _ in range(n_server)] # list of list of indices for each server

    class_counts = [len(indices) for indices in indices_per_class]
    for i in range(len(class_counts)):
        # Determine how many samples in each class each server should have
        class_count = class_counts[i]
        mean = class_count / n_server
        # Get n_server - 1 numbers from a normal distribution with the given mean and stdev
        # n - 1 to easily make sure that the splits add up to class_count later: last split is computed from the remaining
        raw_class_splits = [int(max(0, min(gauss(mean, stdev), class_count))) for _ in range(1, n_server)]

        # Process the raw_class_splits so it adds up to class_count
        class_splits = []
        remaining_samples = class_count
        for raw_class_split in raw_class_splits:
            if remaining_samples - raw_class_split < 0:
                raw_class_split = remaining_samples
            class_splits.append(raw_class_split)
            remaining_samples -= raw_class_split
        class_splits.append(remaining_samples)
        shuffle(class_splits) # To reverse the serial effect (skewness on the first index, etc.)
        assert sum(class_splits) == class_count, "Bad split"

        # Sample indices from indices_per_class based on class_splits
        for j in range(len(class_splits)):
            sampled_indices = sample(indices_per_class[i], class_splits[j])
            indices_per_server[j] += sampled_indices
            indices_per_class[i] = [x for x in indices_per_class[i] if x not in sampled_indices]

    return indices_per_server
