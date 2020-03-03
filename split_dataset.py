from random import gauss, shuffle, seed
seed(42)

n_split = 15 # how many client there is
n_samples = [6000] * 10 # how many samples the classes

classes_splits = []
for n_sample in n_samples:
    mean = n_sample / n_split
    stdev = 1000.0 # Parameterized, 0 = perfectly balanced

    # Get n_split - 1 numbers from a normal distribution with custom mean and stdev
    # n - 1 to easily make sure that the splits add up to n_sample later: last split is computed from the remaining
    raw_class_splits = [gauss(mean, stdev) for _ in range(1, n_split)]
    raw_class_splits = [max(0, min(n_sample, x)) for x in raw_class_splits] # clamp to 0 - n_sample
    raw_class_splits = map(int, raw_class_splits) # integerize

    # Process the raw_class_splits so it adds up to n_sample
    class_splits = []
    remaining_samples = n_sample
    for raw_class_split in raw_class_splits:
        if remaining_samples - raw_class_split < 0:
            raw_class_split = remaining_samples
        class_splits.append(raw_class_split)
        remaining_samples -= raw_class_split
    class_splits.append(remaining_samples)
    shuffle(class_splits) # To reverse the serial iteration effect (skewness on the first index, etc.)

    assert sum(class_splits) == n_sample, "Split incorrect"

    classes_splits.append(class_splits)

# TODO: pick n random datum for each node based on classes_splits
print(classes_splits)
