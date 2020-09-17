from sklearn.model_selection import train_test_split


# Normalize inputs
def normalize_dataset_inputs(dataset):
    new_dataset = dataset.copy()

    inputs = new_dataset.iloc[:,:-1]
    new_dataset.iloc[:,:-1] = (inputs - inputs.mean()) / inputs.std()

    return new_dataset


# Splits data into training, validation, and test sets
#
# test_size is from entire dataset, validation_size is from training set
def split_data(dataset, validation_size=0.2, test_size=0.1):
    # Split data into training and testing
    training, test = train_test_split(dataset, test_size=test_size)

    # Split training data into training and validation
    training, validation = train_test_split(training, test_size=validation_size)

    return training, validation, test
