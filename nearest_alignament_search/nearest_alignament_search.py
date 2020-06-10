from data.data_split import read_data_by_serialized_random_split

if __name__ == '__main__':
    train_assays, val_assays, test_assays = read_data_by_serialized_random_split()