import pickle
import os


def save_checkpoint(data, checkpoint_file):
    with open(checkpoint_file, 'wb') as file:
        pickle.dump(data, file)


def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as file:
            return pickle.load(file)
    return None
