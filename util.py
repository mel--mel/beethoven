import numpy as np
import random

def separate_dataset_io(dataset, is_training=False):
    """
    Args:
        dataset_file: csv dataset
                      n_samples X n_features + 1, if `is_training`
                      n_samples X n_features, if not.
    Returns:
        inputs: n_samples X n_features
        outputs: n_categories X n_samples. If not `is_training`, None.
    """
    inputs = dataset
    outputs = None

    if is_training:
        # last column is an integer representing a specific genre
        output_genres = inputs[:, -1]
        n_genres = len(set(output_genres))

        outputs = np.zeros((n_genres, len(inputs)))
        for nth_sample, genre in enumerate(output_genres):
            outputs[genre, nth_sample] = 1

        # remove genre, leave features intact
        inputs = inputs[:, :-1]

    return (inputs, outputs)

def split_dataset(dataset_file, percentage=0.8):
    inputs = np.genfromtxt(dataset_file, delimiter=',', skip_header=1)
    genres = np.unique(inputs[:, -1])

    training_dataset = []
    test_dataset = []

    genres_songs = {}
    for genre in genres:
        genres_songs[genre] = inputs[inputs[:, -1] == genre]
        random.shuffle(genres_songs[genre])
        split_point = len(genres_songs[genre]) * percentage

        training_dataset.extend(genres_songs[genre][:split_point])
        test_dataset.extend(genres_songs[genre][split_point:])

    training_dataset = np.array(training_dataset)
    test_dataset = np.array(test_dataset)

    return (training_dataset, test_dataset)

def evaluate(dataset_output, results):
    correct = 0
    for nth_sample in range(dataset_output.shape[1]):
        max_position = results[:, nth_sample].argmax()
        if dataset_output[max_position, nth_sample] == 1:
            correct += 1

    percentage = float(correct) / dataset_output.shape[1] * 100
    return round(percentage, 2)
