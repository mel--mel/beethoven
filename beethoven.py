import util
from rbfn import RBFN

if __name__ == '__main__':
    # dataset_file = raw_input('Training dataset location: ')
    dataset_file = 'dataset.csv'
    training_dataset, test_dataset = util.split_dataset(dataset_file)

    training_inputs, training_outputs = util.separate_dataset_io(
        training_dataset, is_training=True)
    rbfn = RBFN(n_centroids=8)
    rbfn.train(training_inputs, training_outputs)

    test_inputs, test_outputs = util.separate_dataset_io(
        test_dataset, is_training=True)
    results = rbfn.predict(test_inputs)
    print util.evaluate(test_outputs, results)
