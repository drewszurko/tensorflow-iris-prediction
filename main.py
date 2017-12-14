from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves.urllib.request import urlopen

import numpy  as np
import tensorflow as tf

# Local/remote location of Iris datasets.
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"


def dataset_availability():
    if not os.path.exists(IRIS_TRAINING):
        raw = urlopen(IRIS_TRAINING_URL).read()
        with open(IRIS_TRAINING, "wb") as f:
            f.write(raw)

    if not os.path.exists(IRIS_TEST):
        raw = urlopen(IRIS_TEST_URL).read()
        with open(IRIS_TEST, "wb") as f:
            f.write(raw)


def load_datasets():
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)
    return training_set, test_set


def build_model():
    model = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                       hidden_units=[10, 20, 10],
                                       n_classes=3,
                                       model_dir="/tmp/iris_model")
    return model


def define_inputs(training_set, test_set):
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_set.data)},
        y=np.array(test_set.target),
        num_epochs=1,
        shuffle=False)
    return train_input_fn, test_input_fn


def generate_new_iris_samples():
    new_iris_samples = np.array(
        [[3.2, 1.6, 1.4, 0.5],
         [4.4, 2.2, 4.0, 2.5],
         [5.4, 2.2, 4.2, 1.2],
         [6.0, 3.5, 5.4, 2.2],
         [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
    return new_iris_samples


def def_new_inputs(new_iris_samples):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": new_iris_samples},
        num_epochs=1,
        shuffle=False)
    return input_fn


if __name__ == '__main__':
    # Check if the Iris datasets are stored locally, otherwise download them.
    dataset_availability()

    # Load our train and test datasets.
    iris_training_set, iris_test_set = load_datasets()

    # Specify that all features have real-value data.
    feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

    # Build a 3 layer Deep Neural Network with 10, 20, 10 units respectively.
    classifier = build_model()

    # Define the training and testing inputs.
    iris_train_input_fn, iris_test_input_fn = define_inputs(iris_training_set, iris_test_set)

    # Train model.
    classifier.train(input_fn=iris_train_input_fn, steps=2000)

    # Evaluate and print model accuracy.
    accuracy_score = classifier.evaluate(input_fn=iris_test_input_fn)["accuracy"]
    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    # Generate five new iris samples.
    generated_iris_samples = generate_new_iris_samples()

    # Define prediction function inputs.
    predict_input_fn = def_new_inputs(generated_iris_samples)

    # Predict and print the species of newly generated Iris samples
    predictions = list(classifier.predict(input_fn=predict_input_fn))
    print("Predicted Iris classes: ", [p["classes"][0] for p in predictions])
