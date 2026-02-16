import numpy as np
from datasets import prepare_dataset
from lin_regression_sgd import SGDLinearRegression


def dataset_loss(dataset):
    X_train, X_test, y_train, y_test = dataset
    # Convert to numpy
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values

    # Train model
    model = SGDLinearRegression(learning_rate=0.01)
    model._initialize_parameters(np.size(X_train, axis=1), np.size(y_train, axis=1))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    train_loss = model._compute_loss(y_pred, y_train)
    y_pred = model.predict(X_test)
    test_loss = model._compute_loss(y_pred, y_test)
    return train_loss, test_loss

if __name__ == "__main__":
    # Load data
    large_dataset = prepare_dataset("data/ur10_dataset.csv")
    small_dataset = prepare_dataset("data/ur10_linear_dataset.csv")
    
    large_dataset_loss = dataset_loss(large_dataset)
    small_dataset_loss = dataset_loss(small_dataset)



    
    
