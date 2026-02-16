import numpy as np
import matplotlib.pyplot as plt
from helpers.metrics import compute_mse
from datasets import prepare_dataset
from lin_regression_sgd import SGDLinearRegression
from lin_regression_analytic import AnalyticalLinearRegression

def compute_loss(y_pred, y_true):
    return compute_mse(y_pred, y_true) 

def dataset_loss(dataset, model):
    X_train, X_test, y_train, y_test = dataset
    # Convert to numpy
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values

    # Train model
    if type(model) == SGDLinearRegression:
        model._initialize_parameters(np.size(X_train, axis=1), np.size(y_train, axis=1))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    train_loss = compute_loss(y_pred, y_train)
    y_pred = model.predict(X_test)
    test_loss = compute_loss(y_pred, y_test)
    return train_loss, test_loss


if __name__ == "__main__":
    # Load data
    large_dataset = prepare_dataset("data/ur10_dataset.csv")
    small_dataset = prepare_dataset("data/ur10_linear_dataset.csv")
    
    # Initialize models 
    sgd_model = SGDLinearRegression(learning_rate=0.01)
    an_model = AnalyticalLinearRegression()

    # Analytic 
    an_large_dataset_loss = dataset_loss(large_dataset, an_model)
    an_small_dataset_loss = dataset_loss(small_dataset, an_model)

    # SGD 
    sgd_large_dataset_loss = dataset_loss(large_dataset, sgd_model)
    sgd_small_dataset_loss = dataset_loss(small_dataset, sgd_model)

    print(f"losses: {an_large_dataset_loss, an_small_dataset_loss, sgd_large_dataset_loss, sgd_small_dataset_loss}")

    fig, axs = plt.subplots(1, 2)    
    labels = ['Large Dataset Train', 'Large Dataset Test', 'Small Dataset Train', 'Small Dataset Test']
    axs[0].bar(labels, [an_large_dataset_loss[0], an_large_dataset_loss[1], an_small_dataset_loss[0], an_small_dataset_loss[1]])
    axs[0].set_title('Analytic Linear Regression Loss')
    axs[0].set_ylabel('Loss')

    axs[1].bar(labels, [sgd_large_dataset_loss[0], sgd_large_dataset_loss[1], sgd_small_dataset_loss[0], sgd_small_dataset_loss[1]])
    axs[1].set_title('SGD Linear Regression Loss')
    axs[1].set_ylabel('Loss')

    plt.show()


    
    
