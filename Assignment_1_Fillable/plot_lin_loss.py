import numpy as np
import matplotlib.pyplot as plt
from helpers.metrics import compute_mse, compute_position_error, compute_rotation_error
from datasets import prepare_dataset
from lin_regression_sgd import SGDLinearRegression
from lin_regression_analytic import AnalyticalLinearRegression

def prep_data(dataset, n_samples:int, use_engineered_features=False):
    X_train, X_test, y_train, y_test = dataset
    # Convert to numpy
    X_train = X_train.values[:n_samples, :]
    y_train = y_train.values[:n_samples, :]
    X_test = X_test.values
    y_test = y_test.values
    # print(f"xy train: {np.shape(X_train)} {np.shape(y_train)} xy test: {np.shape(X_test)} {np.shape(y_test)}")

    if use_engineered_features:
        from feature_engineering import engineer_features
        X_train = engineer_features(X_train)
        X_test = engineer_features(X_test) 
    return X_train, X_test, y_train, y_test

def train(model, X_train, y_train):
    # Train model
    if type(model) == SGDLinearRegression:
        model._initialize_parameters(np.size(X_train, axis=1), np.size(y_train, axis=1))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    train_loss = compute_stats(y_pred, y_train)
    return train_loss

def compute_stats(predictions:np.ndarray, targets:np.ndarray):
    """
    Calculate the loss, position error, and rotation error associated with predicted & target (actual) output values.
    """
    loss = compute_mse(predictions, targets)
    pos_error = compute_position_error(predictions, targets)
    rot_error = compute_rotation_error(predictions, targets)
    tot_error = pos_error + rot_error
    return np.array([[loss], [pos_error], [rot_error], [tot_error]])

if __name__ == "__main__":
    # Load data
    dataset = prepare_dataset("data/ur10_dataset.csv")
    
    # Initialize models 
    sgd_model = SGDLinearRegression(learning_rate=0.01)
    an_model = AnalyticalLinearRegression()

    N = [10, 20, 50, 100, 200, 500, 1000] # , 2000, 5000, 10000, 20000, 50000, 80000
    losses     = np.zeros((1,4))
    pos_errors = np.zeros((1,4))
    rot_errors = np.zeros((1,4))
    tot_errors = np.zeros((1,4))

    for n_samples in N:
        X_train, X_test, y_train, y_test = prep_data(dataset, n_samples)

        # Train Analytic & SGD models 
        an_train = train(an_model, X_train, y_train)
        sgd_train = train(sgd_model, X_train, y_train)

        # Predict 
        y_pred = an_model.predict(X_test)
        an_test = compute_stats(y_pred, y_test)
        y_pred = sgd_model.predict(X_test)
        sgd_test = compute_stats(y_pred, y_test)

        # Store metrics for this training size in each row 
        metrics = np.hstack((an_train, sgd_train, an_test, sgd_test))
        losses     = np.vstack((losses,     metrics[0,:]))
        pos_errors = np.vstack((pos_errors, metrics[1,:]))
        rot_errors = np.vstack((rot_errors, metrics[2,:]))
        tot_errors = np.vstack((tot_errors, metrics[3,:]))
        
    fig, axs = plt.subplots(2, 2)
    pos = [x for x in range(len(N))]
    labels = ['Analytic Training', 'SGD Training', 'Analytic Test', 'SGD Test']

    axs[0,0].plot(pos,  losses[1:, :])
    axs[0,0].set_title('Training vs. Test Loss for Least Squares Regression')
    axs[0,0].set_ylabel('Loss')

    axs[0,1].plot(pos,  pos_errors[1:, :])
    axs[0,1].set_title('Training vs. Test Position Error for Least Squares Regression')
    axs[0,1].set_ylabel('Position Error')

    axs[1,0].plot(pos,  rot_errors[1:, :])
    axs[1,0].set_title('Training vs. Test Rotation Error for Least Squares Regression')
    axs[1,0].set_ylabel('Rotation Error')

    axs[1,1].plot(pos,  tot_errors[1:, :])
    axs[1,1].set_title('Training vs. Test Combined Error for Least Squares Regression')
    axs[1,1].set_ylabel('COmbined Position & Rotation Error')

    # Formatting constant for all subplots
    r, c = np.shape(axs)
    for i in range(c): 
        for j in range(r):
            axs[i, j].xaxis.set_ticks(pos)
            axs[i, j].xaxis.set_ticklabels(N)
            axs[i, j].set_xlabel('Number of Training Samples')
            axs[i, j].legend(labels)

    plt.show()


    
    
