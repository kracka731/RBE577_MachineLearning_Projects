import numpy as np
import matplotlib.pyplot as plt
from helpers.metrics import compute_mse, compute_position_error, compute_rotation_error
from datasets import prepare_dataset
from nn_regression import MLP

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

def train(model, X_train, y_train, lr=0.001, batch_size=32, epochs=100, device="cuda"):
    # Train model

    print("Model type: ",type(model))
    fit_output, accuracy_list, loss_list = model.fit(X_train, y_train, lr=lr, batch_size=batch_size, epochs=epochs, device=device)
    print("Fit done")
    y_pred = model.predict(X_train)
    train_loss = compute_stats(y_pred, y_train)
    return train_loss, accuracy_list, loss_list

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
    model = MLP().to("cuda")

    # epoch_counts = [1, 2, 5, 10, 25, 50, 75, 100]
    epoch_counts = [1, 2]
    learning_rates = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.05]
    losses     = np.zeros((1,2))
    pos_errors = np.zeros((1,2))
    rot_errors = np.zeros((1,2))
    tot_errors = np.zeros((1,2))

    #TODO Set which plot to make.
    epoch_graph = False
    learning_rate_graph = False
    epoch_over_time = True

    
    if epoch_graph:
        for epochs in epoch_counts:
            n_samples = 80000
            X_train, X_test, y_train, y_test = prep_data(dataset, n_samples)

            # Train Multi-Layer Perceptron model
            nn_train, accuracy_list, loss_list  = train(model, X_train, y_train, 0.001, 32, epochs, "cuda")

            # Predict 
            y_pred = model.predict(X_test)
            nn_test = compute_stats(y_pred, y_test)


            # Store metrics for this training size in each row 
            metrics = np.hstack((nn_train, nn_test))
            losses     = np.vstack((losses,     metrics[0,:]))
            pos_errors = np.vstack((pos_errors, metrics[1,:]))
            rot_errors = np.vstack((rot_errors, metrics[2,:]))
            tot_errors = np.vstack((tot_errors, metrics[3,:]))

    elif learning_rate_graph:
        for rate in learning_rates:
            n_samples = 80000
            X_train, X_test, y_train, y_test = prep_data(dataset, n_samples)

            # Train Multi-Layer Perceptron model
            nn_train, accuracy_list, loss_list  = train(model, X_train, y_train, rate, 32, 15, "cuda")


            # Predict 
            y_pred = model.predict(X_test)
            nn_test = compute_stats(y_pred, y_test)


            # Store metrics for this training size in each row 
            metrics = np.hstack((nn_train, nn_test))
            losses     = np.vstack((losses,     metrics[0,:]))
            pos_errors = np.vstack((pos_errors, metrics[1,:]))
            rot_errors = np.vstack((rot_errors, metrics[2,:]))
            tot_errors = np.vstack((tot_errors, metrics[3,:]))
            

    elif epoch_over_time:

        X_train, X_test, y_train, y_test = prep_data(dataset, 80000)
        
        nn_train, accuracy_list, loss_list = train(model, X_train, y_train, 0.001, 32, 50, "cuda")
        
        accuracy_list = np.delete(accuracy_list, 0, 0)
        loss_list = np.delete(loss_list, 0, 0)

        print("accuracy list: ", accuracy_list)
        print("loss_list: ", loss_list)

        y_pred = model.predict(X_test)

        nn_test = compute_stats(y_pred, y_test)


        # Store metrics for this training size in each row 
        metrics = np.hstack((nn_train, nn_test))
        losses     = np.vstack((losses,     metrics[0,:]))
        pos_errors = np.vstack((pos_errors, metrics[1,:]))
        rot_errors = np.vstack((rot_errors, metrics[2,:]))
        tot_errors = np.vstack((tot_errors, metrics[3,:]))

            

    if epoch_graph:
        pos = [x for x in range(len(epoch_counts))]
    elif learning_rate_graph:
        pos = [x for x in range(len(learning_rates))]
    elif epoch_over_time:
        pos = [x for x in range(len(accuracy_list[:,0]))]
    else:
        pos = "Hi"
        print("sup")

    labels = ['MLP Training', 'MLP Test']

    # print("pos size:",np.size(pos))
    # print("pos: ")
    # print("losses size: ",np.size(losses))
    if not epoch_over_time:

        fig, axs = plt.subplots(2, 2)

        axs[0,0].plot(pos,  losses[1:, :])
        axs[0,0].set_title('Training vs. Test Loss for Multi-Layer Perceptron')
        axs[0,0].set_ylabel('Loss')

        axs[0,1].plot(pos,  pos_errors[1:, :])
        axs[0,1].set_title('Training vs. Test Position Error for Multi-Layer Perceptron')
        axs[0,1].set_ylabel('Position Error')

        axs[1,0].plot(pos,  rot_errors[1:, :])
        axs[1,0].set_title('Training vs. Test Rotation Error for Multi-Layer Perceptron')
        axs[1,0].set_ylabel('Rotation Error')

        axs[1,1].plot(pos,  tot_errors[1:, :])
        axs[1,1].set_title('Training vs. Test Combined Error for Multi-Layer Perceptron')
        axs[1,1].set_ylabel('COmbined Position & Rotation Error')

                # Formatting constant for all subplots
        r, c = np.shape(axs)
        for i in range(c): 
            for j in range(r):
                axs[i, j].xaxis.set_ticks(pos)
                if epoch_graph:
                    axs[i, j].xaxis.set_ticklabels(epoch_counts)
                    axs[i, j].set_xlabel('Number of epochs')
                elif learning_rate_graph:
                    axs[i, j].xaxis.set_ticklabels(learning_rates)
                    axs[i, j].set_xlabel('Learning Rate')
                axs[i, j].legend(labels)
    elif epoch_over_time:

        fig, axs = plt.subplots(2,1)

        axs[0].plot(pos, accuracy_list[:,0])
        axs[0].set_title('Training vs. Test Accuracy for Multi-Layer Perceptron')
        axs[0].set_ylabel('Accuracy')

        axs[1].plot(pos, loss_list[:,0])
        axs[1].set_title('Training vs. Test Loss for Multi-Layer Perceptron')
        axs[1].set_ylabel('Loss')

        for i in range(2): 
            print("i: ",i)
            axs[i].xaxis.set_ticks(pos)
            axs[i].xaxis.set_ticklabels(accuracy_list[:,1])
            axs[i].set_xlabel('Epoch')
            axs[i].legend(labels)

    plt.show()


    
    
