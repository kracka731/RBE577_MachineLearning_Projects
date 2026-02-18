# RBE577 Machine Learning for Robotics Project 1 

To set up the environment, run: 

`conda create -n ml_env python=3.8.10 -y`

`conda activate ml_env`

`pip install -r requirements.txt`

To see 3D plots of the end effector predicted and true position and rotation, run the files lin_regression_analytic.py, lin_regression_sgd.py, or nn_regression.py. 

To see plots comparing the loss, position error, rotation error, and combined error, run plot_lin_loss.py or plot_nn_loss.py. 

plot_lin_loss.py compares the metrics of both analytical and SGD linear regression. 
plot_nn_loss.py examines the metrics of the neural network. 

Feature engineering can be enabled or disabled within the lin_regression_analytic.py, lin_regression_sgd.py, or plot_lin_loss.py files by modifying the main executable and setting use_engineered_features to either True or False.