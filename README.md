

# CGoFed: Constrained Gradient Optimization Strategy for Federated Class Incremental Learning

This repository contains the implementation of the CGoFed: Constrained Gradient Optimization Strategy for Federated Class Incremental Learning method to prevent catastrophic forgetting in continual federated learning.

## Requirements

Make sure you have the following installed:

- Python 3.8
- PyTorch
- Other dependencies: `numpy`, `scipy`, `matplotlib`, `sklearn`

You can install the required libraries via:

```bash
pip install -r requirements.txt
```

## Running the Code

### Command Structure

The following command is used to run the model training and evaluation process with specific parameters:

```bash
nohup python -u main.py --batch_size_train 64 --batch_size_test 64 --l_epochs 5 --g_epochs 20 --alpha 1 --beta 1 --tau 0.02 \
     --seed 2023 --pc_valid 0.05 --device 0 --lr 0.01 --momentum 0.9 --lr_min 1e-5 --lr_patience 6 \
     --lr_factor 2 --task_num 10 --clients_num 10 --selected_clients 2 --test 0 > test.log 2>&1 &
```

### Parameters

Here is an explanation of the key parameters used in the command:

- `batch_size_train`: Batch size for training the model (64).
- `batch_size_test`: Batch size for testing the model (64).
- `l_epochs`: Number of local epochs to train each client (5).
- `g_epochs`: Number of global epochs for training (20).
- `alpha`: Weight for the loss function term related to the generative replay (1).
- `beta`: Weight for the loss function term related to the federated learning task (1).
- `tau`: Regularization factor for the replay buffer (0.02).
- `seed`: Random seed for reproducibility (2023).
- `pc_valid`: Percentage of validation data (0.05).
- `device`: The device ID to use for computation (0, assuming GPU 0).
- `lr`: Learning rate for the optimizer (0.01).
- `momentum`: Momentum for the optimizer (0.9).
- `lr_min`: Minimum learning rate (1e-5).
- `lr_patience`: Number of epochs with no improvement before reducing the learning rate (6).
- `lr_factor`: Factor by which the learning rate is reduced (2).
- `task_num`: Number of tasks to be used in federated learning (10).
- `clients_num`: Number of clients in the federated learning setup (10).
- `selected_clients`: Number of clients selected per round for training (2).
- `test`: Set to `0` for training, set to `1` for testing.

### Logging

The output will be saved in a file named `test.log`. Use this log file to monitor the progress and results of the training process.

### Running the Model in the Background

The `nohup` command is used to run the script in the background, which will allow it to continue running even after the terminal session is closed. The output and error logs will be redirected to the `test.log` file.

To monitor the progress:

```bash
tail -f test.log
```

### Testing the Model

To test the trained model after completing the training, set the `--test 1` flag when running the script:

```bash
nohup python -u main.py --test 1 > test.log 2>&1 &
```

## Results

Results will be saved in the log files specified by the command. You can inspect these logs for details on training and evaluation performance.

## Contributing

Feel free to submit issues or pull requests for any improvements, bug fixes, or suggestions.

## License

This project is licensed under the MIT License.

---

Let me know if you need any further modifications!
