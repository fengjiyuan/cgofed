

# CGoFed: Constrained Gradient Optimization Strategy for Federated Class Incremental Learning

This repository contains the implementation of the CGoFed: Constrained Gradient Optimization Strategy for Federated Class Incremental Learning method to prevent catastrophic forgetting in continual federated learning.

## Requirements

Make sure you have the following installed:

- Python 3.9
- PyTorch
- Other dependencies: `numpy`, `scipy`, `matplotlib`, `sklearn`

You can install the required libraries via:

```bash
pip install -r requirements.txt
nohup python -u main_cifar.py > result.txt 2>&1 &
```

## Running the Code

### Command Structure

The following command is used to run the model training and evaluation process with specific parameters:

```bash
nohup python -u main_cifar.py --batch_size_train 64 --batch_size_test 64 --l_epochs 5 --g_epochs 20 --alpha 1 --beta 1 --tau 0.02 \
     --seed 2023 --pc_valid 0.05 --device 0 --lr 0.01 --momentum 0.9 --lr_min 1e-5 --lr_patience 6 \
     --lr_factor 2 --task_num 10 --clients_num 10 --selected_clients 2 > result.log 2>&1 &
```

## Results

Results will be saved in the log files specified by the command. You can inspect these logs for details on training and evaluation performance.

## Contributing

Feel free to submit issues or pull requests for any improvements, bug fixes, or suggestions.

## License

This project is licensed under the MIT License.

---

Let me know if you need any further modifications!
