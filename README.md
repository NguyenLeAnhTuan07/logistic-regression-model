This project implements Logistic Regression for binary classification using two optimization approaches:

### 1. Gradient Descent

The model parameters are optimized iteratively using Batch Gradient Descent.
At each iteration, the algorithm computes gradients using the entire dataset and updates the weights to minimize the Binary Cross-Entropy (Log Loss).

Gradient Descent is useful when:

The dataset size is small to medium

Stable and smooth convergence is preferred

Full-batch updates are computationally feasible
---

### 2. Stochastic Gradient Descent (SGD)

The model also implements Stochastic Gradient Descent (SGD), where parameters are updated using one training sample (or a small batch) at each iteration instead of the entire dataset.

Unlike standard Gradient Descent, SGD performs more frequent updates, which can significantly speed up learning in practice.

Stochastic Gradient Descent is useful when:

The dataset is very large

Faster updates are required

Memory usage must be reduced

Online learning (real-time updates) is needed

Although SGD introduces more noise in the optimization process, it often converges faster and may generalize better in real-world scenarios.

---
The dataset used in this project is obtained from an external source.
You can replace the dataset to generate new weights and make predictions by using a dataset in CSV format.
Rename the dataset to data.csv and place it in the data directory.
Then, update the feature_names.txt file to match the feature names in the dataset.
Note: the last feature name must be the output feature, which is the value you want to predict.

Additionally, note that there is an encoding.py file in the encoding directory.
If your dataset contains string values (for example, a car dataset), you need to encode them into numerical values such as 0, 1, 2, 3, ..., depending on how many unique categories the data contains.

🔗 Dataset link:  
tieuduong : i don't know :))

cotsong: https://www.kaggle.com/datasets/uciml/biomechanical-features-of-orthopedic-patients/data

## Create virtual environment (recommended)
python -m venv venv


Linux / macOS: source venv/bin/activate

Windows: venv\Scripts\activate


pip install numpy pandas

Run the program
Use this when you want to train the model and compute the weights: python train.py

Use this when you want to make predictions using the trained weights: python predict.py
If the learned weights are not satisfactory, you can modify the training hyperparameters in the fit function inside the modelpre directory. Specifically, you may tune the learning rate, number of epochs, and patience to improve the resulting weights.
(This applies only to the Gradient Descent optimization method.)

---
### Conclusion

This project is a simple implementation of Logistic Regression aimed at understanding the core concepts behind classification models.

Users can:

- Modify the dataset

- Retrain the model

- Tune hyperparameters

- Perform binary classification on their own data

Thank you for checking out this project.
Have a great day! ☀️

Author: Nguyen Le Anh Tuan
