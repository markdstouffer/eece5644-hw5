import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

r_minus_1 = 2
r_plus_1 = 4
sigma = 1 

# generate iid samples for training and testing
def generate_data(r, sigma, n_samples):
    theta = np.random.uniform(-np.pi, np.pi, n_samples)
    noise = np.random.normal(0, sigma, (n_samples, 2))
    return r * np.column_stack((np.cos(theta), np.sin(theta))) + noise

# data
X_train_minus_1 = generate_data(r_minus_1, sigma, 1000)
X_train_plus_1 = generate_data(r_plus_1, sigma, 1000)
X_train = np.vstack((X_train_minus_1, X_train_plus_1))
y_train = np.array([-1]*1000 + [1]*1000)
X_test_minus_1 = generate_data(r_minus_1, sigma, 10000)
X_test_plus_1 = generate_data(r_plus_1, sigma, 10000)
X_test = np.vstack((X_test_minus_1, X_test_plus_1))
y_test = np.array([-1]*10000 + [1]*10000)

# Best SVM params search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001]
}

grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=10)
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_svm = grid_search.best_estimator_

svm_test_predictions = best_svm.predict(X_test)
svm_test_accuracy = accuracy_score(y_test, svm_test_predictions)

print(f"Best SVM Parameters: {best_parameters}")
print(f"SVM Test Accuracy: {svm_test_accuracy}")


# Best MLP params search
param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['tanh', 'relu'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01],
}

grid_search_mlp = GridSearchCV(MLPClassifier(max_iter=1000), param_grid_mlp, cv=10, n_jobs=-1)
grid_search_mlp.fit(X_train, y_train)
best_parameters_mlp = grid_search_mlp.best_params_
best_mlp = grid_search_mlp.best_estimator_

mlp_test_predictions = best_mlp.predict(X_test)
mlp_test_accuracy = accuracy_score(y_test, mlp_test_predictions)

print(f"Best MLP Parameters: {best_parameters_mlp}")
print(f"MLP Test Accuracy: {mlp_test_accuracy}")


# split the training data for cross-validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42)

# SVM
svm = SVC(kernel='rbf', C=100, gamma=0.01)
svm.fit(X_train_split, y_train_split)

# MLP
mlp = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', max_iter=1000)
mlp.fit(X_train_split, y_train_split)

# 10-fold cross-validation for SVM & MLP
svm_cv_scores = cross_val_score(svm, X_train, y_train, cv=10)
mlp_cv_scores = cross_val_score(mlp, X_train, y_train, cv=10)

svm_cv_accuracy = svm_cv_scores.mean()
svm_cv_std = svm_cv_scores.std()
mlp_cv_accuracy = mlp_cv_scores.mean()
mlp_cv_std = mlp_cv_scores.std()

print(f"SVM 10-fold CV Accuracy: {svm_cv_accuracy} (+/- {svm_cv_std * 2})")
print(f"MLP 10-fold CV Accuracy: {mlp_cv_accuracy} (+/- {mlp_cv_std * 2})")


# predictions for validation set
svm_predictions = svm.predict(X_val_split)
mlp_predictions = mlp.predict(X_val_split)

svm_accuracy = accuracy_score(y_val_split, svm_predictions)
mlp_accuracy = accuracy_score(y_val_split, mlp_predictions)
svm_loss = zero_one_loss(y_val_split, svm_predictions)
mlp_loss = zero_one_loss(y_val_split, mlp_predictions)

def plot_decision_boundary(classifier, X, y, title):
    plt.figure(figsize=(10, 8))

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.2  # step size
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr, edgecolors='k', marker='o')
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

# Plot decision boundaries
plot_decision_boundary(svm, X_train, y_train, 'SVM Decision Boundary')
plot_decision_boundary(mlp, X_train, y_train, 'MLP Decision Boundary')
plt.show()

print(svm_accuracy, svm_loss, mlp_accuracy, mlp_loss)
