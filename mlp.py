import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import sys

x_train = pd.read_csv(r'data/x_train_scaled.csv')
y_train = pd.read_csv(r'data/y_train.csv')

y_train = y_train['Risk']

sys.stdout = open("errors/errors.txt", "w")
mlp = MLPClassifier(hidden_layer_sizes=80, alpha=0.5,learning_rate_init=0.5, activation='logistic', max_iter=5000, verbose=True, batch_size=655, tol=0, n_iter_no_change=5000)
mlp.fit(x_train, y_train.values.ravel())
sys.stdout.close()