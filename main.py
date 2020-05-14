#Import libs
import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import sys

'''
#Global variables for the MLP parameters
layers = (35)
function = 'identity'
alpha = 0.99
iterations = 1000
'''

#Read data for training
dataAND = pd.read_csv("data/problemAND.csv", header=None)
dataChar = pd.read_csv("data/caracteres-limpo.csv", header=None)

#Reading/handling data test
x_test = pd.read_csv("data/caracteres-ruido.csv", header=None)
x_test = x_test.loc[:,0:62]

#Creating features and target vectors
#for training
x_train = dataChar.loc[:,0:62]
y_train = dataChar.loc[:,63:63]

#Fit only to the training data
scaler = StandardScaler()
scaler.fit(x_train)

#Transform the data
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#Transform the labels on numbers
y_train = y_train.select_dtypes(include=[object])
le = preprocessing.LabelEncoder()
y_train = y_train.apply(le.fit_transform)

#Create the MLP with the parameters setted at the begining of the program
mlp = MLPClassifier(hidden_layer_sizes=(35), activation='logistic', alpha=0.0001, max_iter=1000, batch_size=21)

#Start the network's training process
mlp.fit(x_train, y_train.values.ravel())

#Declare a variable to store the predictions
prediction = mlp.predict(x_test)

#Print the results and confusion matrix for analysis
print(classification_report(y_train, prediction) + "\n")
print(confusion_matrix(y_train, prediction))


###################################################
#Open file to store parameters of the network
sys.stdout=open("out/parameters.txt","w")

#Print the initial parameters of the MLP
print("Network parameters\n------------------------")
print('Alpha: ', mlp.alpha)
print('Number of hidden layers layers: ', mlp.n_layers_ - 2)
print('Number of neurons on each hidden layer: ', mlp.hidden_layer_sizes)
print('Activation Function: ', mlp.activation)
print('Epochs: ', mlp.n_iter_)

#Close the file
sys.stdout.close()
##################################################

###################################################
#Open file to store networks coefs
sys.stdout=open("out/finalWeights.txt","w")

#Print the initial parameters of the MLP
print("Final Weights:")
print(mlp.coefs_)


#Close the file
sys.stdout.close()
##################################################



