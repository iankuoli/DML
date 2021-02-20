import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeClassifier
from sklearn.decomposition import DictionaryLearning
import matplotlib.pyplot as plt
import sys

from dml import DML

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

no = 50  # number of simulations
dml = DML()


#######################################################################################################################################################
# Data from Kaggle: Medicl Cost Personal Dataset
# Link: https://www.kaggle.com/mirichoi0218/insurance

# The 6 variables (columns) of the datafile are described as follows:

# age
# sex
# bmi:         body mass index
# children:    number of childrens
# smoker:      1, if the person is smoking
# region:      Denotes one og four regions
# charges

#######################################################################################################################################################

# Load Data
os.chdir("/home/jupyter/dataset/")
data = pd.read_csv('insurance.csv', sep=",")

# Map String to Numbers
data['sex'] = data['sex'].map({'female': 1, 'male': 0})
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
data['region'] = data['region'].map({'northeast': 1, 'northwest': 2, 'southwest': 3, 'southeast': 4})

# Define Variables
predictorVariable = "charges"
treatmentVariable = "smoker"
remainingVariables = ["age", "sex", "bmi", "children", "region"]

##
# Compute the Naive Approach
resultNaive = []
for ii in range(no):
    theta = dml.compute_naive_approach(data=data, predictor_variable=predictorVariable,
                                       treatment_variable=treatmentVariable, remaining_variables=remainingVariables,
                                       theta_opt="Opt1")
    resultNaive.append(theta)

# Compute some statistics
print("Naive, Theta: ", resultNaive)
print("Naive, Std: ", np.std(resultNaive))
print("Naive, mean: ", np.mean(resultNaive))
print("Naive, median: ", np.median(resultNaive))

# Create the plot
plt.hist(resultNaive, 50, facecolor='g', alpha=0.75)
plt.title("Naive - Insurance")
plt.show()

##
# Compute the DML1 algorithm

#
# MLPRegressor as given Estimator
# ----------------------------------------------------------------------------------------------------------------------
givenEstimator = MLPRegressor().fit(data[remainingVariables], data.loc[:, predictorVariable])
result = []
for ii in range(no):
    theta = dml.dml1(data=data, predictor_variable=predictorVariable, given_estimator=givenEstimator,
                     treatment_variable=treatmentVariable, remaining_variables=remainingVariables,
                     theta_opt="Opt1", num_folds=5)
    result.append(theta)
# Statistics
print("MLPRegressor, Theta: ", result)
print("MLPRegressor, Std: ", np.std(result))
print("MLPRegressor, mean: ", np.mean(result))
print("MLPRegressor, median: ", np.median(result))

# Create the plot
plt.hist(result, 50, facecolor='g', alpha=0.75)
plt.title("MLPRegressor, DML1 - Insurance")
plt.show()

#
# DictionaryLearning as given Estimator
# ----------------------------------------------------------------------------------------------------------------------
givenEstimator = DictionaryLearning().fit(data[remainingVariables], data.loc[:, predictorVariable])
result = []
for ii in range(no):
    theta = dml.dml1(data=data, predictor_variable=predictorVariable, given_estimator=givenEstimator,
                     treatment_variable=treatmentVariable, remaining_variables=remainingVariables,
                     theta_opt="Opt1", num_folds=5)
    result.append(theta)
# Statistics
print("DictionaryLearning, Theta: ", result)
print("DictionaryLearning, Std: ", np.std(result))
print("DictionaryLearning, mean: ", np.mean(result))
print("DictionaryLearning, median: ", np.median(result))

# Create the plot
plt.hist(result, 50, facecolor='g', alpha=0.75)
plt.title("DictionaryLearning, DML1 - Insurance")
plt.show()

#
# DecisionTreeRegressor as given estimator
# ----------------------------------------------------------------------------------------------------------------------
givenEstimator = DecisionTreeRegressor(random_state=6347).fit(data[remainingVariables], data.loc[:, predictorVariable])
result = []
for ii in range(no):
    theta = dml.dml1(data=data, predictor_variable=predictorVariable, given_estimator=givenEstimator,
                     treatment_variable=treatmentVariable, remaining_variables=remainingVariables,
                     theta_opt="Opt1", num_folds=5)
    result.append(theta)
# Statistics
print("DecisionTreeRegressor, Theta: ", result)
print("DecisionTreeRegressor, Std: ", np.std(result))
print("DecisionTreeRegressor, mean: ", np.mean(result))
print("DecisionTreeRegressor, median: ", np.median(result))

# Create the plot
plt.hist(result, 50, facecolor='g', alpha=0.75)
plt.title("DecisionTreeRegressor, DML1 - Insurance")
plt.show()

#
# ExtraTreeClassifier as given Estimator
# ----------------------------------------------------------------------------------------------------------------------
givenEstimator = ExtraTreeClassifier().fit(data[remainingVariables], data.loc[:, predictorVariable])
result = []
for ii in range(no):
    theta = dml.dml1(data=data, predictor_variable=predictorVariable, given_estimator=givenEstimator,
                     treatment_variable=treatmentVariable, remaining_variables=remainingVariables,
                     theta_opt="Opt1", num_folds=5)
    result.append(theta)
# Statistics
print("ExtraTreesRegressor, Theta: ", result)
print("ExtraTreesRegressor, Std: ", np.std(result))
print("ExtraTreesRegressor, mean: ", np.mean(result))
print("ExtraTreesRegressor, median: ", np.median(result))

# Create the plot
plt.hist(result, 50, facecolor='g', alpha=0.75)
plt.title("ExtraTreesRegressor, DML1 - Insurance")
plt.show()
