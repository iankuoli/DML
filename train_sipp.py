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
# Data used in the double machine learning paper.

# Data source: SIPP 1991 (Abadie, 2003)
# Description of the data: the sample selection and variable contruction follow

# Abadie, Alberto (2003), "Semiparametric instrumental variable estimation of treatment response
# models," Journal of Econometrics, Elsevier, vol. 113(2), pages 231-263, April.

# The variables in the data set include:

# net_tfa:  net total financial assets
# e401:     = 1 if employer offers 401(k)
# age
# inc:      income
# fsize:    family size
# educ:     years of education
# db:       = 1 if indivuduals has defined benefit pension
# marr:     = 1 if married
# twoearn:  = 1 if two-earner household
# pira:     = 1 if individual participates in IRA plan
# hown      = 1 if home owner
#######################################################################################################################################################
# load data
os.chdir("/home/jupyter/dataset/")
data = pd.read_stata('sipp1991.dta')

predictorVariable = "net_tfa"
treatmentVariable = "e401"
treatmentVariableTree = "p401"
remainingVariables = ["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown"]

##
# Compute the naive approach

resultNaive = []
for ii in range(no):
    theta = dml.compute_naive_approach(data, predictorVariable, remainingVariables, treatmentVariableTree, "Opt1")
    resultNaive.append(theta)
# Statistics
print("Naive, Theta: ", resultNaive)
print("Naive, Std: ", np.std(resultNaive))
print("Naive, mean: ", np.mean(resultNaive))
print("Naive, median: ", np.median(resultNaive))

# Create the plot
plt.hist(resultNaive, 50, facecolor='g', alpha=0.75)
plt.title("Naive - 401(k) plan")
# plt.show()
plt.savefig("Naive - 401(k) plan.png")

##
# Compute DML1 approach

#
# MLPRegressor as given Estimator
# ----------------------------------------------------------------------------------------------------------------------
givenEstimator = MLPRegressor().fit(data[remainingVariables], data.loc[:, predictorVariable])
result = []
for ii in range(no):
    theta = dml.dml1(data, givenEstimator, predictorVariable, treatmentVariable, remainingVariables, 5, "Opt1")
    result.append(theta)
# Statistics
print("MLPRegressor, Theta: ", result)
print("MLPRegressor, Std: ", np.std(result))
print("MLPRegressor, mean: ", np.mean(result))
print("MLPRegressor, median: ", np.median(result))

# Create the plot
plt.hist(result, 50, facecolor='g', alpha=0.75)
plt.title("MLPRegressor, DML1 - 401(k) plan")
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
plt.title("DictionaryLearning, DML1 - 401(k) plan")
plt.show()

#
# DecisionTreeRegressor as given Estimator
# ----------------------------------------------------------------------------------------------------------------------
givenEstimator = DecisionTreeRegressor(random_state=6347).fit(data[remainingVariables], data.loc[:, predictorVariable])
result = []
for ii in range(no):
    theta = dml.dml1(data=data, predictor_variable=predictorVariable, given_estimator=givenEstimator,
                     treatment_variable=treatmentVariableTree, remaining_variables=remainingVariables,
                     theta_opt="Opt1", num_folds=5)
    result.append(theta)
# Statistics
print("DecisionTreeRegressor, Theta: ", result)
print("DecisionTreeRegressor, Std: ", np.std(result))
print("DecisionTreeRegressor, mean: ", np.mean(result))
print("DecisionTreeRegressor, median: ", np.median(result))

# Create the plot
plt.hist(result, 50, facecolor='g', alpha=0.75)
plt.title("DecisionTreeRegressor, DML1 - 401(k) plan")
plt.show()

#
# ExtraTreeClassifier as given Estimator
# ----------------------------------------------------------------------------------------------------------------------
givenEstimator = ExtraTreeClassifier().fit(data[remainingVariables], data.loc[:, predictorVariable])
result = []
for ii in range(no):
    theta = dml.dml1(data=data, predictor_variable=predictorVariable, given_estimator=givenEstimator,
                     treatment_variable=treatmentVariableTree, remaining_variables=remainingVariables,
                     theta_opt="Opt1", num_folds=5)
    result.append(theta)
# Statistics
print("ExtraTreesRegressor, Theta: ", result)
print("ExtraTreesRegressor, Std: ", np.std(result))
print("ExtraTreesRegressor, mean: ", np.mean(result))
print("ExtraTreesRegressor, median: ", np.median(result))

# Create the plot
plt.hist(result, 50, facecolor='g', alpha=0.75)
plt.title("ExtraTreesRegressor, DML1 - 401(k) plan")
plt.show()
