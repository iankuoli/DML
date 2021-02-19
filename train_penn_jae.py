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

# Data source: Yannis Bilias, "Sequential Testing of Duration Data: The Case of Pennsylvania 'Reemployment Bonus' Experiment",
# Journal of Applied Econometrics, Vol. 15, No. 6, 2000, pp. 575-594

# Description of the data set taken from Bilias (2000):

# The 23 variables (columns) of the datafile utilized in the article may be described as follows:

# abdt:       chronological time of enrollment of each claimant in the Pennsylvania reemployment bonus experiment.
# tg:         indicates the treatment group (bonus amount - qualification period) of each claimant.
# inuidur1:   a measure of length (in weeks) of the first spell ofunemployment
# inuidur2:   a second measure for the length (in weeks) of
# female:     dummy variable; it indicates if the claimant's sex is female (=1) or male (=0).
# black:      dummy variable; it  indicates a person of black race (=1).
# hispanic:   dummy variable; it  indicates a person of hispanic race (=1).
# othrace:    dummy variable; it  indicates a non-white, non-black, not-hispanic person (=1).
# dep:        the number of dependents of each claimant;
# q1-q6:      six dummy variables indicating the quarter of experiment  during which each claimant enrolled.
# recall:     takes the value of 1 if the claimant answered ``yes'' when was asked if he/she had any expectation to be recalled
# agelt35:    takes the value of 1 if the claimant's age is less  than 35 and 0 otherwise.
# agegt54:    takes the value of 1 if the claimant's age is more than 54 and 0 otherwise.
# durable:    it takes the value of 1 if the occupation  of the claimant was in the sector of durable manufacturing and 0 otherwise.
# nondurable: it takes the value of 1 if the occupation of the claimant was in the sector of nondurable manufacturing and 0 otherwise.
# lusd:       it takes the value of 1 if the claimant filed  in Coatesville, Reading, or Lancaster and 0 otherwise.
#             These three sites were considered to be located in areas characterized by low unemployment rate and short duration of unemployment.
# husd:       it takes the value of 1 if the claimant filed in Lewistown, Pittston, or Scranton and 0 otherwise.
#             These three sites were considered to be located in areas characterized by high unemployment rate and short duration of unemployment.
# muld:       it takes the value of 1 if the claimant filed in Philadelphia-North, Philadelphia-Uptown, McKeesport, Erie, or Butler and 0 otherwise.
#             These three sites were considered to be located in areas characterized by moderate unemployment rate and long duration of unemployment."
#######################################################################################################################################################

# Load data
os.chdir("C:\\Users\\Christopher\\Desktop\\APA")
data = pd.read_csv('penn_jae.csv', sep=";")

# Define Variables
predictorVariable = "inuidur1"
treatmentVariable = "tg"
remainingVariables = ["female", "black", "othrace", "dep", "q2", "q3", "q4", "q5", "q6",
                      "agelt35", "agegt54", "durable", "lusd", "husd"]

##
# Compute the naive approach
resultNaive = []
for ii in range(no):
    theta = dml.compute_naive_approach(data=data, predictor_variable=predictorVariable,
                                       treatment_variable=treatmentVariable, remaining_variables=remainingVariables,
                                       theta_opt="Opt1")
    resultNaive.append(theta)
# statistics
print("Naive, Theta: ", resultNaive)
print("Naive, Std: ", np.std(resultNaive))
print("Naive, mean: ", np.mean(resultNaive))
print("Naive, median: ", np.median(resultNaive))

# Create the plot
plt.hist(resultNaive, 50, facecolor='g', alpha=0.75)
plt.title("Naive - Pennsylvania")
plt.show()

##
# Compute the DML1 algorithm

# MLPRegressor as given Estimator
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
plt.title("MLPRegressor, DML1 - Pennsylvania")
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
plt.title("DictionaryLearning, DML1 - Pennsylvania")
plt.show()

#
# DecisionTreeRegressor as given Estimator
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
plt.title("DecisionTreeRegressor, DML1 - Pennsylvania")
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
plt.title("ExtraTreesRegressor, DML1 - Pennsylvania")
plt.show()
