import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeClassifier
from sklearn.decomposition import DictionaryLearning
import matplotlib.pyplot as plt
import sys

from dml import DML
from data_loader import *

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

no = 50  # number of simulations
data_type = 'Insurance'  # [Insurance, Pennsylvania, SIPP1991]
estimate_type = 'MLPRegressor'  # [Naive, MLPRegressor, DictionaryLearning, DecisionTreeRegressor, ExtraTreeClassifier]
theta_opt = "Opt1"
data, var_dict = load_data(data_type)
dml = DML()


def model_estimation(estimator_type, dataset, random_state=6347):
    if estimator_type == 'Naive':
        return None
    return globals()[estimator_type](random_state=random_state).fit(dataset[var_dict['remainingVariables']],
                                                                    dataset.loc[:, var_dict['predictorVariable']])


def theta_estimation(estimator, dataset, estimator_type, theta_type="Opt1"):
    thetas = []
    if estimator_type:
        for ii in range(no):
            theta = dml.dml1(data=dataset,
                             predictor_variable=var_dict['predictorVariable'],
                             given_estimator=estimator,
                             treatment_variable=var_dict['treatmentVariable'],
                             remaining_variables=var_dict['remainingVariables'],
                             theta_opt=theta_type, num_folds=5)
            thetas.append(theta)
    else:
        for ii in range(no):
            theta = dml.compute_naive_approach(data=data,
                                               predictor_variable=var_dict['predictorVariable'],
                                               treatment_variable=var_dict['treatmentVariable'],
                                               remaining_variables=var_dict['remainingVariables'],
                                               theta_opt=theta_type)
            thetas.append(theta)
    return thetas


given_estimator = model_estimation(dataset=data, estimator_type=estimate_type)
result = theta_estimation(estimator=given_estimator, dataset=data,
                          estimator_type=estimate_type, theta_type=theta_opt)

# Compute some statistics
print("{} ==> theta: {} \n "
      "std: {} , mean: {} , median: {}".format(estimate_type, result,
                                               np.std(result), np.mean(result), np.median(result)))

# Create the plot
plt.hist(result, 50, facecolor='g', alpha=0.75)
plt.title("{}, DML{} - {}".format(estimate_type, theta_opt[-1], data_type))
plt.show()
