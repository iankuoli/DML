import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


class DML:
    ##
    # Choose nuisance Estimator
    # Selects the a machine learning model based on the given Estimator
    # Provides a GridSearch for parameter optimization
    ##
    @staticmethod
    def _get_nusiance_estimator(given_estimator):
        #
        # Case 1 - Dictionary calls
        # --------------------------------------------------------------------------------------------------------------
        if type(given_estimator).__name__ in ["DictionaryLearning", "MiniBatchDictionaryLearning"]:
            nuisanceEstimator = linear_model.Lasso(alpha=0.1)
            # return nuisanceEstimator
            param_grid = [
                {'normalize ': (False, True)}
            ]
            return GridSearchCV(nuisanceEstimator, param_grid, cv=5)

        #
        # Case 2 - Trees
        # --------------------------------------------------------------------------------------------------------------
        elif type(given_estimator).__name__ in ["DecisionTreeClassifier", "DecisionTreeRegressor"]:
            nuisanceEstimator = DecisionTreeRegressor(random_state=1245)
            # return nuisanceEstimator
            param_grid = {'criterion': ("mse", "friedman_mse", "mae"),
                          'splitter': ("best", "random"),
                          "max_depth": [10, 25, 50, 75, 100],
                          "presort": (False, True),
                          "max_features": ("auto", "sqrt", "log2")
                          }
            return GridSearchCV(nuisanceEstimator, param_grid, cv=5)

        #
        # Case 3 - Sparse neural and deep neural nets
        # --------------------------------------------------------------------------------------------------------------
        elif type(given_estimator).__name__ in ["MLPClassifier", "MLPRegressor"]:
            nuisanceEstimator = MLPRegressor(hidden_layer_sizes=(15, 20),
                                             activation='relu',
                                             solver='adam',
                                             learning_rate='adaptive',
                                             max_iter=1000,
                                             learning_rate_init=0.01,
                                             alpha=0.01)
            param_grid = {"hidden_layer_sizes": [25, 50, 100, 200],
                          "activation": ("identity", "logistic", "tanh", "relu"),
                          "solver": ("lbfgs", "sgd", "adam"),
                          "alpha": [0.0001, 0.0005, 0.001, 0.0015, 0.002],
                          "learning_rate": ("constant", "invscaling", "adaptive"),
                          "learning_rate_init": [0.001, 0.002, 0.003],
                          "max_iter": [100, 150, 200, 250]
                          }
            return GridSearchCV(nuisanceEstimator, param_grid, cv=5)

        #
        # Case 4 - least one model mentioned in 1)-3) above calls for the use of an ensemble/aggregated method
        # --------------------------------------------------------------------------------------------------------------
        elif type(given_estimator).__name__ in ["ExtraTreesClassifier", "ExtraTreesRegressor", "IsolationForest"]:
            nuisanceEstimator = DecisionTreeRegressor(random_state=1245)
            # return nuisanceEstimator
            param_grid = {'criterion': ("mse", "friedman_mse", "mae"),
                          'splitter': ("best", "random"),
                          "max_depth": [10, 25, 50, 75, 100],
                          "presort": (False, True),
                          "max_features": ("auto", "sqrt", "log2")
                          }
            return GridSearchCV(nuisanceEstimator, param_grid, cv=5)

    ##
    # Function dml1 implements the double machine learning algorithm 1.
    # The function computes the treatment effect for a given estimator.
    ##
    @staticmethod
    def dml1(data, given_estimator, predictor_variable, treatment_variable, remaining_variables, num_folds,
             theta_opt):
        # 1) Split data
        dataList = np.array_split(data.sample(frac=1), num_folds)
        result = []

        for ii in range(len(dataList)):

            # 2) Get nuisance estimator
            nusianceEstimatorM = DML._get_nusiance_estimator(given_estimator)
            nusianceEstimatorG = DML._get_nusiance_estimator(given_estimator)

            # Prepare D (treatment effect), Y (predictor variable), X (controls)
            mainData = dataList[ii]
            D_main = mainData[treatment_variable]
            Y_main = mainData[predictor_variable]
            X_main = mainData[remaining_variables]

            dataList_ = dataList[:]
            dataList_.pop(ii)
            compData = pd.concat(dataList_)
            D_comp = compData[treatment_variable]
            Y_comp = compData[predictor_variable]
            X_comp = compData[remaining_variables]

            # Compute g as a machine learning estimator, which is trained on the predictor variable
            g_comp = nusianceEstimatorG.fit(X_main, Y_main).predict(X_comp)
            g_main = nusianceEstimatorG.fit(X_comp, Y_comp).predict(X_main)

            # Compute m as a machine learning estimator, which is trained on the treatment variable
            m_comp = nusianceEstimatorM.fit(X_main, D_main).predict(X_comp)
            m_main = nusianceEstimatorM.fit(X_comp, D_comp).predict(X_main)

            # Compute V
            V_comp = D_comp - m_comp
            V_main = D_main - m_main

            # We provide two different theta estimators for computing theta
            if theta_opt == "Opt1":
                theta_comp = DML.theta_estimator1(Y_comp, V_comp, D_comp, g_comp)
                theta_main = DML.theta_estimator1(Y_main, V_main, D_main, g_main)
            else:
                theta_comp = DML.theta_estimator2(Y_comp, V_comp, g_comp)
                theta_main = DML.theta_estimator2(Y_main, V_main, g_main)

            result.append((theta_comp + theta_main) / 2)

        # Aggregate theta
        return np.mean(result)

    ##
    # Theta Estimator one
    ##
    @staticmethod
    def theta_estimator1(Y, V, D, g, eps=1e-12):
        return np.mean((V * (Y - g))) / (np.mean((V * D)) + eps)

    ##
    # Theta Estimator two
    ##
    @staticmethod
    def theta_estimator2(Y, V, g, eps=1e-12):
        return np.mean((V * (Y - g))) / (np.mean((V * V)) + eps)

    ##
    # A naive Approach to estimate the treatment effect based on Random Forest models.
    #
    ##
    @staticmethod
    def compute_naive_approach(data, predictor_variable, remaining_variables, treatment_variable, theta_opt):

        # 1) Random data splitting
        main, auxiliary = np.array_split(data.sample(frac=1), 2)

        # 2) Define and train two Random Forests
        RF_G = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
        RF_M = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)

        g = RF_G.fit(auxiliary[remaining_variables], auxiliary[predictor_variable]).predict(main[remaining_variables])
        m = RF_M.fit(auxiliary[remaining_variables], auxiliary[treatment_variable]).predict(main[remaining_variables])

        # 3) Compute V
        V = main[treatment_variable] - m

        # 4) Compute theta
        if theta_opt == "Opt1":
            theta = DML.theta_estimator1(main[predictor_variable], V, main[treatment_variable], g)
        else:
            theta = DML.theta_estimator2(main[predictor_variable], V, g)

        return theta
