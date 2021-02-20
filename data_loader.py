import os
import pandas as pd


root_path = "/home/jupyter/dataset/"


def load_data(data_name):
    if data_name == 'Insurance':
        return load_insurance()
    elif data_name == 'Pennsylvania':
        return load_pennjae()
    elif data_name == 'SIPP1991':
        return load_sipp()
    else:
        exit("Unidentified dataset name!!!")


def load_insurance():
    ####################################################################################################################
    # Data from Kaggle: Medicl Cost Personal Dataset
    # Link: https://www.kaggle.com/mirichoi0218/insurance
    #
    # The 6 variables (columns) of the datafile are described as follows:
    #
    # age
    # sex
    # bmi:         body mass index
    # children:    number of childrens
    # smoker:      1, if the person is smoking
    # region:      Denotes one og four regions
    # charges
    #
    ####################################################################################################################
    data = pd.read_csv(os.path.join(root_path, 'insurance.csv'), sep=",")

    # Map String to Numbers
    data['sex'] = data['sex'].map({'female': 1, 'male': 0})
    data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
    data['region'] = data['region'].map({'northeast': 1, 'northwest': 2, 'southwest': 3, 'southeast': 4})

    # Define Variables
    ret_dict = {'predictorVariable': "charges",
                'treatmentVariable': "smoker",
                'remainingVariables': ["age", "sex", "bmi", "children", "region"]
                }
    return data, ret_dict


def load_pennjae():
    ####################################################################################################################
    # Data used in the double machine learning paper.
    #
    # Yannis Bilias, "Sequential Testing of Duration Data: The Case of Pennsylvania 'Reemployment Bonus' Experiment",
    # Journal of Applied Econometrics, Vol. 15, No. 6, 2000, pp. 575-594
    #
    # Description of the data set taken from Bilias (2000):
    #
    # The 23 variables (columns) of the datafile utilized in the article may be described as follows:
    #
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
    # recall:     takes the value of 1 if the claimant answered ``yes'' when was asked if he/she had any expectation
    #             to be recalled
    # agelt35:    takes the value of 1 if the claimant's age is less  than 35 and 0 otherwise.
    # agegt54:    takes the value of 1 if the claimant's age is more than 54 and 0 otherwise.
    # durable:    it takes the value of 1 if the occupation  of the claimant was in the sector of durable manufacturing
    #             and 0 otherwise.
    # nondurable: it takes the value of 1 if the occupation of the claimant was in the sector of non-durable
    #             manufacturing and 0 otherwise.
    # lusd:       it takes the value of 1 if the claimant filed  in Coatesville, Reading, or Lancaster and 0 otherwise.
    #             These three sites were considered to be located in areas characterized by low unemployment rate and
    #             short duration of unemployment.
    # husd:       it takes the value of 1 if the claimant filed in Lewistown, Pittston, or Scranton and 0 otherwise.
    #             These three sites were considered to be located in areas characterized by high unemployment rate and
    #             short duration of unemployment.
    # muld:       it takes the value of 1 if the claimant filed in Philadelphia-North, Philadelphia-Uptown, McKeesport,
    #             Erie, or Butler and 0 otherwise.
    #             These three sites were considered to be located in areas characterized by moderate unemployment rate
    #             and long duration of unemployment."
    ####################################################################################################################

    # Define Variables
    data = pd.read_csv(os.path.join(root_path, 'penn_jae.csv'), sep=";")

    # Define Variables
    ret_dict = {'predictorVariable': "inuidur1",
                'treatmentVariable': "tg",
                'remainingVariables': ["female", "black", "othrace", "dep", "q2", "q3", "q4", "q5", "q6",
                                       "agelt35", "agegt54", "durable", "lusd", "husd"]
                }
    return data, ret_dict


def load_sipp():
    ####################################################################################################################
    # Data used in the double machine learning paper.
    #
    # Data source: SIPP 1991 (Abadie, 2003)
    # Description of the data: the sample selection and variable contruction follow
    #
    # Abadie, Alberto (2003), "Semiparametric instrumental variable estimation of treatment response
    # models," Journal of Econometrics, Elsevier, vol. 113(2), pages 231-263, April.
    #
    # The variables in the data set include:
    #
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
    ####################################################################################################################
    data = pd.read_stata(os.path.join(root_path, 'sipp1991.dta'))

    # Define Variables
    ret_dict = {'predictorVariable': "net_tfa",
                'treatmentVariable': "e401",
                'treatmentVariableTree': "p401",
                'remainingVariables': ["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown"]
                }
    return data, ret_dict
