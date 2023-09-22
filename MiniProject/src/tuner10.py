# Imports


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

import math
import seaborn as sns
import warnings  # To ignore the warnings
import pandas as pd
import numpy as np  # For mathematical calculations
import matplotlib.pyplot as plt  # For plotting graphs
from datetime import datetime  # To access datetime
from pandas import Series  # To work on series

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from xgboost import XGBClassifier

# import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

train_layer_10 = pd.read_csv("../data/layer10/train_10.csv")
valid_layer_10 = pd.read_csv("../data/layer10/valid_10.csv")

x_train = train_layer_10.copy().drop(
    columns=["label_1", "label_2", "label_3", "label_4"]
)
x_valid = valid_layer_10.copy().drop(
    columns=["label_1", "label_2", "label_3", "label_4"]
)
x_feature_names = ["feature_" + str(i) for i in range(1, 769)]

id_train = train_layer_10["label_1"].to_frame()
age_train = train_layer_10["label_2"].to_frame()  # id has NaN
gender_train = train_layer_10["label_3"].to_frame()
accent_train = train_layer_10["label_4"].to_frame()  # Accent has bias to 6

id_valid = valid_layer_10["label_1"].to_frame()
age_valid = valid_layer_10["label_2"].to_frame()
gender_valid = valid_layer_10["label_3"].to_frame()
accent_valid = valid_layer_10["label_4"].to_frame()

# Scaling using RobustScaler
scaler_robust = RobustScaler()
scaler_robust.fit(x_train)

x_train_scaled = pd.DataFrame(scaler_robust.transform(x_train), columns=x_feature_names)
x_valid_scaled = pd.DataFrame(scaler_robust.transform(x_valid), columns=x_feature_names)

print("Function Initialised")


def do_pca(train, valid, variance: float = None, n_components: int = None):
    pca_obj = PCA(n_components=0.95, svd_solver="full")
    if variance:
        pca_obj = PCA(n_components=variance, svd_solver="full")
    elif n_components:
        pca_obj = PCA(n_components=n_components)
    pca_obj.fit(train)
    n_components = pca_obj.components_.shape[0]

    x_train = pd.DataFrame(
        pca_obj.transform(train),
        columns=["feature_pca_" + str(i) for i in range(1, n_components + 1)],
    )
    x_valid = pd.DataFrame(
        pca_obj.transform(valid),
        columns=["feature_pca_" + str(i) for i in range(1, n_components + 1)],
    )

    return x_train, x_valid, n_components, pca_obj


def get_id_data():
    id_data_train_cat = pd.concat([x_train_scaled, id_train], axis=1)
    id_data_valid_cat = pd.concat([x_valid_scaled, id_valid], axis=1)

    # Remove rows with null values
    id_data_cleaned_train_cat = id_data_train_cat.dropna()
    id_data_cleaned_valid_cat = id_data_valid_cat.dropna()

    # Separate X and y again
    id_x_train_cat = id_data_cleaned_train_cat.drop(columns=["label_1"])
    id_y_train_cat = id_data_cleaned_train_cat["label_1"]
    id_x_valid_cat = id_data_cleaned_valid_cat.drop(columns=["label_1"])
    id_y_valid_cat = id_data_cleaned_valid_cat["label_1"].to_frame()

    return id_x_train_cat, id_x_valid_cat, id_y_train_cat, id_y_valid_cat


def get_age_data():
    age_data_train_cat = pd.concat([x_train_scaled, age_train], axis=1)
    age_data_valid_cat = pd.concat([x_valid_scaled, age_valid], axis=1)

    # Remove rows with null values
    age_data_cleaned_train_cat = age_data_train_cat.dropna()
    age_data_cleaned_valid_cat = age_data_valid_cat.dropna()

    # Separate X and y again
    age_x_train_cat = age_data_cleaned_train_cat.drop(columns=["label_2"])
    age_y_train_cat = age_data_cleaned_train_cat["label_2"]
    age_x_valid_cat = age_data_cleaned_valid_cat.drop(columns=["label_2"])
    age_y_valid_cat = age_data_cleaned_valid_cat["label_2"].to_frame()

    return age_x_train_cat, age_x_valid_cat, age_y_train_cat, age_y_valid_cat


def get_gender_data():
    gender_data_train_cat = pd.concat([x_train_scaled, gender_train], axis=1)
    gender_data_valid_cat = pd.concat([x_valid_scaled, gender_valid], axis=1)

    # Remove rows with null values
    gender_data_cleaned_train_cat = gender_data_train_cat.dropna()
    gender_data_cleaned_valid_cat = gender_data_valid_cat.dropna()

    # Separate X and y again
    gender_x_train_cat = gender_data_cleaned_train_cat.drop(columns=["label_3"])
    gender_y_train_cat = gender_data_cleaned_train_cat["label_3"]
    gender_x_valid_cat = gender_data_cleaned_valid_cat.drop(columns=["label_3"])
    gender_y_valid_cat = gender_data_cleaned_valid_cat["label_3"].to_frame()

    return (
        gender_x_train_cat,
        gender_x_valid_cat,
        gender_y_train_cat,
        gender_y_valid_cat,
    )


def get_accent_data():
    accent_data_train_cat = pd.concat([x_train_scaled, accent_train], axis=1)
    accent_data_valid_cat = pd.concat([x_valid_scaled, accent_valid], axis=1)

    # Remove rows with null values
    accent_data_cleaned_train_cat = accent_data_train_cat.dropna()
    accent_data_cleaned_valid_cat = accent_data_valid_cat.dropna()

    # Separate X and y again
    accent_x_train_cat = accent_data_cleaned_train_cat.drop(columns=["label_4"])
    accent_y_train_cat = accent_data_cleaned_train_cat["label_4"]
    accent_x_valid_cat = accent_data_cleaned_valid_cat.drop(columns=["label_4"])
    accent_y_valid_cat = accent_data_cleaned_valid_cat["label_4"].to_frame()

    return (
        accent_x_train_cat,
        accent_x_valid_cat,
        accent_y_train_cat,
        accent_y_valid_cat,
    )


def tune_hyperparameters_svc(classifier_label: str, grid):
    # Set random seed
    np.random.seed(42)
    # Make a dictionary to keep model scores
    final_accuracy = 0
    model = SVC()

    rs_model = RandomizedSearchCV(
        estimator=model,
        param_distributions=grid,
        n_iter=20,
        cv=5,
        verbose=3,
        random_state=42,
        n_jobs=-1,
    )

    x_train = None
    x_valid = None
    y_train = None
    y_valid = None

    # switch case on classifier model
    if classifier_label == "id":
        rs_file = "id_rs_model.pkl"
        x_train_i, x_valid_i, y_train, y_valid = get_id_data()

        x_train, x_valid, n_components, pca_obj = do_pca(
            x_train_i, x_valid_i, n_components=70
        )

        rs_model.fit(x_train, y_train)
        joblib.dump(rs_model, rs_file)
        final_accuracy = rs_model.score(x_valid, y_valid)

        full_ds_model = SVC(**rs_model.best_params_)
        full_ds_model.fit(x_train_i, y_train)
        full_ds_model_score = full_ds_model.score(x_valid_i, y_valid)

        print("Final Accuracy: ", final_accuracy)
        print("Full DS Accuracy: ", full_ds_model_score)
        print("Best Params: ", rs_model.best_params_)

        return final_accuracy, rs_model, pca_obj, full_ds_model, full_ds_model_score

    elif classifier_label == "age":
        rs_file = "age_rs_model.pkl"
        x_train_i, x_valid_i, y_train, y_valid = get_age_data()

        x_train, x_valid, n_components, pca_obj = do_pca(
            x_train_i, x_valid_i, n_components=180
        )

        rs_model.fit(x_train, y_train)
        joblib.dump(rs_model, rs_file)
        final_accuracy = rs_model.score(x_valid, y_valid)

        full_ds_model = SVC(**rs_model.best_params_)
        full_ds_model.fit(x_train_i, y_train)
        full_ds_model_score = full_ds_model.score(x_valid_i, y_valid)

        print("Final Accuracy: ", final_accuracy)
        print("Full DS Accuracy: ", full_ds_model_score)
        print("Best Params: ", rs_model.best_params_)

        return final_accuracy, rs_model, pca_obj, full_ds_model, full_ds_model_score

    elif classifier_label == "gender":
        rs_file = "gender_rs_model.pkl"
        x_train_i, x_valid_i, y_train, y_valid = get_gender_data()

        x_train, x_valid, n_components, pca_obj = do_pca(
            x_train_i, x_valid_i, n_components=32
        )

        rs_model.fit(x_train, y_train)
        joblib.dump(rs_model, rs_file)
        final_accuracy = rs_model.score(x_valid, y_valid)

        full_ds_model = SVC(**rs_model.best_params_)
        full_ds_model.fit(x_train_i, y_train)
        full_ds_model_score = full_ds_model.score(x_valid_i, y_valid)

        print("Final Accuracy: ", final_accuracy)
        print("Full DS Accuracy: ", full_ds_model_score)
        print("Best Params: ", rs_model.best_params_)

        return final_accuracy, rs_model, pca_obj, full_ds_model, full_ds_model_score

    elif classifier_label == "accent":
        rs_file = "accent_rs_model.pkl"
        x_train_i, x_valid_i, y_train, y_valid = get_accent_data()

        x_train, x_valid, n_components, pca_obj = do_pca(
            x_train_i, x_valid_i, n_components=32
        )

        rs_model.fit(x_train, y_train)
        joblib.dump(rs_model, rs_file)
        final_accuracy = rs_model.score(x_valid, y_valid)

        full_ds_model = SVC(**rs_model.best_params_)
        full_ds_model.fit(x_train_i, y_train)
        full_ds_model_score = full_ds_model.score(x_valid_i, y_valid)

        print("Final Accuracy: ", final_accuracy)
        print("Full DS Accuracy: ", full_ds_model_score)
        print("Best Params: ", rs_model.best_params_)

        return final_accuracy, rs_model, pca_obj, full_ds_model, full_ds_model_score


def get_pca_model(classifier_label: str, model_hyperparameters: dict, variance: float):
    if classifier_label == "id":
        x_train_i, x_valid_i, y_train, y_valid = get_id_data()

        x_train, x_valid, n_components, pca_obj = do_pca(
            x_train_i, x_valid_i, variance=variance
        )

        model = SVC(**model_hyperparameters)
        model.fit(x_train, y_train)
        score = model.score(x_valid, y_valid)

        print("Final Accuracy: ", score)

        return model, score, pca_obj

    elif classifier_label == "age":
        x_train_i, x_valid_i, y_train, y_valid = get_age_data()

        x_train, x_valid, n_components, pca_obj = do_pca(
            x_train_i, x_valid_i, variance=variance
        )

        model = SVC(**model_hyperparameters)
        model.fit(x_train, y_train)
        score = model.score(x_valid, y_valid)

        print("Final Accuracy: ", score)

        return model, score, pca_obj

    elif classifier_label == "gender":
        x_train_i, x_valid_i, y_train, y_valid = get_gender_data()

        x_train, x_valid, n_components, pca_obj = do_pca(
            x_train_i, x_valid_i, variance=variance
        )

        model = SVC(**model_hyperparameters)
        model.fit(x_train, y_train)
        score = model.score(x_valid, y_valid)

        print("Final Accuracy: ", score)

        return model, score, pca_obj

    elif classifier_label == "accent":
        x_train_i, x_valid_i, y_train, y_valid = get_accent_data()

        x_train, x_valid, n_components, pca_obj = do_pca(
            x_train_i, x_valid_i, variance=variance
        )

        model = SVC(**model_hyperparameters)
        model.fit(x_train, y_train)
        score = model.score(x_valid, y_valid)

        print("Final Accuracy: ", score)

        return model, score, pca_obj