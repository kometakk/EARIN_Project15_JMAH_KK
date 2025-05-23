import kagglehub

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

import random
import numpy

def prepare_models():
    log_reg = LogisticRegression(max_iter=200)
    rf_clf = RandomForestClassifier(random_state=42)

    return(log_reg, rf_clf)

if __name__ == '__main__':
    # Download latest version
    data_file = open("./heart_attack_prediction_dataset.csv", "r")