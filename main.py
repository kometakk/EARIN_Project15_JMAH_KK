import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC as SupportVectorMachineClassifier
# We'll use Gaussian N-B, because it works also for continuous input parameters
# Bernouli works for binary params, multinominal works for occurance-counting params
# Info taken from sklearn and GeeksForGeeks websites
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

import random
import numpy as np
import csv
from io import StringIO

def load_clear_patient_data_new():
    # Load the data using genfromtxt
    data = np.genfromtxt(
        "./heart.csv",
        delimiter=",",
        names=True,      # Use the first line as column names
        dtype=None,      # Automatically determine data types
        encoding='utf-8' # Ensure proper string decoding
    )

    X_list = []
    y_list = []
    for row in data:
        X_list.append(
            [
                row['age'],
                row['sex'],
                row['cp'],
                row['trtbps'],
                row['chol'],
                row['fbs'],
                row['restecg'],
                row['thalachh'],
                row['exng'],
                row['oldpeak'],
                row['slp'],
                row['caa'],
                row['thall'],
            ]
        )
        y_list.append(row['output'])

    X = np.array(X_list, dtype=object)
    y = np.array(y_list)
    return X, y

def load_patient_data_old():
    # Load the data using genfromtxt
    data = np.genfromtxt(
        "./heart_attack_prediction_dataset.csv",
        delimiter=",",
        names=True,      # Use the first line as column names
        dtype=None,      # Automatically determine data types
        encoding='utf-8' # Ensure proper string decoding
    )
    return data

# pacient_data: Patient ID,Age,Sex,Cholesterol,Blood Pressure,Heart Rate,Diabetes,Family History,Smoking,Obesity,Alcohol Consumption,Exercise Hours Per Week,Diet,Previous Heart Problems,Medication Use,Stress Level,Sedentary Hours Per Day,Income,BMI,Triglycerides,Physical Activity Days Per Week,Sleep Hours Per Day,Country,Continent,Hemisphere,Heart Attack Risk
# to
# X:Age,IsMale,Cholesterol,SystolicPressure,DisatolicPressure,HeartRate,Diabetes,FamilyHistory,Smoking,Obesity,AlcoholConsumption,ExerciseHoursPerWeek,Diet,PreviousHeartProblems,MedicationUse,StressLevels,SedentaryHoursPerDay,Income,BMI,Triglycerides,PhysicalActivityDaysPerWeek,SleepHoursPerDay,Country
# y:HeartAttackRisk
def clear_pacient_data_old(pacient_data):
    X_list = []
    y_list = []

    country_dict = dict()
    country_dict_index = 0

    for row in pacient_data:
        is_male = 1 if row['Sex'].lower() == 'male' else 0

        systolic, diastolic = map(int, row['Blood_Pressure'].split('/'))

        diet_map = {'Healthy': 2, 'Average': 1, 'Unhealthy': 0}
        diet = diet_map.get(row['Diet'], -1)

        country = row['Country']
        if(country not in country_dict):
            country_dict[country] = country_dict_index
            country_dict_index += 1
        this_country_index = country_dict[country]

        X_row = [
            row['Age'],
            is_male,
            row['Cholesterol'],
            systolic,
            diastolic,
            row['Heart_Rate'],
            row['Diabetes'],
            row['Family_History'],
            row['Smoking'],
            row['Obesity'],
            row['Alcohol_Consumption'],
            row['Exercise_Hours_Per_Week'],
            diet,
            row['Previous_Heart_Problems'],
            row['Medication_Use'],
            row['Stress_Level'],
            row['Sedentary_Hours_Per_Day'],
            row['Income'],
            row['BMI'],
            row['Triglycerides'],
            row['Physical_Activity_Days_Per_Week'],
            row['Sleep_Hours_Per_Day'],
            this_country_index
        ]

        y_row = row['Heart_Attack_Risk']

        X_list.append(X_row)
        y_list.append(y_row)

    X = np.array(X_list, dtype=object)
    y = np.array(y_list)

    return X, y

def train_models(X, y):
    log_regr = LogisticRegression(random_state=42) # Scale
    dec_tree = DecisionTreeClassifier(random_state=42) # No scaling
    svm_clssf = SupportVectorMachineClassifier(random_state=42) # Scale
    rf_clssf = RandomForestClassifier(random_state=42) # No scaling
    gauss_naiveb = GaussianNB() # ?
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_regr.fit(X_train_scaled, y_train)
    dec_tree.fit(X_train, y_train)
    svm_clssf.fit(X_train_scaled, y_train)
    rf_clssf.fit(X_train, y_train)
    gauss_naiveb.fit(X_train_scaled, y_train)

    y_log_regr = log_regr.predict(X_test_scaled)
    y_dec_tree = dec_tree.predict(X_test)
    y_svm_clssf = svm_clssf.predict(X_test_scaled)
    y_rf_clssf = rf_clssf.predict(X_test)
    y_gauss_naiveb = gauss_naiveb.predict(X_test_scaled)
    random.seed(42)
    y_random_guessing = np.array(random.choices([0, 1], k=len(y_test)))

    # y columns:
    # y_log_regr, y_dec_tree, y_svm_clssf, y_rf_clssf, y_gauss_naiveb, y_random_guessing
    model_results = np.array([y_log_regr, y_dec_tree, y_svm_clssf, y_rf_clssf, y_gauss_naiveb, y_random_guessing])
    model_names = ['Logistic Regression', 'Decision Tree', 'SVM Classifier', 'Random Forest', 'Gaussian NB', 'Random Guessing Classifier']
    model_results_accuracies = []
    model_results_precision = []
    model_results_recall = []
    model_results_conf_matrix = []
    for i in range(len(model_results)):
        model_results_accuracies.append(accuracy_score(y_test, model_results[i]))
        model_results_precision.append(precision_score(y_test, model_results[i]))
        model_results_recall.append(recall_score(y_test, model_results[i]))
        model_results_conf_matrix.append(confusion_matrix(y_test, model_results[i]))
        print(f"{model_names[i]} results: {model_results[i]}")
    output_file = open("model_metrics_scores.txt", "w")
    # confusion_matrix[actual][predicted]

    column_names = ["Model name", "Accuracy", "Precision", "Recall", "CM TrueNeg", "CM FalsePos", "CM FalseNeg", "CM TruePos"]
    output_string = ""
    for i in range(len(column_names)):
        output_string += column_names[i]
        if(i != len(column_names)-1):
            output_string += ","
    output_string += "\n"
    output_file.write(output_string)

    for i in range(len(model_results)):
        output_string = ""
        output_string += str(model_names[i]) + ","
        output_string += str(model_results_accuracies[i]) + ","
        output_string += str(model_results_precision[i]) + ","
        output_string += str(model_results_recall[i]) + ","
        output_string += str(model_results_conf_matrix[i][0][0]) + ","
        output_string += str(model_results_conf_matrix[i][0][1]) + ","
        output_string += str(model_results_conf_matrix[i][1][0]) + ","
        output_string += str(model_results_conf_matrix[i][1][1]) + "\n"
        output_file.write(output_string)
    output_file.close()
        

    
def train_logistic_regression_with_grid_search(X, y, output_csv_path="./log_reg_trainings.csv"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty_solver': [
            ('l1', 'liblinear'),
            ('l1', 'saga'),
            ('l2', 'liblinear'),
            ('l2', 'lbfgs'),
        ],
        'max_iter': [100, 200, 500],
        'class_weight': [None, 'balanced']
    }

    headers = [
        "C", "Penalty", "Solver", "Max iterations",
        "Accuracy", "Precision", "Recall",
        "CM TrueNeg", "CM FalsePos", "CM FalseNeg", "CM TruePos"
    ]
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for c in param_grid['C']:
            for penalty, solver in param_grid['penalty_solver']:
                for max_iter in param_grid['max_iter']:
                    for class_weight in param_grid['class_weight']:
                        try:
                            model = LogisticRegression(
                                C=c,
                                penalty=penalty,
                                solver=solver,
                                max_iter=max_iter,
                                random_state=42,
                                class_weight='balanced'
                            )
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)

                            acc = accuracy_score(y_test, y_pred)
                            prec = precision_score(y_test, y_pred, zero_division=0)
                            rec = recall_score(y_test, y_pred, zero_division=0)
                            cm = confusion_matrix(y_test, y_pred).ravel()

                            row = [c, penalty, solver, max_iter, acc, prec, rec] + list(cm)
                            writer.writerow(row)

                        except Exception as e:
                            print(f"Skipped combination (C={c}, penalty={penalty}, solver={solver}, max_iter={max_iter}) due to error: {e}")

                    
def train_decision_tree_with_grid_search(X, y, output_csv_path="./decision_tree_trainings.csv"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced']
    }

    headers = [
        "Criterion", "Max Depth", "Min Samples Split", "Min Samples Leaf", "Class Weight",
        "Accuracy", "Precision", "Recall",
        "CM TrueNeg", "CM FalsePos", "CM FalseNeg", "CM TruePos"
    ]
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for criterion in param_grid['criterion']:
            for max_depth in param_grid['max_depth']:
                for min_split in param_grid['min_samples_split']:
                    for min_leaf in param_grid['min_samples_leaf']:
                        for class_weight in param_grid['class_weight']:
                            try:
                                model = DecisionTreeClassifier(
                                    criterion=criterion,
                                    max_depth=max_depth,
                                    min_samples_split=min_split,
                                    min_samples_leaf=min_leaf,
                                    class_weight=class_weight,
                                    random_state=42
                                )
                                model.fit(X_train_scaled, y_train)
                                y_pred = model.predict(X_test_scaled)

                                acc = accuracy_score(y_test, y_pred)
                                prec = precision_score(y_test, y_pred, zero_division=0)
                                rec = recall_score(y_test, y_pred, zero_division=0)
                                cm = confusion_matrix(y_test, y_pred).ravel()

                                row = [
                                    criterion, max_depth, min_split, min_leaf, class_weight,
                                    acc, prec, rec
                                ] + list(cm)
                                writer.writerow(row)

                            except Exception as e:
                                print(f"Skipped combination due to error: {e}")


def train_svm_with_grid_search(X, y, output_csv_path="./svm_trainings.csv"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'class_weight': [None, 'balanced']
    }

    headers = [
        "C", "Kernel", "Gamma", "Class Weight",
        "Accuracy", "Precision", "Recall",
        "CM TrueNeg", "CM FalsePos", "CM FalseNeg", "CM TruePos"
    ]
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for c in param_grid['C']:
            for kernel in param_grid['kernel']:
                for gamma in param_grid['gamma']:
                    for class_weight in param_grid['class_weight']:
                        try:
                            model = SupportVectorMachineClassifier(
                                C=c,
                                kernel=kernel,
                                gamma=gamma,
                                class_weight=class_weight,
                                random_state=42
                            )
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)

                            acc = accuracy_score(y_test, y_pred)
                            prec = precision_score(y_test, y_pred, zero_division=0)
                            rec = recall_score(y_test, y_pred, zero_division=0)
                            cm = confusion_matrix(y_test, y_pred).ravel()

                            row = [
                                c, kernel, gamma, class_weight,
                                acc, prec, rec
                            ] + list(cm)
                            writer.writerow(row)

                        except Exception as e:
                            print(f"Skipped combination due to error: {e}")

def train_random_forest_with_grid_search(X, y, output_csv_path="./random_forest_trainings.csv"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': [None, 'balanced']
    }

    headers = [
        "N Estimators", "Criterion", "Max Depth", "Min Samples Split", "Min Samples Leaf", "Class Weight",
        "Accuracy", "Precision", "Recall",
        "CM TrueNeg", "CM FalsePos", "CM FalseNeg", "CM TruePos"
    ]
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for n_estimators in param_grid['n_estimators']:
            for criterion in param_grid['criterion']:
                for max_depth in param_grid['max_depth']:
                    for min_split in param_grid['min_samples_split']:
                        for min_leaf in param_grid['min_samples_leaf']:
                            for class_weight in param_grid['class_weight']:
                                try:
                                    model = RandomForestClassifier(
                                        n_estimators=n_estimators,
                                        criterion=criterion,
                                        max_depth=max_depth,
                                        min_samples_split=min_split,
                                        min_samples_leaf=min_leaf,
                                        class_weight=class_weight,
                                        random_state=42,
                                        n_jobs=-1
                                    )
                                    model.fit(X_train_scaled, y_train)
                                    y_pred = model.predict(X_test_scaled)

                                    acc = accuracy_score(y_test, y_pred)
                                    prec = precision_score(y_test, y_pred, zero_division=0)
                                    rec = recall_score(y_test, y_pred, zero_division=0)
                                    cm = confusion_matrix(y_test, y_pred).ravel()

                                    row = [
                                        n_estimators, criterion, max_depth, min_split, min_leaf, class_weight,
                                        acc, prec, rec
                                    ] + list(cm)
                                    writer.writerow(row)

                                except Exception as e:
                                    print(f"Skipped combination due to error: {e}")

def train_gaussian_nb_with_grid_search(X, y, output_csv_path="./gaussian_nb_trainings.csv"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'var_smoothing': np.logspace(-12, -6, 7)
    }

    headers = [
        "Var Smoothing",
        "Accuracy", "Precision", "Recall",
        "CM TrueNeg", "CM FalsePos", "CM FalseNeg", "CM TruePos"
    ]
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for var_smoothing in param_grid['var_smoothing']:
            try:
                model = GaussianNB(var_smoothing=var_smoothing)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                cm = confusion_matrix(y_test, y_pred).ravel()

                row = [
                    var_smoothing,
                    acc, prec, rec
                ] + list(cm)
                writer.writerow(row)

            except Exception as e:
                print(f"Skipped var_smoothing={var_smoothing} due to error: {e}")

def evaluate_random_guessing(X, y, output_csv_path="./random_guessing_results.csv"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rng = np.random.default_rng(seed=42)
    y_pred = rng.integers(0, 2, size=len(y_test))

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).ravel()

    headers = ["Accuracy", "Precision", "Recall", "CM TrueNeg", "CM FalsePos", "CM FalseNeg", "CM TruePos"]
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerow([acc, prec, rec] + list(cm))

def evaluate_constant_zero(X, y, output_csv_path="./constant_zero_results.csv"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred = np.zeros_like(y_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).ravel()

    headers = ["Accuracy", "Precision", "Recall", "CM TrueNeg", "CM FalsePos", "CM FalseNeg", "CM TruePos"]
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerow([acc, prec, rec] + list(cm))

def evaluate_constant_one(X, y, output_csv_path="./constant_one_results.csv"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred = np.ones_like(y_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).ravel()

    headers = ["Accuracy", "Precision", "Recall", "CM TrueNeg", "CM FalsePos", "CM FalseNeg", "CM TruePos"]
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerow([acc, prec, rec] + list(cm))


if __name__ == '__main__':
    ### Most recent:
    X, y = load_clear_patient_data_new()
    path_templ = './New results/results_'
    train_logistic_regression_with_grid_search(X, y, f'{path_templ}logistic_regression.csv')
    train_decision_tree_with_grid_search(X, y, f'{path_templ}decision_tree.csv')
    train_svm_with_grid_search(X, y, f'{path_templ}support_vector_machine.csv')
    train_random_forest_with_grid_search(X, y, f'{path_templ}random_forest.csv')
    train_gaussian_nb_with_grid_search(X, y, f'{path_templ}gaussian_naive_bayes.csv')
    evaluate_random_guessing(X, y, f'{path_templ}random_guessing.csv')
    evaluate_constant_zero(X, y, f'{path_templ}constant_zero.csv')
    evaluate_constant_one(X, y, f'{path_templ}constant_one.csv')

    ### Training on previous trash data:
    #data = load_patient_data_old()
    #print(data[0])
    #X, y = clear_pacient_data_old(data)

    ##train_logistic_regression_with_grid_search(X, y)
    ##train_decision_tree_with_grid_search(X, y)
    ##train_svm_with_grid_search(X, y)
    ##train_random_forest_with_grid_search(X, y)
    ##train_gaussian_nb_with_grid_search(X, y)
    ##evaluate_random_guessing(X, y)
    ##evaluate_constant_zero(X, y)
    ##evaluate_constant_one(X, y)

    ### Old:
    #data = load_patient_data()
    #print(data[0])
    #X, y = clear_pacient_data(data)
    #print(X[len(X)-1])
    #print(y[len(X)-1])
    #train_models(X, y)