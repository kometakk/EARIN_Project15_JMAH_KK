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
from io import StringIO

def load_patient_data():
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
def clear_pacient_data(pacient_data):
    X_list = []
    y_list = []

    country_dict = dict()
    country_dict_index = 0

    for row in pacient_data:
        # Convert sex to binary
        is_male = 1 if row['Sex'].lower() == 'male' else 0

        # Split blood pressure
        systolic, diastolic = map(int, row['Blood_Pressure'].split('/'))

        # Encode diet (basic encoding; can be one-hot or ordinal based on context)
        diet_map = {'Healthy': 2, 'Average': 1, 'Unhealthy': 0}
        diet = diet_map.get(row['Diet'], -1)

        country = row['Country']
        if(country not in country_dict):
            country_dict[country] = country_dict_index
            country_dict_index += 1
        this_country_index = country_dict[country]

        # Create feature vector
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

    # Convert to numpy arrays
    X = np.array(X_list, dtype=object)  # dtype=object to allow mixed types
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
        

    



if __name__ == '__main__':
    data = load_patient_data()
    #print(data[0])
    X, y = clear_pacient_data(data)
    print(X[len(X)-1])
    print(y[len(X)-1])
    train_models(X, y)
