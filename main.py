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
from sklearn.metrics import accuracy_score, classification_report

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

    predictions_log_regr = log_regr.predict(X_test_scaled)
    predictions_dec_tree = dec_tree.predict(X_test)
    predictions_svm_clssf = svm_clssf.predict(X_test_scaled)
    predictions_rf_clssf = rf_clssf.predict(X_test)
    predictions_gauss_naiveb = gauss_naiveb.predict(X_test_scaled)
    predictions_random_guessing = np.array(random.choices([0, 1], k=len(y_test)))

    print("Logistic Regression Accuracy:", accuracy_score(y_test, predictions_log_regr)*100, "%")
    print("Decision Tree Accuracy:", accuracy_score(y_test, predictions_dec_tree)*100, "%")
    print("SVM Classifier Accuracy:", accuracy_score(y_test, predictions_svm_clssf)*100, "%")
    print("Random Forest Accuracy:", accuracy_score(y_test, predictions_rf_clssf)*100, "%")
    print("Gaussian Naive Bayes Accuracy:", accuracy_score(y_test, predictions_gauss_naiveb)*100, "%")
    print("Random Guessing Accuracy:", accuracy_score(y_test, predictions_random_guessing)*100, "%")

if __name__ == '__main__':
    data = load_patient_data()
    #print(data[0])
    X, y = clear_pacient_data(data)
    print(X[len(X)-1])
    print(y[len(X)-1])
    train_models(X, y)
