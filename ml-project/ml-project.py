import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print("="*60)
print("      DIABETES RISK PREDICTION SYSTEM")
print("="*60)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

df = pd.read_csv("diabetes_data_upload.csv")

print("\nDataset Loaded Successfully")
print("Total Records :", len(df))
print("Total Columns :", len(df.columns))

# -------------------------------------------------
# Convert Yes/No Male/Female to Numbers
# -------------------------------------------------

encoder = LabelEncoder()

for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = encoder.fit_transform(df[col])

# -------------------------------------------------
# Features and Target
# -------------------------------------------------

X = df.drop("class", axis=1)

y = df["class"]

# -------------------------------------------------
# Train Test Split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ======================================================
# Logistic Regression
# ======================================================

print("\n")
print("="*60)
print("LOGISTIC REGRESSION")
print("="*60)

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_prediction = lr.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_prediction)

print("Accuracy :", round(lr_accuracy*100,2), "%")

print("\nConfusion Matrix")

print(confusion_matrix(y_test, lr_prediction))

print("\nClassification Report")

print(classification_report(y_test, lr_prediction))

# Cross Validation

lr_cv = cross_val_score(
    lr,
    X,
    y,
    cv=5
)

print("\nCross Validation Scores")

print(lr_cv)

print("\nAverage Accuracy :",
      round(lr_cv.mean()*100,2), "%")

# ======================================================
# Random Forest
# ======================================================

print("\n")
print("="*60)
print("RANDOM FOREST")
print("="*60)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_prediction = rf.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_prediction
)

print("Accuracy :", round(rf_accuracy*100,2), "%")

print("\nConfusion Matrix")

print(confusion_matrix(
    y_test,
    rf_prediction
))

print("\nClassification Report")

print(classification_report(
    y_test,
    rf_prediction
))

rf_cv = cross_val_score(
    rf,
    X,
    y,
    cv=5
)

print("\nCross Validation Scores")

print(rf_cv)

print("\nAverage Accuracy :",
      round(rf_cv.mean()*100,2), "%")

# ======================================================
# Compare Models
# ======================================================

print("\n")
print("="*60)
print("MODEL COMPARISON")
print("="*60)

print("Logistic Regression Accuracy :",
      round(lr_cv.mean()*100,2), "%")

print("Random Forest Accuracy      :",
      round(rf_cv.mean()*100,2), "%")

if rf_cv.mean() > lr_cv.mean():
    print("\nBest Model : Random Forest")
else:
    print("\nBest Model : Logistic Regression")

# ======================================================
# Predict New Patient
# ======================================================

print("\n")
print("="*60)
print("NEW PATIENT PREDICTION")
print("="*60)

age = int(input("Age : "))

gender = input("Gender (Male/Female) : ")

polyuria = input("Polyuria (Yes/No) : ")

polydipsia = input("Polydipsia (Yes/No) : ")

weight = input("Sudden Weight Loss (Yes/No) : ")

weakness = input("Weakness (Yes/No) : ")

polyphagia = input("Polyphagia (Yes/No) : ")

genital = input("Genital Thrush (Yes/No) : ")

vision = input("Visual Blurring (Yes/No) : ")

healing = input("Delayed Healing (Yes/No) : ")

muscle = input("Muscle Stiffness (Yes/No) : ")



def convert(value):

    if value.lower() == "yes":
        return 1

    if value.lower() == "male":
        return 1

    return 0


new_patient = [[
    age,
    convert(gender),
    convert(polyuria),
    convert(polydipsia),
    convert(weight),
    convert(weakness),
    convert(polyphagia),
    convert(genital),
    convert(vision),
    convert(healing),
    convert(muscle)
]]

new_patient_df = pd.DataFrame(new_patient, columns=X.columns)
prediction = rf.predict(new_patient_df)

print("\n")

if prediction[0] == 1:
    print("Prediction : POSITIVE (Diabetes)")
else:
    print("Prediction : NEGATIVE (No Diabetes)")

print("\nProject Completed Successfully.")