import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc


dataset = pd.read_csv('Diabetes Classification.csv')

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


print("from :, ", x[1, :])
print(y)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

print("this : ,",x[1, :])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(x_test)
print(y_pred)

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(x_train.shape)
print("hey")


predict = input("Want to predict more values (y/n) ? : ")
if predict.lower() == 'y' :
    x_live = input("Enter data in sequence: Gender(M/F) Age BMI Chol TG HDL LDL Cr BUN: ").strip().split(" ")
    gender = x_live[0].strip().strip("'\"").upper()

    new = []  # start with an empty list
    new.append(gender)

    for i in range(1, 9):
        new.append(float(x_live[i]))

    new = np.array([new], dtype=object) 
            
    
    try :
       y = ct.transform(new)
       print(y)
    except Exception as e:
       print(e)
    
    
    while True:
        x_live = input("Enter data in sequence: Age Gender(M/F) BMI Chol TG HDL LDL Cr BUN: ").strip().split(" ")
        if not x_live or x_live[0] == '':
            break

        # Manual encoding: assuming Gender is 2nd input (index 1)
        gender = x_live[1].upper()
        if gender == 'M':
            x_live[1] = 1
        elif gender == 'F':
            x_live[1] = 0
        else:
            print("Invalid gender. Use M or F.")
            continue

        try:
            # Convert everything to float
            x_live = np.array([float(val) for val in x_live]).reshape(1, -1)
        except ValueError:
            print("Please enter only numeric values (after encoding Gender).")
            continue

        # Apply same preprocessing as training
        x_live_transformed = ct.transform(x_live)  # Use transform, not fit_transform!
        print("Transformed input:", x_live_transformed)

        y_live = model.predict(x_live_transformed)
        print("Prediction:", y_live)

       

