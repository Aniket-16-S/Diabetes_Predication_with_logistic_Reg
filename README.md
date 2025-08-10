# Diabetes Classification using Logistic Regression

This Python script trains a **Logistic Regression** model to classify whether a person has diabetes based on medical data.

## Working
- Loads dataset from `Diabetes Classification.csv`
- Encodes categorical data (**Gender**) using `OneHotEncoder`
- Splits data into training and test sets
- Trains a Logistic Regression model
- Evaluates model using:
  - Confusion Matrix
  - Accuracy Score
  - Classification Report
- Plots a Confusion Matrix heatmap
- Allows live prediction from user input

## Requirements
Install the required dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
## Usage
Place Diabetes Classification.csv in the same directory as the script.

Run the script:
```bash
python diabetes_classification.py
```
After training, the program will:

Show evaluation metrics and heatmap

Ask if you want to predict more values (y/n)

Enter patient details in sequence when prompted.
Input Format for Prediction :
`Age` `Gender(M/F)` `BMI` `Chol` `TG` `HDL` `LDL` `Cr` `BUN`
```bash 
45 M 26.4 200 150 40 100 1.2 15
```

## Explore my other projects 
[click here Aniket-16-S](https://github.com/Aniket-16-S)

