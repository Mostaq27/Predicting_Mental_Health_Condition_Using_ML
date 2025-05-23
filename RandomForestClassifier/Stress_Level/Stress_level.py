# First, install the required packages via your terminal (do this once):
# pip install pandas numpy scikit-learn matplotlib seaborn xgboost

from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd

# Hide the root Tk window
Tk().withdraw()

# Open file dialog
file_path = askopenfilename(title="/AI_lab_Conference_Model/Train_Data/DASS.csv")
df = pd.read_csv(file_path)
print(df.head())

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/AI_lab_Conference_Model/Train_Data/DASS.csv")
df.head()

# Features and target
stress_features = ['Q3_1_S1','Q3_2_S2','Q3_3_S3','Q3_4_S4','Q3_5_S5','Q3_6_S6','Q3_7_S7']
X = df[stress_features]
y = df['Stress_Level']  # Change to 'Anxiety_Level' or 'Depression_Level' if needed

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt='g')
plt.title("Confusion Matrix for Stress Level")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()