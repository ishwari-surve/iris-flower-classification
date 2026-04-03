import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier,plot_tree

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

Border = "-"*40
########################################################

#Step 1: Load the dataset

#########################################################
print(Border)
print("Step 1: Load the dataset")
print(Border)

DatasetPath = "iris.csv"

df = pd.read_csv(DatasetPath)

print("Dataset gets loaded successfully...")
print("Initial entries from dataset:")
print(df.head())

########################################################

#Step 2: Data Analysis(EDA)

#########################################################

print(Border)
print("Step 2: Data Analysis")
print(Border)

print("Shape of dataset :",df.shape)
print("column Names :",list(df.columns))

print("Missing Values (Per Column)")
print(df.isnull().sum())

print("Class Distribution (Species count)")
print(df["species"].value_counts())

print("Statistical Report of Dataset")
print(df.describe())

########################################################

#Step 3:Decide Independent and Dependent variables

#########################################################

print(Border)
print("Step 3: Decide Independent and Dependent variables")
print(Border)

# X: Independent variables / Features
# Y: Dependent variable / Label

feature_cols = [
    'sepal length (cm)',
    'sepal width  (cm)',
    'petal length  (cm)',
    'petal width  (cm)'
]

X = df[feature_cols]
Y = df['species']

print("X shape :", X.shape)
print("Y shape :", Y.shape)

########################################################

#Step 4 :Visualization of Dataset

#########################################################

print(Border)
print("Step 4 : Visualization of Dataset")
print(Border)

# Scatter plot
plt.figure(figsize=(7,5))

for sp in df["species"].unique():
    temp = df[df["species"] == sp]
    plt.scatter(temp["petal length  (cm)"],
                temp["petal width  (cm)"],
                label=sp)

plt.title("Iris : Petal length vs Petal width")
plt.xlabel("petal length  (cm)")
plt.ylabel("petal width  (cm)")

plt.legend()
plt.grid(True)
plt.show()

########################################################

#Step 5 :Split the dataset for training and testing

#########################################################

print(Border)
print("Step 5 :Split the dataset for training and testing")
print(Border)

# Test size = 20%
# Train size = 80%

X_train, X_test , Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print("Data splitting activity done :")

print("X - Independent : ",X.shape)
print("Y - Dependent : ",Y.shape)

print("X_train : ",X_train.shape)
print("X_test : ",X_test.shape)
print("Y_train : ",Y_train.shape) # (120,)
print("Y_test : ",Y_test.shape) #(30)

########################################################

#Step 6 :Build the model

#########################################################

print(Border)
print("Step 6 :Build the model")
print(Border)

print("We are going to use DecisionTreeClassifier")

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42

)

print("Model successfully created :",model)

########################################################

#Step 7 :Training the model

#########################################################

print(Border)
print("Step 7 :Training the model")
print(Border)

model.fit(X_train,Y_train)

print("Model training completed")

########################################################

#Step 8 :Evaluate the model

#########################################################

print(Border)
print("Step 8 :Evaluate the model")
print(Border)

Y_pred = model.predict(X_test)

print("Model evaluation (testing) complete")

print(Y_pred.shape)

print("Expected answers :")
print(Y_test)

print("Predicted answers :")
print(Y_pred)

########################################################

#Step 9 :Evaluate the model for performance

#########################################################

print(Border)
print("Step 9 :Evaluate the model for performance")
print(Border)

accuracy = accuracy_score(Y_test,Y_pred)
print("Accuracy of model is :",accuracy*100)

cm = confusion_matrix(Y_test,Y_pred)

print("Confusion matrix :")
print(cm)

print("Classification Report")
print(classification_report(Y_test,Y_pred))

########################################################

#Step 10 :Plot confussion matrix

#########################################################

print(Border)
print("Step 10 :Plot confussion matrix")
print(Border)

data = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
data.plot()
plt.title("Confusion matrix of Iris dataset")
plt.show()

from sklearn.neighbors import KNeighborsClassifier

########################################################

# Step 11: Build and train KNN model

#########################################################

print(Border)
print("Step 11: Build and train KNN model")
print(Border)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
print("KNN model training completed")

########################################################

# Step 12: Evaluate KNN model

#########################################################

print(Border)
print("Step 12: Evaluate KNN model")
print(Border)

Y_pred_knn = knn.predict(X_test)

knn_accuracy = accuracy_score(Y_test, Y_pred_knn)
print("KNN Accuracy:", round(knn_accuracy * 100, 2), "%")

print("Confusion Matrix - KNN:")
cm_knn = confusion_matrix(Y_test, Y_pred_knn)
print(cm_knn)

print("Classification Report - KNN:")
print(classification_report(Y_test, Y_pred_knn))

########################################################

# Step 13: Compare Decision Tree vs KNN

#########################################################

print(Border)
print("Step 13: Model Comparison")
print(Border)

dt_accuracy = accuracy_score(Y_test, Y_pred)

print(f"Decision Tree Accuracy : {round(dt_accuracy * 100, 2)}%")
print(f"KNN Accuracy           : {round(knn_accuracy * 100, 2)}%")

if knn_accuracy > dt_accuracy:
    print("Best Model: KNN")
elif dt_accuracy > knn_accuracy:
    print("Best Model: Decision Tree")
else:
    print("Both models perform equally")