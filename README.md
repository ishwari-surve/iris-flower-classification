# Iris Flower Classification

## Objective
Classify Iris flower species (Setosa, Versicolor, Virginica) based on
sepal and petal measurements using Decision Tree and KNN classification algorithms.

## Dataset
- File: iris.csv
- Total Records: 150
- Independent Variables: Sepal Length, Sepal Width, Petal Length, Petal Width
- Dependent Variable: Species (Setosa, Versicolor, Virginica)
- Class Distribution: 50 records per species

## Technologies Used
- Python 3
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Project Workflow
- Step 1: Load the dataset
- Step 2: Data Analysis and EDA
- Step 3: Decide independent and dependent variables
- Step 4: Visualization of dataset (Petal Length vs Petal Width scatter plot)
- Step 5: Split dataset into training and testing sets (80/20)
- Step 6: Build Decision Tree model
- Step 7: Train the model
- Step 8: Test the model
- Step 9: Evaluate model performance
- Step 10: Plot confusion matrix for Decision Tree
- Step 11: Build and train KNN model
- Step 12: Evaluate KNN model performance
- Step 13: Compare Decision Tree vs KNN accuracy

## Models Used
- Decision Tree Classifier (criterion: gini, max depth: 5)
- K-Nearest Neighbors Classifier (n_neighbors: 5)

## Model Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

## How to Run
1. Clone this repository
2. Install required libraries:
   pip install pandas matplotlib seaborn scikit-learn
3. Make sure iris.csv is in the same folder as the script
4. Run the script:
   python iris_classification.py

## Result
Both Decision Tree and KNN models are built and compared on the Iris dataset.
The model with higher accuracy is selected as the best performing model.
Confusion matrix and classification report are generated for detailed performance analysis.



