
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Step 1: Load the dataset
data = pd.read_csv("C:\\Users\\PC\\OneDrive\\Documents\\bank\\bank-full.csv", sep=';')

# Step 2: Data preprocessing

# Convert categorical variables to numeric using LabelEncoder
le = LabelEncoder()

# List of columns that are categorical
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                    'contact', 'month', 'poutcome', 'y']

# Apply LabelEncoder to each of the categorical columns
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Step 3: Splitting data into features and target variable
X = data.drop('y', axis=1)  # Features (independent variables)
y = data['y']  # Target variable (whether customer purchased or not)

# Step 4: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Build the Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = clf.predict(X_test)

# Step 7: Evaluate the model

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Accuracy Score
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 8: Visualizing the Decision Tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X.columns, class_names=['no', 'yes'], filled=True)
plt.show()
