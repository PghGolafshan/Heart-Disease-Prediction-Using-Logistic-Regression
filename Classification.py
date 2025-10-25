from google.colab import drive
drive.mount('/content/drive')

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import RFE

# Load the dataset (change the path as per your directory structure in Colab)
file_path = '/content/drive/MyDrive/heart/heart.csv'
heart_data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(heart_data.head())


# Check for missing values in the dataset
missing_values = heart_data.isnull().sum()
print("Missing Values:\n", missing_values)

# Check data types of each column
data_types = heart_data.dtypes
print("\nData Types:\n", data_types)

# Check the number of unique values in each column
unique_values = heart_data.nunique()
print("\nNumber of Unique Values:\n", unique_values)

# Get a descriptive statistical summary of the dataset
statistical_summary = heart_data.describe()
print("\nStatistical Summary:\n", statistical_summary)


# Exploratory Data Analysis (EDA) with Visualizations
# Histograms for each feature
heart_data.hist(bins=15, figsize=(15, 10))
plt.suptitle("Histograms of Different Features")
plt.show()

# Box plots for each feature
heart_data.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False, figsize=(15, 10))
plt.suptitle("Box Plots of Different Features")
plt.show()

# Correlation Analysis
correlation_matrix = heart_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Heart Disease Features")
plt.show()

# Feature Importance using Recursive Feature Elimination (RFE) with Logistic Regression
model = LogisticRegression(max_iter=1000)  # Increasing max_iter for convergence
rfe = RFE(model, n_features_to_select=1)
fit = rfe.fit(heart_data.drop("output", axis=1), heart_data["output"])
feature_ranking = pd.DataFrame({'Feature': heart_data.drop("output", axis=1).columns,
                                'Importance': fit.ranking_}).sort_values(by="Importance")

print("Feature Ranking:\n", feature_ranking)



# Function to remove outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Applying the function to each column (modify this list based on your dataset's features)
columns_to_check = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']  # Example columns
for column in columns_to_check:
    heart_data = remove_outliers(heart_data, column)

# Checking the shape of the dataset after outlier removal
print("Dataset shape after outlier removal:", heart_data.shape)


from sklearn.preprocessing import StandardScaler

# Separating the features and the target variable
X = heart_data.drop('output', axis=1)  # Assuming 'output' is the target variable
y = heart_data['output']

# Initializing the StandardScaler
scaler = StandardScaler()

# Fitting the scaler to the features and transforming them
X_scaled = scaler.fit_transform(X)

# Creating a new DataFrame with the scaled features
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Displaying the first few rows of the scaled dataset
print(X_scaled_df.head())



from sklearn.preprocessing import StandardScaler

# Separating the features and the target variable
X = heart_data.drop('output', axis=1)  # Assuming 'output' is the target variable
y = heart_data['output']

# Initializing the StandardScaler
scaler = StandardScaler()

# Fitting the scaler to the features and transforming them
X_scaled = scaler.fit_transform(X)

# Creating a new DataFrame with the scaled features
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Displaying the first few rows of the scaled dataset
print(X_scaled_df.head())



from sklearn.linear_model import LogisticRegression

# Initializing the Logistic Regression model
logistic_model = LogisticRegression()

# Training the model with the training data
logistic_model.fit(X_train, y_train)

# Optional: Print the model's coefficients and intercept
print("Model Coefficients:", logistic_model.coef_)
print("Model Intercept:", logistic_model.intercept_)


# Making predictions on the test data
y_pred = logistic_model.predict(X_test)

# Optionally, to view the predicted values
print("Predicted values on test data:", y_pred)


from sklearn.metrics import accuracy_score, confusion_matrix

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


# Predicting on the training set for overfitting check
y_train_pred = logistic_model.predict(X_train)

# Calculating accuracy on the training set
train_accuracy = accuracy_score(y_train, y_train_pred)

# Comparing training and test set accuracy
print("Training Set Accuracy:", train_accuracy)
print("Test Set Accuracy:", accuracy)



import numpy as np

def sigmoid(z):
    """ Sigmoid activation function """
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, weights):
    """ Compute the binary cross-entropy cost """
    m = len(y)
    h = sigmoid(X @ weights)
    epsilon = 1e-5  # to avoid log(0) error
    cost = (1/m) * ((-y).T @ np.log(h + epsilon) - (1 - y).T @ np.log(1 - h + epsilon))
    return cost

def gradient_descent(X, y, weights, learning_rate, iterations):
    """ Gradient Descent to minimize the logistic regression cost function """
    m = len(y)
    cost_history = []

    for i in range(iterations):
        weights = weights - (learning_rate/m) * (X.T @ (sigmoid(X @ weights) - y))
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)

    return weights, cost_history

# Preparing the data
m, n = X_scaled_df.shape
X_with_intercept = np.hstack((np.ones((m, 1)), X_scaled_df))  # Add intercept term
initial_weights = np.zeros(n + 1)  # Initial weights (including intercept)

# Gradient Descent parameters
learning_rate = 0.01
iterations = 1000

# Performing Gradient Descent
optimal_weights, cost_history = gradient_descent(X_with_intercept, y, initial_weights, learning_rate, iterations)

# Print the optimal weights
print("Optimal Weights:", optimal_weights)

# Optionally, plot the cost history over iterations
import matplotlib.pyplot as plt

plt.plot(range(1, iterations + 1), cost_history, color='blue')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.show()
