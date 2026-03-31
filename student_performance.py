#Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#Create dataset
data = {
    'hours_studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'attendance': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    'marks': [35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

#Define input and output
X = df[['hours_studied', 'attendance']]
y = df['marks']

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#Train model
model = LinearRegression()
model.fit(X_train, y_train)

#Predict test data
y_pred = model.predict(X_test)

#Evaluate model
error = mean_absolute_error(y_test, y_pred)
print("\nMean Absolute Error:", error)

#Correct prediction function (NO WARNING)
def predict_marks(hours, attendance):
    input_data = pd.DataFrame([[hours, attendance]], 
                              columns=['hours_studied', 'attendance'])
    result = model.predict(input_data)
    return result[0]

#Test example
print("\nExample Prediction:")
print("Predicted Marks:", predict_marks(6, 80))