# #Train and Save a Machine Learning Model
# Implement a training script (e.g., using scikit-learn) for a classification model like
# logistic regression or decision tree.
# Save the trained model using joblib or pickle.
# Commit the training script and output model file to Git
import joblib
from sklearn.datasets import load_iris      
from sklearn.linear_model import LogisticRegression

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = LogisticRegression().fit(X, y)

# Save model
joblib.dump(model, 'iris_model.joblib')
print("Model trained and saved successfully as 'iris_model.joblib'")
