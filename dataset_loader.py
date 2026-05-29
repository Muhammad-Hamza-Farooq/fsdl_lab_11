from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)

# BUG HERE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Total samples:", len(X))
print("Training samples:", len(X_train))
print("Test samples:", len(X_test))
print("Train %:", round(len(X_train) / len(X) * 100, 1))
print("Test %:", round(len(X_test) / len(X) * 100, 1))
