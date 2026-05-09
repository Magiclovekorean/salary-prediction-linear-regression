




import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer







df = pd.read_csv("data.csv")
X = df.drop(columns=["No", "house price of unit area"])
y = df["house price of unit area"]

# Separate training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=1
)








model = LinearRegression()
model.fit(X_train, y_train)







y_pred = model.predict(X_test)
mse = mean_squared_error(y_pred, y_test)
print(f"MSE: {mse}")



features = df.drop(columns=["No", "house price of unit area"]).columns.tolist()

X_predict = []
for i in range(len(features)):
    X_predict.append(int(input(f"Enter {features[i]}")))

y_predicted = model.predict(X_predict)

