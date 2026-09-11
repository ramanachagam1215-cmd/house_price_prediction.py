import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
data = {
    "area":[1000, 1200, 1500,1800, 2000],
    "price": [200000, 240000, 300000, 360000, 400000]
}
df = pd.DataFrame(data)

print(df)
x = df[["area"]]
y = df["price"]
x_train, x_test, y_train, y_test, = train_test_split(x, y, test_size=0.4 , random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
area = float(input("Enter house area in sq.ft:"))

predicted_price = model.predict([[area]])

print("\nPredicted house price:", predicted_price[0])
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)


