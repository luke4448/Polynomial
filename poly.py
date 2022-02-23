# Creating polynomial regression
# Parking lot
# import the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
y = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]

# method for polynomial model
model = np.poly1d(np.polyfit(x, y, 3))
# Did some research from w3schools.com
# Specifying the display position
# position 1 and end at position 22
line = np.linspace(1, 22, 100)

plt.scatter(x, y)
plt.plot(line, model(line))
plt.title('Parking Bays')
plt.xlabel('Diameter in inches')
plt.ylabel('Amount of bays')
plt.show()
# creating polynomial regression method
poly = PolynomialFeatures(degree=2)
X_F1_poly = poly.fit_transform(X_F1)
# train method  to test and split
X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, X_F1_poly, random_state=0)
linreg = LinearRegression().fit(X_train, y_train)
