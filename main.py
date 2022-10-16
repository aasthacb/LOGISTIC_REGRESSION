import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# I have manually converted Categorical variables for Wind Directions and Rain_today and Rain_tomorrow to numerical variables
# using excel search and replace and saved a new csv file...
# I could have done it through code too, but just to keep the focus mainly on logistic regression for now...

data = pd.read_csv("logistic_numerical_dataset.csv")


#finding missing values and then filling them with the mean value of the numerical feature.


numerical_variables = ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed", "WindSpeed3pm",
                       "WindSpeed3pm", "Humidity9am", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am",
                       "Temp3pm"]

a = len(numerical_variables)


#filling NA values in numerical columns with the mean value of that column
def fill_missing_values(a, numerical_variables):
    for a in numerical_variables:
        data[a] = data[a].fillna(data[a].mean())


fill_missing_values(a, numerical_variables)


#for removing rows with NA values from non-numerical columns like WindDir9am,WindDir3pm, WindGustDir
data = data.dropna(how="any", axis=0)
data.isnull().sum()  #checking if we still have any null data left



# To simplify the problem further, dropping Date and Location columns from the dataset

data.drop(["Date"], axis=1, inplace=True)  # inplace=True implies that the original dataset (data) will be overwritten.
data.drop(["Location"], axis=1, inplace=True)

#separating source and target datasets
y = data.RainTomorrow
x_data = data.drop(["RainTomorrow"], axis=1)

# normalising x_data
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

#test
#as a simple step, I am taking first 80% of the data as training date and rest 20% as test data

# m= count of 80% of the rows
m = int(x_data.shape[0]*0.8)

x_train = x.iloc[1:m, :].T
x_test  = x.iloc[(m + 1):, :].T
y_train = y.iloc[1:m].T.values
y_test = y.iloc[(m+1):].T.values

#print("x_train: ", x_train.shape, x_train)
#print("y_train: ", y_train.shape, y_train)


#parameter initialize and sigmoid function

# initializing bias to 0.0 and weights to 0.01
def initialize_weights_and_bias(no_of_parameters):
    w = np.full((no_of_parameters, 1), 0.01)
    b = 0.0
    return w, b

#hypothesis function
def sigmoid(z):
    gz = 1 / (1 + np.exp(-z))
    return gz


def find_cost_gradient(weight_matrix, bias, x_train, y_train):
    #to find cost fuction
    z = np.dot(weight_matrix.T, x_train) + bias
    gz = sigmoid(z)
    loss = -y_train * np.log(gz) - (1 - y_train) * np.log(1 - gz)
    cost = (np.sum(loss)) / x_train.shape[1]  # x_train.shape[1]  is for scaling

    # applying gradient descent equation
    gradient_descent_w = (np.dot(x_train, ((gz - y_train).T))) / x_train.shape[1]
    gradient_descent_b = np.sum(gz - y_train) / x_train.shape[1]
    gradients = {"gradient_descent_w": gradient_descent_w, "gradient_descent_b": gradient_descent_b}

    return cost, gradients


# Updating(learning) parameters
def update(weight, bias, x_train, y_train, learning_rate, number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []

    for i in range(number_of_iterarion):
        # find cost and gradients
        cost, gradients = find_cost_gradient(weight, bias, x_train, y_train)
        cost_list.append(cost)

        weight = weight - learning_rate * gradients["gradient_descent_w"]
        bias = bias - learning_rate * gradients["gradient_descent_b"]
        if i % 10 == 0: # to consider cost every 10th iteration
            cost_list2.append(cost)
            index.append(i)
        #    print("Cost after iteration %i: %f" % (i, cost))

    # we update(learn) parameters weights and bias
    parameters = {"weight": weight, "bias": bias}

    #plotting graph Cost vs No. of Iterations
    plt.plot(index, cost_list2)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list


# prediction
def predict(weight, bias, x_test):
    z = sigmoid(np.dot(weight.T, x_test) + bias)
    Y_prediction = np.zeros((1, x_test.shape[1]))
    # if z is > 0.5, our prediction is sign one
    # if z is <= 0.5, our prediction is sign zero
    for i in range(z.shape[1]):
        if z[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction


def logistic_regression(x_train, y_train, learning_rate, num_iterations):
    # initialize
    dimension = x_train.shape[0]
    weight, bias = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(weight, bias, x_train, y_train, learning_rate, num_iterations)

    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)

    # Print test Errors
    print("Prediction accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))


logistic_regression(x_train, y_train, learning_rate=1, num_iterations=1000)


