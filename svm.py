from sklearn.svm import SVM
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from emd import preprocessor
import numpy as np

training_set = ['foo0.csv', 'foo1.csv',
                'foo2.csv', 'foo3.csv', 'foo4.csv', 'foo5.csv']

testing_set = ["1_3",  "1_5", "1_6", "1_7",
               "2_3", "2_4", "2_5", "2_6", "2_7", "3_3"]

Folders = ["Learning_set", "Full_Test_Set"]

actual_result = [5730, 1610, 1460, 7570, 7530, 1390, 3090, 1290, 580, 820]

Y = np.array([3.8, 3.1, 2.2, 3.1, 2.5, 2.6])

x = []

for j, bearing_set in enumerate(training_set):

    print("training on the bearing {}:".format(j))

    temp = pd.read_csv(bearing_set, header=None)

    x.append(temp)

X = []

for i in range(len(x)):

    N = 2803 - len(x[0])

    t1 = np.pad(x[0], (0, N), 'median')

    X.append(t1)

svm = SVM()

X = np.array(X)

X = np.squeeze(X)

model = svm.fit(X, Y)

accuracy = []

final_result = []

for i in range(len(testing_set)):

    temp = preprocessor(Folders[1], testing_set[i])

    N = 2803 - len(temp)

    temp_1 = np.array(temp).reshape(1, len(temp))

    temp_2 = np.pad(temp_1[0], (0, N), 'median')

    temp_2 = temp_2.reshape(1, len(temp_2))

    predicted_DDT = model.predict(temp_2)

    predicted_FT = temp_1[0][-1]

    def absolute_difference_function(list_value): return abs(
        list_value - predicted_DDT)

    closest_value = min(temp, key=absolute_difference_function)

    result = np.where(temp_1 == closest_value)

    rul = len(temp_1[0]) - result[1][0]

    final_result.append(rul*10)

    accuracy.append(abs((rul*10)-actual_result[i])/actual_result[i])

for i, res in enumerate(final_result):

    print("The predicted remaining useful life of the given bearing is : {} seconds and given RUL for the same is: {}".format(
        final_result[i], actual_result[i]))

for i, acc in enumerate(accuracy):

    print("The accuracy for the following test set {} is: {}%".format(i+1, acc*100))
