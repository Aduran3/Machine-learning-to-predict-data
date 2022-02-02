# Classification with k-Nearest Neighbors and the Digits Dataset
#
# @author     Duran, Aaron
# @date       12/05/2021

from sklearn.datasets import load_digits
digits = load_digits()
print(digits.DESCR)

# A) display the two-dimensional array representing the sample image at index 35
# and numeric value of the digit the image represents
print("The array is:", digits.images[35])
print("35th numeric value is:", digits.target[35])

import matplotlib.pyplot as plt
# B) image for the sample image at index 35 of the Digits dataset
plt.imshow(digits.images[35])
plt.show()

from sklearn.model_selection import train_test_split
# C) what numbers of samples would the following statement reserve for
# training and testing purposes?
X_train, X_test, y_train, y_test = train_test_split(
digits.data, digits.target, random_state=11, test_size=0.70)
print("Training set size:", X_train.shape)
print("Testing set size:",X_test.shape )
print("70% is for testing, 3934 samples, while 30% is for training, 1686 samples")

# D) code to get and display the number of training examples and the number of testing
# examples.
XTrain, XTest, YTrain, YTest = 0, 0, 0, 0
for i in X_test:                   # for loops used to count the samples
    XTest = XTest + 1              # of the x and y numbers
for i in X_train:
    XTrain = XTrain + 1
for i in y_test:
    YTest = YTest + 1
for i in y_train:
    YTrain = YTrain + 1

print("Samples for X training:", X_train) # Here we display each training and testing numbers
print("number of Training Samples of x_train:", XTrain)
print("Samples for X testing:", X_test)
print("number of testing samples of x_test:", XTest)
print("Samples for Y training:", y_train)
print("number of training samples of y_train:", YTrain)
print("Samples for Y testing:", y_test)
print("number of testing samples for y_train: ", YTest)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X=X_train, y=y_train)

predicted = knn.predict(X=X_test)
expected = y_test
print("the predicted is:", predicted[:20])
print("the expected is:", expected[:20])

# E) Rewrite the list comprehension in snippet [18] using a for loop
wrongsList = []
for (i, j) in zip(predicted, expected):
    if i != j:
        wrongsList.append((i, j))
print("The wrongs list is:", wrongsList)

print("Model accuracy:%.2f%%" %(knn.score(X_test,y_test)*100))

# [ 0, 1, 130, 0, 0, 0, 0, 1, 6, 0]
# 130 out of the 138 times, row #2 was successful in predicting.
# With this information, at predicted #2 will have a accuracy rate of
# 94.2% of being correct.
