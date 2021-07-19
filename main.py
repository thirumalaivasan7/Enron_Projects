# !/usr/bin/python

"""
    Starter code for the regression mini-project.

    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).
    Draws a little scatterplot of the training/testing data
    You fill in the regression code where indicated:
"""

import sys
import pickle

sys.path.append("C:/studextra\ud120-projects-master/tools")
from feature_format import featureFormat, targetFeatureSplit


dictionary = pickle.load(open("C:\studextra\ud120-projects-master/final_project/final_project_dataset_modified.pkl", "r"))

### list the features you want to look at--first item in the
### list will be the "target" feature
features_list = ["bonus", "salary"]
data = featureFormat(dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit(data)

### training-testing split needed in regression, just like classification
from sklearn.model_selection import train_test_split

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5,
                                                                          random_state=42)
train_color = "b"
test_color = "b"

### Your regression goes here!
### Please name it reg, so that the plotting code below picks it up and
### plots it correctly. Don't forget to change the test_color above from "b" to
### "r" to differentiate training points from test points.
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(feature_train, target_train)
print(reg.coef_)
print(reg.intercept_)
print(reg.score(feature_test, target_test))
print(reg.score(feature_train, target_train))

### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt

for feature, target in zip(feature_test, target_test):
    plt.scatter(feature, target, color="r")
for feature, target in zip(feature_train, target_train):
    plt.scatter(feature, target, color="b")
### labels for the legend
plt.scatter(feature_test[0], target_test[0], color="r", label="test")
plt.scatter(feature_test[0], target_test[0], color="b", label="train")

### draw the regression line, once it's coded
try:
    plt.plot(feature_test, reg.predict(feature_test))
except NameError:
    pass
reg.fit(feature_test, target_test)
plt.plot(feature_train, reg.predict(feature_train), color="b")
print("score2:", reg.score(feature_train,target_train))
print("slope2: " ,reg.coef_)
print("intercept2:", reg.intercept_)

plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()
