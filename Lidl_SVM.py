
# Lidl Support Vector Machine
# not very accurate as real SVM with more complex and precise calculations
# this is SVM for people who don't really understand the 'advanced' math

# This is for those who understand at least ax + by + c = 0, distances between line and a point, between points,
# --> moving line closer to the points with Perceptron Algorithm

# In short this is simplified SVM algorithm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('ggplot')


def line_points(a=0, b=0, c=0, ref=[-1.0, 1.0]):
    """
    given a,b,c for straight line as ax+by+c=0,
    return a pair of points based on ref values
    e.g line_points(-1,1,2) == [(-1.0, -3.0), (1.0, -1.0)]
    *line_points --> unpacks the list and makes (-1.0, -3.0)  (1.0, -1.0)
    """

    if (a == 0) and (b == 0):
        raise Exception("linePoints: a and b cannot both be zero")

    return [(-c / a, p) if b == 0 else (p, (-c - a * p) / b) for p in ref]


class Lidl_SVM:

    def __init__(self, data):

        '''
        :param data: data in dictionary, with keys -1 and 1 as classifications !!!
        '''

        self.data = data

        # final a, b, c, and c for parallel line + na - (line equation ax + by + c = 0)
        self.a_final, self.b_final, self.c_final = None, None, None
        self.c_parallel_plus, self.c_parallel_minus = None, None


    def optimize_line_points(self, learning_rate=0.01, iterations=1000):

        '''
        :param learning_rate: learning rate how big steps you want to make, while optimizing, default optimizing on 0.01
        :param iterations: number of iterations you want to make
        :return: optimized a, b, c and also c for positive and negative parallel line
        '''

        # final closest point of each class -1 and 1 from the plane
        closest_points_final = {}

        # finding the maximum margin, maximum width of the street
        margin_final = 0

        # give closest_points_final dict keys -1 and 1, so far with no value
        for classification in self.data:
            closest_points_final[classification] = None


        # number of features
        len_features = len([feature for class_features in list(self.data.values()) for feature in class_features])


        for _ in range(len_features // 2):

            # random generated a, b, c
            a, b, c = np.random.randn(3)

            # closest_points from the plane
            closest_points = {}

            # give closest_points dict keys -1 and 1, so far with no value
            for classification in self.data:
                closest_points[classification] = None

            # go through every feature in each class iterations of times
            for _ in range(iterations):

                for classification in self.data:
                    for feature in self.data[classification]:

                        # move line closer to the point if it's bad classified, optimization for plane

                        if (classification == -1) and (np.sign(a * feature[0] + b * feature[1] + c) == 1):
                            a -= learning_rate * feature[0]
                            b -= learning_rate * feature[1]
                            c -= learning_rate


                        elif (classification == 1) and (np.sign(a * feature[0] + b * feature[1] + c) == -1):
                            a += learning_rate * feature[0]
                            b += learning_rate * feature[1]
                            c += learning_rate

            # margin width of optimized line
            margin_width = 0

            for classification in self.data:
                min_distance = 100

                for feature in self.data[classification]:
                    # distance between point and line
                    # point = [x, y]
                    # line = ax + by + c
                    # | ax + by + c | / sqrt(a ** 2 + b ** 2)
                    distance = abs(a * feature[0] + b * feature[1] + c) / np.sqrt(a ** 2 + b ** 2)

                    # find the closest point for each class and the distance between that point and the plane
                    if distance < min_distance:
                        min_distance = distance
                        closest_points[classification] = feature

                margin_width += min_distance


            # find the maximum margin of all the optimized lines
            # find the best optimized line with the widest margin, and take its a, b, c, and c for parallel line + and -
            if margin_width > margin_final:
                margin_final = margin_width

                self.a_final, self.b_final, self.c_final = a, b, c

                # set final closest points
                for classification in self.data:
                    min_distance = 100

                    for feature in self.data[classification]:
                        distance = abs(self.a_final * feature[0] + self.b_final * feature[1] + self.c_final) / \
                                   np.sqrt(self.a_final ** 2 + self.b_final ** 2)

                        if distance < min_distance:
                            min_distance = distance
                            closest_points_final[classification] = feature

                self.c_parallel_plus, self.c_parallel_minus = self.c_final, self.c_final


                # optimize c for parallel lines
                # the lower the c the more up it goes from the plane, and vice versa

                while self.a_final * closest_points_final[1][0] + self.b_final * closest_points_final[1][1] + self.c_parallel_plus >= 0:
                    self.c_parallel_plus -= learning_rate

                while self.a_final * closest_points_final[-1][0] + self.b_final * closest_points_final[-1][1] + self.c_parallel_minus <= 0:
                    self.c_parallel_minus += learning_rate

                # c of the plane is right in the middle of the margin, of the parallel lines
                self.c_final = (self.c_parallel_plus + self.c_parallel_minus) / 2

        return self.a_final, self.b_final, self.c_final, self.c_parallel_plus, self.c_parallel_minus


    def predict(self, new_data):
        # make prediction, new_data must be in 2D list !!!

        self.new_data = new_data

        predictions = []

        for feature in new_data:
            predictions.append(np.sign(self.a_final * feature[0] + self.b_final * feature[1] + self.c_final))

        return predictions


    def plot(self):
        # plot the lines and features

        # line optimized
        plt.axline(*line_points(a=self.a_final, b=self.b_final, c=self.c_final),
                   color="#6C3483", label='Optimized Plane')

        # parallel line +
        plt.axline(*line_points(a=self.a_final, b=self.b_final, c=self.c_parallel_plus),
                   color="#186A3B", label='Parallel Lines', linestyle='--')

        # parallel line -
        plt.axline(*line_points(a=self.a_final, b=self.b_final, c=self.c_parallel_minus),
                   color="#186A3B", linestyle='--')


        # scatter plot for each class its unique color
        colors = ['r', 'b']
        for key, color in zip(self.data, colors):
            for feature in self.data[key]:
                plt.scatter(feature[0], feature[1], c=color)


        # scatter plot new data, user wants to predict
        for new_feature in self.new_data:
            plt.scatter(new_feature[0], new_feature[1], c='#C50EC2', s=100, marker='*')
        plt.scatter([], [], c='#C50EC2', s=100, marker='*', label='New Data')


        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.title('Lidl SVM Plot')
        plt.legend()

        plt.show()


# # example 1
# data_dict = {-1: [[-45, 10],
#                   [-23, -15],
#                   [-31, 2]],
#
#              1: [[63, 18],
#                  [31, 45],
#                  [27, 38]]}

# # example 2
# data_dict = {-1: np.array([[1, 2],
#                            [2, 8],
#                            [3, 8]]),
#
#              1: np.array([[5, 1],
#                           [6, -1],
#                           [7, 3]])}
#

# # example 3
# data_dict = {-1: np.array([[-15, -8],
#                            [0, 0],
#                            [-8, 2],
#                            [2, -4],
#                            [23, -2]]),
#
#              1: np.array([[2, 27],
#                           [-7, 15],
#                           [38, 13]])}


# example 4 with dataset
df = pd.read_csv('iris_modified.csv')

# -1 --> class 0 = Iris-setosa
# 1 --> class 1 = Iris-versicolor
# class 2 --> Iris-virginica
data_dict = {-1: np.array(df[['pet_len', 'pet_wd']][(df['class'] == 0)]),
             1: np.array(df[['pet_len', 'pet_wd']][(df['class'] == 1)])}


# make an instance of Lidl_SVM with given data in dictionary !!!
svm_model = Lidl_SVM(data=data_dict)

# optimize a, b, c, and c for parallel lines (+ and -)
# a, b, c, c_parallel_plus, c_parallel_minus = svm_model.optimize_line_points(learning_rate=0.1, iterations=10_000)
svm_model.optimize_line_points(learning_rate=0.001, iterations=1_000)

# make prediction on new data, must be 2D array !!!
new_data = np.array([[1.8, 1.2], [2.2, 2], [1.2, 2.9], [0.5, 4.5], [9.2, 0.3], [1.9, 0.9]])
prediction = svm_model.predict(new_data)

# plot the lines and features
svm_model.plot()
