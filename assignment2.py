# id:4--4-4
from mpl_toolkits.mplot3d import Axes3D
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_all(data, predictions, results):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    X1, X2 = [p[0] for p in predictions], [p[1] for p in predictions]
    ax.plot_trisurf(X1, X2, results, color='#0000ee88')
    X, Y, Z = [d[0] for d in data], [d[1] for d in data], [d[2] for d in data]
    ax.scatter(X, Y, Z, label='training data', c='#ff000088', s=8)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('target')
    ax.set_xlim3d(-3, 3)
    ax.set_ylim3d(-3, 3)
    ax.set_zlim3d(-30, 30)
    ax.legend()
    plt.show()


def part_i_a(data):
    X1, X2, y = [d[0] for d in data], [d[1] for d in data], [d[2] for d in data]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X1, X2, y, label='training data', c='#00ff00', )
    ax.set_zlim3d(-3, 3)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('target')
    ax.legend()
    plt.show()


def part_i_bce(data, isLasso=True):
    #part b
    X1, X2, y = [d[0] for d in data], [d[1] for d in data], [d[2] for d in data]
    X = np.column_stack((X1, X2))
    poly = PolynomialFeatures(5).fit_transform(X)
    range = [1, 10, 100, 1000, 10000]

    #part c
    test = []
    grid = np.linspace(-3, 3)
    for i in grid:
        for j in grid:
            test.append([i, j])
    test = np.array(test)
    test_with_poly = PolynomialFeatures(5).fit_transform(test)

    #together
    for C in range:
        if isLasso:
            model = Lasso(alpha=1 / (2 * C)).fit(poly, y)
        else: #part e
            model = Ridge(alpha=1/(2*C)).fit(poly,y)
        ps = model.predict(test_with_poly)
        print('C=', C, 'intercept=', model.intercept_, 'coefficients=', model.coef_)
        plot_all(data, test, ps)


raw_data = pd.read_csv('data.csv')
raw_data = raw_data.values.tolist()
part_i_a(raw_data)
part_i_bce(raw_data)
part_i_bce(raw_data, isLasso=False)

