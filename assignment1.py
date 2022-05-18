# id: 12-24-12
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC


def plot_all(data, predictions, boundary=None):

    X1, X2, y = [d[0] for d in data], [d[1] for d in data], [d[2] for d in data]
    x1, y1 = [], [] # correctly predicted +1 green
    x2, y2 = [], [] # correctly predicted -1 blue
    x3, y3 = [], [] # incorrectly predicted +1 yellow
    x4, y4 = [], [] # incorrectly predicted -1 orange

    for i in range(len(y)):
        if y[i] == predictions[i]:
            if y[i] == 1:
                x1.append(X1[i])
                y1.append(X2[i])
            else:
                x2.append(X1[i])
                y2.append(X2[i])
        if y[i] != predictions[i]:
            if y[i] == 1:
                x3.append(X1[i])
                y3.append(X2[i])
            else:
                x4.append(X1[i])
                y4.append(X2[i])

    plt.scatter(x1, y1, c='#00ff00', s=4)
    plt.scatter(x2, y2, c='#0000ff', s=4)
    plt.scatter(x3, y3, c='#FFFF00', s=4)
    plt.scatter(x4, y4, c='#FF8000', s=4)

    #decision boundary
    X = np.arange(-1, 1.1, 0.1)
    plt.plot(X, boundary(X), c='#EE6AA7')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(['decision boundary', 'correctly predicted +1', 'correctly predicted -1', 'incorrect +1', 'incorrect -1'])
    plt.show()
    return


def part_a_i(data):
    X1, X2, y = [d[0] for d in data], [d[1] for d in data], [d[2] for d in data]
    x1, y1 = [], []
    x2, y2 = [], []
    for i in range(len(y)):
        if y[i] == 1:
            x1.append(X1[i])
            y1.append(X2[i])
        else:
            x2.append(X1[i])
            y2.append(X2[i])

    plt.scatter(x1, y1, c='#00ff00', s=6, label='+1')
    plt.scatter(x2, y2, c='#0000ff', s=6, label='-1')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


def part_a_ii(data):
    X1, X2, y = [d[0] for d in data], [d[1] for d in data], [d[2] for d in data]
    X = np.column_stack((X1, X2))
    model = LogisticRegression(penalty='none', solver='lbfgs').fit(X,y)
    print(model.intercept_, model.coef_[0])
    return model


def part_a_iii(data, model):
    X1, X2, y = [d[0] for d in data], [d[1] for d in data], [d[2] for d in data]
    X = np.column_stack((X1, X2))
    ps = model.predict(X)
    accuracy = accuracy_score(y, ps)
    print('accuracy = ', accuracy)

    #decision boundary function
    m0 = model.coef_[0][0]
    m1 = model.coef_[0][1]
    c = model.intercept_
    boundary = lambda x: -((m0 * x) + c)/m1
    plot_all(data, ps, boundary)


def part_b(data):
    X1, X2, y = [d[0] for d in data], [d[1] for d in data], [d[2] for d in data]
    X = np.column_stack((X1, X2))
    for C in (0.001, 1, 100):
        model = LinearSVC(C=C).fit(X, y)
        ps = model.predict(X)
        accuracy = accuracy_score(y, ps)
        m0 = model.coef_[0][0]
        m1 = model.coef_[0][1]
        c = model.intercept_
        boundary = lambda x: -((m0 * x)/(c / m1))
        print(C, model.intercept_, model.coef_[0], accuracy)
        plot_all(data, ps, boundary)


def part_c(data):
    x1, x2, x3, x4 = [], [], [], []
    for d1, d2, y in data:
        x1.append(d1)
        x2.append(d2)
        x3.append(pow(d1, 2))
        x4.append(pow(d2, 2))
    X = np.column_stack((x1, x2, x3, x4))
    y = [d[2] for d in data]
    model = LogisticRegression(penalty='none', solver='lbfgs').fit(X, y)
    print(model.intercept_, model.coef_[0])
    ps = model.predict(X)
    accuracy = accuracy_score(y, ps)
    print('accuracy', accuracy)
    m0, m1, m2, m3 = model.coef_[0]
    c = model.intercept_
    boundary = lambda x: (-m1 + pow(m1 * m1 - 4*m3*(m2*x*x +m0*x +c), 0.5)) / 2*m3
    plot_all(data, ps, boundary)


raw_data = pd.read_csv('data.csv').values.tolist()
part_a_i(raw_data)
part_a_ii(raw_data)
part_a_iii(raw_data, part_a_ii(raw_data))
part_b(raw_data)
part_c(raw_data)