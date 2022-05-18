import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve


def plot_data_3d(data):
    X1, X2, y = [d[0] for d in data], [d[1] for d in data], [d[2] for d in data]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X1, X2, y, c='#00ff00')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('target')
    plt.show()


def plot_features(data):
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


def part_i_a(data):
    X1, X2, y = [d[0] for d in data], [d[1] for d in data], [d[2] for d in data]
    X = np.column_stack((X1, X2))
    poly_range = [1, 2, 3, 4, 5]
    range_c = [0.01, 0.1, 1, 10, 100]
    for n in poly_range:
        means, standevs = [], []
        poly = PolynomialFeatures(n).fit_transform(X)
        for C in range_c:
            model = LogisticRegression(penalty='l2', C=C, max_iter=1000).fit(poly, y)
            scores = cross_val_score(model, poly, y, cv=5, scoring='accuracy')
            print('N=', n, 'C-value=', C)
            print('Accuracy: %0.2f(+/- %0.2f)' % (scores.mean(), scores.std()))
            means.append(scores.mean())
            standevs.append(scores.std())
        plt.errorbar(range_c, means, yerr=standevs, fmt='b')
        plt.xlabel('C')
        plt.ylabel('mean accuracy')
        plt.xscale('log')
        plt.title(f'degree = {n}')
        plt.show()


def part_i_b(data):
    X1, X2, y = [d[0] for d in data], [d[1] for d in data], [d[2] for d in data]
    X = np.column_stack((X1, X2))
    k_range = [1, 5, 10, 15, 25, 50, 100]
    means, standevs = [], []
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k, weights='uniform').fit(X, y)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print('k=', k, 'Accuracy: %0.2f(+/- %0.2f)' % (scores.mean(), scores.std()))
        means.append(scores.mean())
        standevs.append(scores.std())

    plt.errorbar(k_range, means, yerr=standevs, fmt='y')
    plt.xlabel('k')
    plt.xticks(k_range)
    plt.ylabel('mean accuracy')
    plt.show()

def part_i_cd(data, logistic=True):
    X1, X2, y = [d[0] for d in data], [d[1] for d in data], [d[2] for d in data]
    X = np.column_stack((X1, X2))
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
    polyX = PolynomialFeatures(5).fit_transform(X_train)
    if logistic:
        model = LogisticRegression(penalty='l2', C=10).fit(X_train, Y_train)
        ps = model.predict(X_test)
        conf = confusion_matrix(Y_test, ps)
        false, true, _ = roc_curve(Y_test, model.decision_function(X_test))
        plt.plot(false, true)
    else:
        model = KNeighborsClassifier(n_neighbors=15, weights='uniform').fit(X_train, Y_train)
        ps = model.predict(X_test)
        conf = confusion_matrix(Y_test, ps)
        false, true, _ = roc_curve(Y_test, model.predict_proba(X_test)[:,1])
        plt.plot(false, true)
    print(conf)
    dummy = DummyClassifier(strategy='most_frequent').fit(X_train, Y_train)
    ps_dummy = dummy.predict(X_test)
    print('baseline', confusion_matrix(Y_test, ps_dummy))

    #part d - plotting roc curve

    plt.xlabel('False positive')
    plt.ylabel('True positive')
    plt.plot([0, 1], [0, 1], color='pink')
    plt.title('ROC curve')
    plt.show()


data1 = pd.read_csv('data1.csv', comment='#').values.tolist()
data2 = pd.read_csv('data2.csv', comment='#').values.tolist()
#plot_data_3d(data1)
#plot_features(data1)
#part_i_a(data1)
#part_i_b(data1)
part_i_cd(data1, True)
#part_i_cd(data1, False)
