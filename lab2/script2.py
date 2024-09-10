from random import shuffle
from sklearn import tree # pip install scikit-learn
import matplotlib.pyplot as plt # pip install matplotlib

def reader(name):
    values = []
    with open(name, "r") as file:
        hdr = file.readline()[:-1].split(",")
        for line in file.readlines():
            value = *(None if rec == "?" else (float if i == 9 else int)(rec) for i, rec in enumerate(line.split(","))),
            values.append(value)

    L = len(values)
    print("Записей:", L)
    shuffle(values)
    edge = L * 7 // 10
    print("Для обучения:", edge, "тестов:", L - edge)
    learn, tests = values[:edge], values[edge:]

    return hdr, learn, tests

def save_img(name, clf, hdr):
    plt.figure(figsize=(100, 50)) # 10000x5000 pxs без bbox_inches='tight'
    # https://scikit-learn.org/stable/modules/tree.html#tree-mathematical-formulation
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html#sklearn.tree.plot_tree
    # hdr вместо x[0], x[1], x[2]...
    plot = tree.plot_tree(clf, filled=True, feature_names=hdr, rounded=True)
    #print("Узлов:", len(plot), plot[0].__class__)
    plt.title("Decision tree trained on 70% the iris features", fontdict = {"fontsize": 100})
    #plt.show()
    plt.savefig(f"decision_tree_{name}.png", bbox_inches='tight')
    plt.close()

def checker(y, y2, alg):
    assert len(y) == len(y2)
    errs = 0
    for a, b in zip(y, y2):
        v = "YEAH" if a == b else "ERROR"
        if a != b: errs += 1
        #print(f"real: {a}, prediction: {b}, verdict: {v}")
    L = len(y)
    acc = (L - errs) / L
    print(f"всего записей: {L} | верно: {L - errs} | неверно: {errs} | аккуратность: {acc * 100 :.3f}%")
    return L, L - errs, errs, acc, alg

def learner(X, y, crit, mln, md):
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    clf = tree.DecisionTreeClassifier(
        criterion = ("gini", "entropy", "log_loss")[crit],
        max_leaf_nodes = mln,
        max_depth = md,
        #random_state = 0
    )
    clf.fit(X, y)
    return clf

hdr, learn, tests = reader("heart_data.csv")
X = *(row[:-1] for row in learn), # iris.data
y = *(row[-1] for row in learn), # iris.target

recs = []
Trecs = []
for crit in range(3):
    for md in range(1, 21):
        for mln in range(2, 101):
            print(crit, mln, md)
            clf = learner(X, y, crit, mln, md)
            alg = crit, mln, md, clf

            #print("~" * 77)
            #print("Тот же набор")
            y2 = clf.predict(X)
            rec = checker(y, y2, alg)

            #print("~" * 77)
            #print("Тестовый же набор")
            tX = *(row[:-1] for row in tests), # iris.data
            ty = *(row[-1] for row in tests), # iris.target
            y2 = clf.predict(tX)
            Trec = checker(ty, y2, alg)

            recs.append(rec)
            Trecs.append(Trec)

Lrecs = sorted(recs, key = lambda rec: rec[3], reverse = True)
Trecs = sorted(Trecs, key = lambda rec: rec[3], reverse = True)

for recs, name, name2 in ((Lrecs, "обучении", "L"), (Trecs, "тестировании", "T")):
    print("~" * 70)
    print(f"10 лучших случаев на {name}:")
    n = 0
    for L, yeah, errs, acc, (crit, mln, md, clf) in recs[:10]:
        crit = ("gini", "entropy", "log_loss")[crit]
        print(f"верно: {yeah} | неверно: {errs} | аккуратность: {acc * 100 :.3f}% | alg:", (crit, mln, md))
        save_img(f"{name2}{n}", clf, hdr)
        n += 1

""" Последний вывод:

2 96 20
всего записей: 417 | верно: 416 | неверно: 1 | аккуратность: 99.760%
всего записей: 180 | верно: 134 | неверно: 46 | аккуратность: 74.444%
2 97 20
всего записей: 417 | верно: 416 | неверно: 1 | аккуратность: 99.760%
всего записей: 180 | верно: 136 | неверно: 44 | аккуратность: 75.556%
2 98 20
всего записей: 417 | верно: 414 | неверно: 3 | аккуратность: 99.281%
всего записей: 180 | верно: 131 | неверно: 49 | аккуратность: 72.778%
2 99 20
всего записей: 417 | верно: 417 | неверно: 0 | аккуратность: 100.000%
всего записей: 180 | верно: 139 | неверно: 41 | аккуратность: 77.222%
2 100 20
всего записей: 417 | верно: 415 | неверно: 2 | аккуратность: 99.520%
всего записей: 180 | верно: 137 | неверно: 43 | аккуратность: 76.111%
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
10 лучших случаев на обучении:
верно: 417 | неверно: 0 | аккуратность: 100.000% | alg: ('gini', 79, 12)
верно: 417 | неверно: 0 | аккуратность: 100.000% | alg: ('gini', 89, 12)
верно: 417 | неверно: 0 | аккуратность: 100.000% | alg: ('gini', 95, 12)
верно: 417 | неверно: 0 | аккуратность: 100.000% | alg: ('gini', 77, 13)
верно: 417 | неверно: 0 | аккуратность: 100.000% | alg: ('gini', 84, 13)
верно: 417 | неверно: 0 | аккуратность: 100.000% | alg: ('gini', 96, 13)
верно: 417 | неверно: 0 | аккуратность: 100.000% | alg: ('gini', 77, 14)
верно: 417 | неверно: 0 | аккуратность: 100.000% | alg: ('gini', 82, 14)
верно: 417 | неверно: 0 | аккуратность: 100.000% | alg: ('gini', 83, 14)
верно: 417 | неверно: 0 | аккуратность: 100.000% | alg: ('gini', 86, 14)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
10 лучших случаев на тестировании:
верно: 146 | неверно: 34 | аккуратность: 81.111% | alg: ('gini', 15, 7)
верно: 146 | неверно: 34 | аккуратность: 81.111% | alg: ('gini', 29, 9)
верно: 146 | неверно: 34 | аккуратность: 81.111% | alg: ('gini', 49, 18)
верно: 146 | неверно: 34 | аккуратность: 81.111% | alg: ('gini', 60, 20)
верно: 145 | неверно: 35 | аккуратность: 80.556% | alg: ('gini', 6, 3)
верно: 145 | неверно: 35 | аккуратность: 80.556% | alg: ('gini', 7, 3)
верно: 145 | неверно: 35 | аккуратность: 80.556% | alg: ('gini', 8, 3)
верно: 145 | неверно: 35 | аккуратность: 80.556% | alg: ('gini', 9, 3)
верно: 145 | неверно: 35 | аккуратность: 80.556% | alg: ('gini', 10, 3)
верно: 145 | неверно: 35 | аккуратность: 80.556% | alg: ('gini', 11, 3)
"""
