from sklearn import linear_model, impute
import matplotlib.pyplot as plt
import warnings
from fractions import Fraction as frac
from random import shuffle
import os

def save_img(name, estimator):
    n_features = len(estimator.coef_)
    plt.figure(figsize=(12, 6))
    plt.bar(range(n_features), estimator.coef_, color="blue")
    plt.title("Коэффициенты Lasso для каждого признака")
    plt.xlabel("Признаки")
    plt.ylabel("Коэффициенты")
    plt.xticks(range(n_features), hdr[:n_features], rotation = 15, fontsize = None)  # Подписи для признаков
    plt.grid()
    plt.savefig(name)
    plt.close()

def save_img2(name, X, y, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.7, color="blue")
    plt.plot([min(y), max(y)], [min(y), max(y)], color="red", linestyle="--", linewidth=2, label="Идеальная линия")
    plt.title("Предсказанные значения против фактических значений")
    plt.xlabel("Фактические значения")
    plt.ylabel("Предсказанные значения (Lasso)")
    plt.legend()
    plt.grid()
    plt.savefig(name)
    plt.close()

def save_img3(name, alphas, accuracies, L, R):
    plt.figure(figsize=(20, 12))
    plt.plot(alphas, accuracies, marker='o', linestyle='-', color='b', alpha=0.2, markersize=8)
    plt.title('Влияние alpha на точность модели')
    plt.xlabel('Alpha')
    plt.ylabel('Точность')
    plt.grid(axis='y')
    #plt.xticks(alphas)
    #plt.ylim(min(accuracies) - 0.05, max(accuracies) + 0.05)
    plt.axhline(y=max(accuracies), color='r', linestyle='--', label="Максимальная точность")
    plt.axvline(x=L, color='y', linestyle=':')
    plt.axvline(x=R, color='y', linestyle=':')
    plt.legend()
    plt.savefig(name)
    #plt.show()
    plt.close()

# вариант 17
# задание: (17 % 4) + 1 = 2 (LASSO)

#class lol:
#    def __format__(self, f):
#        print("F:", repr(f), len(f)) # F: ' cat ' 5 (пробелы тоже учитывает)
#        return "woof"
#print(f"meow: '{lol(): cat }'")

def norm(table):
    #print(table[0])
    rg = range(1, len(table[0]) - 1) # отсекаем тип и качество
    for feature in rg:
        arg = *(item for item in (value[feature] for value in table) if item is not None),
        mi, ma = min(arg), max(arg)
        dMiMa = ma - mi
        #print(mi, ma)
        for row in table:
            zn = row[feature]
            if zn is not None: row[feature] = (zn - mi) / dMiMa
    #for row in table[:10]: print(row)

def reader(name):
    alls = []
    app = alls.append
    NaN = None # float("NaN") в целом одно и тоже самое для импутора
    with open(name, "rb") as file:
        hdr = file.readline()[:-1].decode("utf-8").split(",")
        for line in file.readlines():
            value = [NaN if not item else item[0] == 114 if i == 0 else (int if i == 12 else float)(item) for i, item in enumerate(line.split(b","))]
            app(value)
    norm(alls)
    # norm(alls) # все mi и ma будут 0 и 1, что ни к чему не приведёт,
    # следовательно, первый norm работает правильно

    containsNaN = []
    for i in range(500):
        value = alls[i]
        if NaN in value:
            print(i, (*map(lambda x: round(x, 3) if x is not None else None, value),))
            containsNaN.append(i)

    imputer = impute.SimpleImputer(strategy='mean')
    alls = imputer.fit_transform(alls)
    print()
    for i in containsNaN: print((*map(lambda x: round(x, 3) if x is not None else None, alls[i]),))
    
    white = []
    red = []
    app = red.append, white.append
    for value in alls: app[round(value[0])](value)
    print(f"W: {len(white)}   R: {len(red)}   all: {len(white) + len(red)}")

    return white, red, hdr

def learner(data, alpha):
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    X = *(i[:-1] for i in data),
    y = *(i[-1] for i in data),
    #from sklearn.datasets import make_regression
    #X, y = make_regression(n_samples=100, n_features=12, noise=10, random_state=0)
    estimator = linear_model.Lasso(alpha = alpha)
    estimator.fit(X, y)
    return X, y, estimator

def checker(y, y2):
    assert len(y) == len(y2)
    errs = 0
    for a, b in zip(y, y2):
        if a != round(b): errs += 1    #Если бы у 'a' изначально не было округления (тоже самое, что и: abs(a - b) >= 0.5)
        #if abs(a - b) >= 1: errs += 1    Если бы у 'a' было округление, но до нас дошло только округлённое значение
        #if abs(a - b) >= 0.75: errs += 1
        #v = "YEAH" if a == b else "ERROR"
        #print(f"real: {a}, prediction: {b}, verdict: {v}")
    L = len(y)
    acc = (L - errs) / L
    return L, L - errs, errs, acc

def learner2(learn, tests, codename):
    if not os.path.exists(codename): os.mkdir(codename)

    L, R, steps = frac(0), frac(1, 10), 250
    for it in range(3):
        alphas = []
        accuracies = []
        dLR = R - L
        prev = None
        for i in range(steps + 1):
            alpha = L + dLR * i / steps
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X, y, estimator = learner(learn, float(alpha))
            if id(learn) != id(tests):
                X = *(i[:-1] for i in tests),
                y = *(i[-1] for i in tests),
            y_pred = estimator.predict(X)
            count, yeah, errs, acc = checker(y, y_pred)

            cur = i // (steps // 10)
            if prev != cur:
                print(f"\u03B1: {alpha} | всего: {count} | верно: {yeah} | неверно: {errs} | аккуратность: {acc * 100 :.3f}%")
                prev = cur
            alphas.append(alpha)
            accuracies.append(acc)

        Max, i = max(zip(accuracies, range(steps + 1)), key = lambda x: x[0])
        maxA = L + dLR * i / steps
        d = dLR / 8
        prevL, prevR = L, R
        print(L, R)
        L, R = maxA - d, maxA + d
        print(L, R)
        if L < prevL: L, R = prevL, R + (prevL - L)
        elif R > prevR: L, R = L - (R - prevR), prevR
        print(Max, i, "|=>", maxA, L, R)
        print()
        #save_img3(f"alpha_vs_accuracy_{it}_{L:.5g}-{R:.5g}.png", alphas, accuracies, L, R)
        save_img3(f"{codename}/alpha_vs_accuracy_{it}.png", alphas, accuracies, L, R)

    print(f"Идеальная \u03B1: {maxA} -> {float(maxA)}")
    X, y, estimator = learner(learn, float(maxA))
    if id(learn) != id(tests):
        X = *(i[:-1] for i in tests),
        y = *(i[-1] for i in tests),
    y_pred = estimator.predict(X)
    save_img(f"images/lasso_coefficients_{codename}.png", estimator)
    save_img2(f"images/lasso_predictions_{codename}.png", X, y, y_pred)
    count, yeah, errs, acc = checker(y, y_pred)
    return maxA, acc

white, red, hdr = reader("winequalityN.csv")
alls = white + red
shuffle(white)
shuffle(red)
shuffle(alls)

final = []

for data, codename, TisL in (
    (white, "WLL", True), # White Learn Learn
    (white, "WLT", False), # White Learn Tests
    (red, "RLL", True), # Red Learn Learn
    (red, "RLT", False), # Red Learn Tests
    (alls, "ALL", True), # Alls Learn Learn
    (alls, "ALT", False), # Alls Learn Tests
):
    if TisL: learn = tests = data
    else:
        L = len(data) * 7 // 10
        learn, tests = data[:L], data[L:]
    print("~" * 77)
    print(codename)
    print(f"Обучающая часть: {len(learn)}   Тестовая: {len(tests)}")
    alpha, acc = learner2(learn, tests, codename)
    final.append((codename, alpha, acc))
    print("~" * 77)

for codename, alpha, acc in final:
    assert 40000 % alpha.denominator == 0, alpha
    print(f"{codename} |=> \u03B1: {alpha} -> {float(alpha)} -> {alpha.numerator * 40000 // alpha.denominator}/40000 | acc: {acc * 100 :.3f}%")

""" Первый запуск
WLL |=> α: 2207/40000 -> 0.055175 -> 2207/40000 | acc: 82.489%
WLT |=> α: 7/125 -> 0.056 -> 2240/40000 | acc: 83.542%
RLL |=> α: 13/2500 -> 0.0052 -> 208/40000 | acc: 72.866%
RLT |=> α: 1/40000 -> 2.5e-05 -> 1/40000 | acc: 74.626%
ALL |=> α: 1/10000 -> 0.0001 -> 4/40000 | acc: 74.050%
ALT |=> α: 379/40000 -> 0.009475 -> 379/40000 | acc: 73.846%
"""

""" Второй запуск
WLL |=> α: 2207/40000 -> 0.055175 -> 2207/40000 | acc: 82.489%
WLT |=> α: 2073/40000 -> 0.051825 -> 2073/40000 | acc: 82.500%
RLL |=> α: 13/2500 -> 0.0052 -> 208/40000 | acc: 72.866%
RLT |=> α: 117/40000 -> 0.002925 -> 117/40000 | acc: 71.905%
ALL |=> α: 1/10000 -> 0.0001 -> 4/40000 | acc: 74.050%
ALT |=> α: 3/40000 -> 7.5e-05 -> 3/40000 | acc: 72.872%
"""

""" Третий запуск
WLL |=> α: 2207/40000 -> 0.055175 -> 2207/40000 | acc: 82.489%
WLT |=> α: 1901/40000 -> 0.047525 -> 1901/40000 | acc: 81.250%
RLL |=> α: 13/2500 -> 0.0052 -> 208/40000 | acc: 72.866%
RLT |=> α: 3/1000 -> 0.003 -> 120/40000 | acc: 74.082%
ALL |=> α: 1/10000 -> 0.0001 -> 4/40000 | acc: 74.050%
ALT |=> α: 3/10000 -> 0.0003 -> 12/40000 | acc: 74.667%
"""

""" Четвёртый запуск
WLL |=> α: 2207/40000 -> 0.055175 -> 2207/40000 | acc: 82.489%
WLT |=> α: 1099/20000 -> 0.05495 -> 2198/40000 | acc: 85.208%
RLL |=> α: 13/2500 -> 0.0052 -> 208/40000 | acc: 72.866%
RLT |=> α: 61/40000 -> 0.001525 -> 61/40000 | acc: 72.993%
ALL |=> α: 1/10000 -> 0.0001 -> 4/40000 | acc: 74.050%
ALT |=> α: 1/20000 -> 5e-05 -> 2/40000 | acc: 74.462%
"""

""" Пятый запуск
WLL |=> α: 2207/40000 -> 0.055175 -> 2207/40000 | acc: 82.489%
WLT |=> α: 989/20000 -> 0.04945 -> 1978/40000 | acc: 82.917%
RLL |=> α: 13/2500 -> 0.0052 -> 208/40000 | acc: 72.866%
RLT |=> α: 3/40000 -> 7.5e-05 -> 3/40000 | acc: 73.333%
ALL |=> α: 1/10000 -> 0.0001 -> 4/40000 | acc: 74.050%
ALT |=> α: 17/8000 -> 0.002125 -> 85/40000 | acc: 74.308%
"""
