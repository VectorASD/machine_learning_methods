from random import shuffle
from math import hypot

from PIL import Image, ImageDraw

def img_gen(data, N):
    w = h = 4096
    dot_r = 7
    w = dot_r * 7 * 6
    min_x = min_y = float("inf")
    max_x = max_y = -float("inf")
    for x, y, C in data:
        min_x, min_y = min(min_x, x), min(min_y, y)
        max_x, max_y = max(max_x, x), max(max_y, y)
    dx, dy = max_x - min_x, max_y - min_y
    w2, h2 = w - dot_r * 2 - 1, h - dot_r * 2 - 1
    print(min_x, min_y, max_x, max_y)

    img = Image.new("RGB", (w, h), "#eeffad")
    draw = ImageDraw.Draw(img)
    for x, y, C in data:
        ix = (x - min_x) * w2 // dx + dot_r
        iy = (y - min_y) * h2 // dy + dot_r
        draw.circle((ix, iy), dot_r, fill=("#adeeff" if C else "#ffad80"), outline=("#4080ff" if C else "#ff8040"), width=1)
    img.save(f"res_{N}.png")

# вариант 17
# тип классификатора: 17 % 3 + 1 = 3
# ядро: (17 * 6 + 13) % 8 % 3 + 1 = 1

# Метод парзеновского окна с относительным размером окна
# Q–квартическое K(x) = 15/16 * (1 - r**2)**2 [r <= 1]

def reader(name):
    C0 = []
    C1 = []
    with open(name, "r") as file:
        hdr = file.readline()
        #print(hdr)
        for line in file.readlines():
            row = *map(int, line.split(",")),
            (C1 if row[2] else C0).append(row)

    L0, L1 = len(C0), len(C1)
    print("Количество первых классов:", L0)
    print("Количество вторых классов:", L1)
    count = min(L0, L1, (L0 + L1) // 3)
    print("Оптимальное количество каждого класса:", count) # удивился 3333, а потом вспомнил, что там ровно 10000 строчек

    shuffle(C0) # не особо справляется перемешивание, видимо ;'-}
    shuffle(C1)
    learn = C0[:count] + C1[:count]
    tests = C0[count:] + C1[count:]
    shuffle(learn)
    #shuffle(tests)

    print("Всего обучающих:", len(learn), "тестовых:", len(tests))
    return learn, tests

def square_core(len, r):
    if len >= r: return 0
    return 15/16 * (1 - (len / r) ** 2) ** 2
    # L / R выдаёт число от 0 до 1

def window_test(learn, x, y, r):
    C0w = C1w = 0
    for x2, y2, C in learn:
        len = hypot(x - x2, y - y2)
        w = square_core(len, r)
        if C: C1w += w
        else: C0w += w
    try: return C1w / (C0w + C1w)
    except ZeroDivisionError: return # окно пустое, либо все точки на его границе

def tester(data, r):
    errs = 0
    for x, y, C in data:
        isC1 = window_test(data, x, y, r)
        if isC1 is not None and (isC1 > 0.5) != (C == 1): errs += 1
    r = "%.6f" % r
    print(f"r: {r} | errs: {errs}")
    return errs

def learner(learn, min_r, max_r, steps):
    dr = max_r - min_r
    s1 = steps - 1
    min_errs = float("inf")
    yeah_r = min_r
    for i in range(steps):
        r = min_r + dr * i / s1
        errs = tester(learn, r)
        if errs <= min_errs:
            min_errs = errs
            yeah_r = r
        print("%.6f" % r, errs, yeah_r)
    return yeah_r

def learner2(learn, min_r):
    R = int(min_r) # здесь R -> right
    if not R: R = 1
    while True:
        if tester(learn, R): break
        R *= 2
        assert R < 1e30
    L = R / 2
    while abs(R - L) >= 0.000001:
        mid = (L + R) / 2
        #print(L, mid, R)
        if tester(learn, mid): R = mid
        else: L = mid
    return L, R

N = 5
learn, tests = reader(f"data{N}.csv")
#img_gen(learn + tests, f"{N}a")

#for i in range(101): print(square_core(i / 100 * 5, 5))
#for r in range(10, 100001, 1000): print(r, window_test(learn, 0, 0, r))
#learner(learn, 1, 10, 100)

#learn = learn[:1000]
L = len(learn)
pairlen = lambda a, b: hypot(a[0] - b[0], a[1] - b[1])
min_R = min(pairlen(learn[i], learn[j]) for i in range(L - 1) for j in range(i + 1, L))

print("min R:", min_R)

yeah, nop = learner2(learn, min_R)
print("best R: %.6f" % yeah)
print("error R: %.6f" % nop)

print("(best R) ошибок на тестах:", tester(tests, yeah))
print("(error R) ошибок на тестах:", tester(tests, nop))

""" N = 1

Количество первых классов: 4574
Количество вторых классов: 5426
Оптимальное количество каждого класса: 3333
Всего обучающих: 6666 тестовых: 3334
min R: 0.0
r: 1.000000 | errs: 0
r: 2.000000 | errs: 1
r: 1.500000 | errs: 0
r: 1.750000 | errs: 0
r: 1.875000 | errs: 1
r: 1.812500 | errs: 0
r: 1.843750 | errs: 0
r: 1.859375 | errs: 1
r: 1.851562 | errs: 1
r: 1.847656 | errs: 0
r: 1.849609 | errs: 1
r: 1.848633 | errs: 1
r: 1.848145 | errs: 1
r: 1.847900 | errs: 1
r: 1.847778 | errs: 1
r: 1.847717 | errs: 0
r: 1.847748 | errs: 0
r: 1.847763 | errs: 1
r: 1.847755 | errs: 0
r: 1.847759 | errs: 1
r: 1.847757 | errs: 0
r: 1.847758 | errs: 0
best R: 1.847758
error R: 1.847759
r: 1.847758 | errs: 0
(best R) ошибок на тестах: 0
r: 1.847759 | errs: 0
(error R) ошибок на тестах: 0
"""

""" N = 2

Количество первых классов: 2446
Количество вторых классов: 7554
Оптимальное количество каждого класса: 2446
Всего обучающих: 4892 тестовых: 5108
min R: 0.0
r: 1.000000 | errs: 0
r: 2.000000 | errs: 0
r: 4.000000 | errs: 4
r: 3.000000 | errs: 0
r: 3.500000 | errs: 2
r: 3.250000 | errs: 0
r: 3.375000 | errs: 0
r: 3.437500 | errs: 2
r: 3.406250 | errs: 2
r: 3.390625 | errs: 2
r: 3.382812 | errs: 0
r: 3.386719 | errs: 0
r: 3.388672 | errs: 2
r: 3.387695 | errs: 2
r: 3.387207 | errs: 2
r: 3.386963 | errs: 0
r: 3.387085 | errs: 2
r: 3.387024 | errs: 0
r: 3.387054 | errs: 2
r: 3.387039 | errs: 0
r: 3.387047 | errs: 0
r: 3.387051 | errs: 0
r: 3.387053 | errs: 0
r: 3.387053 | errs: 0
best R: 3.387053
error R: 3.387054
r: 3.387053 | errs: 0
(best R) ошибок на тестах: 0
r: 3.387054 | errs: 0
(error R) ошибок на тестах: 0
"""

""" N = 3

Количество первых классов: 7012
Количество вторых классов: 2988
Оптимальное количество каждого класса: 2988
Всего обучающих: 5976 тестовых: 4024
min R: 0.0
r: 1.000000 | errs: 0
r: 2.000000 | errs: 0
r: 4.000000 | errs: 9
r: 3.000000 | errs: 2
r: 2.500000 | errs: 0
r: 2.750000 | errs: 1
r: 2.625000 | errs: 1
r: 2.562500 | errs: 0
r: 2.593750 | errs: 0
r: 2.609375 | errs: 0
r: 2.617188 | errs: 1
r: 2.613281 | errs: 1
r: 2.611328 | errs: 0
r: 2.612305 | errs: 0
r: 2.612793 | errs: 0
r: 2.613037 | errs: 0
r: 2.613159 | errs: 1
r: 2.613098 | errs: 0
r: 2.613129 | errs: 1
r: 2.613113 | errs: 0
r: 2.613121 | errs: 0
r: 2.613125 | errs: 0
r: 2.613127 | errs: 1
r: 2.613126 | errs: 0
best R: 2.613126
error R: 2.613127
r: 2.613126 | errs: 0
(best R) ошибок на тестах: 0
r: 2.613127 | errs: 0
(error R) ошибок на тестах: 0
"""

""" N = 4

Количество первых классов: 7985
Количество вторых классов: 2015
Оптимальное количество каждого класса: 2015
Всего обучающих: 4030 тестовых: 5970
min R: 0.0
r: 1.000000 | errs: 0
r: 2.000000 | errs: 0
r: 4.000000 | errs: 6
r: 3.000000 | errs: 0
r: 3.500000 | errs: 4
r: 3.250000 | errs: 2
r: 3.125000 | errs: 2
r: 3.062500 | errs: 2
r: 3.031250 | errs: 2
r: 3.015625 | errs: 0
r: 3.023438 | errs: 0
r: 3.027344 | errs: 2
r: 3.025391 | errs: 0
r: 3.026367 | errs: 0
r: 3.026855 | errs: 0
r: 3.027100 | errs: 2
r: 3.026978 | errs: 2
r: 3.026917 | errs: 0
r: 3.026947 | errs: 2
r: 3.026932 | errs: 2
r: 3.026924 | errs: 0
r: 3.026928 | errs: 2
r: 3.026926 | errs: 2
r: 3.026925 | errs: 0
best R: 3.026925
error R: 3.026926
r: 3.026925 | errs: 0
(best R) ошибок на тестах: 0
r: 3.026926 | errs: 0
(error R) ошибок на тестах: 0
"""

""" N = 5

Количество первых классов: 1468
Количество вторых классов: 8532
Оптимальное количество каждого класса: 1468
Всего обучающих: 2936 тестовых: 7064
min R: 0.0
r: 1.000000 | errs: 0
r: 2.000000 | errs: 0
r: 4.000000 | errs: 1
r: 3.000000 | errs: 0
r: 3.500000 | errs: 1
r: 3.250000 | errs: 1
r: 3.125000 | errs: 1
r: 3.062500 | errs: 1
r: 3.031250 | errs: 1
r: 3.015625 | errs: 0
r: 3.023438 | errs: 0
r: 3.027344 | errs: 1
r: 3.025391 | errs: 0
r: 3.026367 | errs: 0
r: 3.026855 | errs: 0
r: 3.027100 | errs: 1
r: 3.026978 | errs: 1
r: 3.026917 | errs: 0
r: 3.026947 | errs: 1
r: 3.026932 | errs: 1
r: 3.026924 | errs: 0
r: 3.026928 | errs: 1
r: 3.026926 | errs: 1
r: 3.026925 | errs: 0
best R: 3.026925
error R: 3.026926
r: 3.026925 | errs: 0
(best R) ошибок на тестах: 0
r: 3.026926 | errs: 0
(error R) ошибок на тестах: 0
"""

"""
Выводы: либо мне так очень круто везёт, что в learn ситуации всегда хуже, чем в tests
т.е. "(error R) ошибок на тестах" всегда даёт 0
либо подбор одинакового количество и тех, и тех классов делает магию, а моя идея,
заложенная в learner2 - идеальна для данного типа задач.

Тем ни менее подобраны около идеальные значения радиуса:
data1.csv: 1.847758
data2.csv: 3.387053
data3.csv: 2.613126
data4.csv: 3.026925
data5.csv: 3.026925

Кстати, последние 2 случая дали одинаковые значения, хоть и диагональ
разделения двух классов на res_4a.png и res_5a.png совершенно разные

Я надеялся, что рисунки будут интереснее, с "пятнами", ежели, просто диагонали,
сделанные скорее всего для 4 лабораторной. Их поиском занимается самая маленькая
версия перцептрона с нейроном смещения
"""
