# Для установки Jupyter Notebook
# pip install notebook
# Зависимости: Installing collected packages: webencodings, wcwidth, pywin32, pure-eval, fastjsonschema, websocket-client, webcolors, uri-template, types-python-dateutil, traitlets, tornado, tinycss2, soupsieve, sniffio, send2trash, rpds-py, rfc3986-validator, rfc3339-validator, pyzmq, pyyaml, pywinpty, python-json-logger, psutil, prompt-toolkit, prometheus-client, platformdirs, parso, pandocfilters, overrides, nest-asyncio, mistune, jupyterlab-pygments, jsonpointer, json5, httpcore, fqdn, executing, defusedxml, decorator, debugpy, bleach, babel, async-lru, asttokens, terminado, stack-data, referencing, matplotlib-inline, jupyter-core, jedi, comm, beautifulsoup4, arrow, argon2-cffi-bindings, anyio, jupyter-server-terminals, jupyter-client, jsonschema-specifications, isoduration, ipython, httpx, argon2-cffi, jsonschema, ipykernel, nbformat, nbclient, jupyter-events, nbconvert, jupyter-server, notebook-shim, jupyterlab-server, jupyter-lsp, jupyterlab, notebook
# Запуск: jupyter notebook
# jupyter notebook --version
# 7.2.2

from keras.datasets import mnist # pip install keras tensorflow
import matplotlib.pyplot as plt # pip install matplotlib
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, GaussianDropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

early_stopping = EarlyStopping(monitor='accuracy', patience=float("inf"), verbose=1, mode='max', baseline=0.9999, restore_best_weights=True)
# - monitor: Метрическая величина, которую вы хотите отслеживать. В данном случае вы можете использовать `accuracy`.
# - patience: Количество эпох, которое нужно ждать, если метрика не улучшается, прежде чем остановить обучение. Если вы установите его в 0, обучение остановится сразу при достижении точности 0.9999.
# - verbose: Уровень отображения информации о процессе остановки. Установив его в 1, вы получите текстовый вывод. 
# - mode: Режим отслеживания. 'max' указывает, что нужно остановить обучение, если точность достигает максимума.
# - baseline: Начальное значение для метрики. В данном случае мы устанавливаем его в 0.9999, чтобы остановить обучение, если метрика достигает этого значения.
# - restore_best_weights: Если `True`, то веса модели будут восстановлены на уровне, который соответствует лучшему значению `monitor`.



# Вариант заданий: Распознавание цифр на изображении (MNIST digits classification dataset)
# Нейросеть должна состоять из четырех полносвязных слоёв,
# обязательное использование GaussianDropout,
# в качестве оптимизатора использовать SGD

(X_train, y_train), (X_test, y_test) = mnist.load_data()

def check1():
    print(X_train[0])
    print(y_train[:100])
    plt.imshow(X_train[0], cmap="gray");
    plt.show()

def check2():
    num_classes = 60
    rows = (num_classes + 9) // 10
    cols = min(10, num_classes)
    fig, axs = plt.subplots(rows, cols, figsize=(20,20))
    #print([y_train == i for i in range(10)])
    for i in range(num_classes):
        sample = X_train[i]
        ax = axs[*divmod(i, 10)]
        ax.imshow(sample, cmap="gray")
        ax.set_title(f"Label:{y_train[i]}")
        ax.axis('off')
    plt.show()

# версия самих разработчиков mnist
def mnist_version_model_builder():
    model = Sequential()

    model.add(Input(shape=(784,)))
    model.add(Dense(units=128, activation="relu")) # ReLU -> сигма(x) = max(0, x)
    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(units=10, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# моя версия под собственный вариант задания
def my_version_model_builder():
    model = Sequential()

    model.add(Input(shape=(784,)))
    model.add(Dense(units=512, activation="relu"))
    model.add(Dropout(0.5)) # GaussianDropout = Dropout(0.5)
    model.add(Dense(units=256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=10, activation="softmax"))  # Выходной слой

    sgd = SGD(learning_rate = 0.01, momentum = 0.9, nesterov = True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    return model

def learner(X_train, y_train, X_test, y_test):
    print("start")
    X_train = X_train/255.0
    X_test = X_test/255.0
    print(X_train.shape)
    X_train = X_train.reshape(X_train.shape[0], -1)
    print(X_train.shape)
    X_test = X_test.reshape(X_test.shape[0], -1)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    #model = mnist_version_model_builder()
    model = my_version_model_builder()
    model.summary()

    #import os
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    BATCH_SIZE = 512
    epochs = 11
    while True:
        history = model.fit(x=X_train, y=y_train, batch_size = BATCH_SIZE, epochs = epochs, verbose=2, callbacks=[early_stopping])
        acc = max(history.history["accuracy"])
        if acc > 0.999: train_loss, train_acc = model.evaluate(X_train, y_train, verbose=2)
        else: train_loss = train_acc = -1 
        print(history.history, acc, train_loss, train_acc)
        if acc > 0.9999 or train_acc > 0.9999: break

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=2)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

    print(f"Train Loss: {train_loss}, Train Accuracy: {train_acc}")
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

learner(X_train, y_train, X_test, y_test)
input("Enter...")
