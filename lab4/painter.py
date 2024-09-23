from math import hypot
from random import randint
import warnings

import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw, ImageTk

print("open keras model loader")
from keras.models import load_model # pip install keras tensorflow
from keras.datasets import mnist
print("load model")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = load_model("trained_model.keras")
model2 = load_model("trained_model2.keras")
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("OK", model)

def DrawingApp(master):
    # https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
    def draw_bars(arr):
        sX, sY = barsXY
        bars_image.paste('white', (0, 0, sX, sY))
        for i in range(10):
            x, x2 = i * sX // 10, (i + 1) * sX // 10
            y = (1 - arr[i]) * sY
            bars_draw.rectangle([(x, y), (x2, sY)], "green")
            bars_draw.text((x + 5, 5), str(i), "red")
        for i in range(1, 10):
            x = i * sX // 10
            bars_draw.line([(x, 0), (x, sY)], "blue", 1)
        photos[3] = ImageTk.PhotoImage(image = bars_image)
        bars_canvas.create_image(0, 0, image = photos[3], anchor = tk.NW)

    def brain():
        resized = small_image.resize((28, 28), Image.LANCZOS)
        arr = 1 - np.array(resized) / 255
        arr = arr.reshape(1, -1) # shape -> (1, 784)
        res = (model2 if useModel2.get() else model).predict(arr, verbose=0)[0]
        #res -= min(res)
        #res /= max(res)
        #print(*("%.6f" % i for i in res))
        draw_bars(res)

    def update_brush_size(size):
        nonlocal brush_size
        brush_size = int(size)

        brush_image.paste(255, (0, 0, maxBSsize, maxBSsize))
        brush_draw.circle((maxBS + 6, maxBS + 6), brush_size, fill = 0)
        photos[2] = ImageTk.PhotoImage(image = brush_image)
        brush_canvas.create_image(0, 0, image = photos[2], anchor = tk.NW)

    def release(e):
        nonlocal prevXY
        prevXY = None

    def random_img():
        nonlocal small_image, large_image, large_draw

        resized = Image.fromarray(255 - X_train[randint(0, len(X_train) - 1)], "L")
        small_image = resized.resize((canvasXY, canvasXY), Image.NEAREST)
        large_image = resized.resize((canvasXY, canvasXY), Image.LANCZOS)
        large_draw = ImageDraw.Draw(large_image)

        redraw(False)

    def redraw(calc_small_image = True):
        nonlocal small_image, large_image, large_draw

        if calc_small_image:
            resized = large_image.resize((28, 28), Image.LANCZOS)
            small_image = resized.resize((canvasXY, canvasXY), Image.NEAREST)
            if repeater.get():
                large_image = resized.resize((canvasXY, canvasXY), Image.LANCZOS)
                large_draw = ImageDraw.Draw(large_image)

        photos[0] = ImageTk.PhotoImage(image = large_image)
        photos[1] = ImageTk.PhotoImage(image = small_image)
        large_canvas.create_image(0, 0, image = photos[0], anchor = tk.NW)
        small_canvas.create_image(0, 0, image = photos[1], anchor = tk.NW)

        brain()

    photos = [None] * 4
    def paint(x, y, radius, color):
        nonlocal prevXY

        large_draw.circle((x, y), radius, fill = color)
        if prevXY:
            px, py = prevXY
            large_draw.line([(px, py), (x, y)], color, radius * 2 + 1, "curve")
        prevXY = x, y

        redraw()

    def new_canvas(side):
        canvas = tk.Canvas(master, width = canvasXY, height = canvasXY, bg = 'white')
        canvas.pack(side = side)
        canvas.bind("<B1-Motion>", lambda e: paint(e.x, e.y, brush_size, 0))
        canvas.bind("<B3-Motion>", lambda e: paint(e.x, e.y, brush_size * 3, 255))
        canvas.bind("<ButtonRelease-1>", release)
        canvas.bind("<ButtonRelease-3>", release)
        return canvas

    def clear():
        large_image.paste(255, (0, 0, canvasXY, canvasXY))
        redraw()

    master.title("Paint Net AI 3000-бесконечностей супер-пупер-делюкс-эдишн")

    brush_size = 20
    maxBS = 32
    maxBSsize = maxBS * 2 + 9
    canvasXY = 512
    prevXY = None
    barsXY = 130, 300

    large_canvas = new_canvas(tk.LEFT)
    small_canvas = new_canvas(tk.RIGHT)

    large_image = Image.new("L", (canvasXY, canvasXY), 255)
    small_image = Image.new("L", (28, 28), 255)

    large_draw = ImageDraw.Draw(large_image)
    #small_draw = ImageDraw.Draw(small_image)



    scaler = tk.Scale(master, from_=1, to=maxBS, orient=tk.HORIZONTAL, label="Размер кисти", command = update_brush_size)
    scaler.set(brush_size)
    scaler.pack()

    brush_canvas = tk.Canvas(master, width = maxBSsize, height = maxBSsize, bg = 'white')
    brush_canvas.pack()
    brush_image = Image.new("L", (maxBSsize, maxBSsize), 255)
    brush_draw = ImageDraw.Draw(brush_image)

    button = tk.Button(master, text="clear", command = clear)
    button.pack()

    button = tk.Button(master, text="random", command = random_img)
    button.pack()

    bars_canvas = tk.Canvas(master, width = barsXY[0], height = barsXY[1], bg = 'white')
    bars_canvas.pack()
    bars_image = Image.new("RGB", barsXY, 'white')
    bars_draw = ImageDraw.Draw(bars_image)

    repeater = tk.IntVar()
    useModel2 = tk.IntVar()
    checkbutton = tk.Checkbutton(text = "repeater", variable = repeater)
    checkbutton.pack(anchor=tk.NW)
    checkbutton2 = tk.Checkbutton(text = "model2", variable = useModel2, command = redraw)
    checkbutton2.pack(anchor=tk.NW)

    redraw()





if __name__ == "__main__":
    root = tk.Tk()
    DrawingApp(root)
    root.mainloop()
