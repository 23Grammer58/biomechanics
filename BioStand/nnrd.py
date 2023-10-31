import tkinter as tk
from tkinter import filedialog
import nrrd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("NRRD Files", "*.nrrd")])
    if file_path:
        try:
            data, header = nrrd.read(file_path)
            show_image(data)
        except Exception as e:
            error_label.config(text=f"Ошибка: {str(e)}")


def show_image(data):
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap='gray')
    plt.axis('off')

    canvas = FigureCanvasTkAgg(plt.gcf(), master=frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()


# Создание главного окна
root = tk.Tk()
root.title("NRRD Файловый Визуализатор")

# Создание кнопки для загрузки файла
open_button = tk.Button(root, text="Открыть NRRD файл", command=open_file)
open_button.pack(pady=10)

# Создание фрейма для отображения изображения
frame = tk.Frame(root)
frame.pack()

# Создание метки для вывода ошибок
error_label = tk.Label(root, fg="red")
error_label.pack()

# Запуск главного цикла
root.mainloop()
