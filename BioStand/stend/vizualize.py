import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

root = tk.Tk()
root.title("Визуализация данных из CSV")
root.geometry("400x200")


def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)

        popup = tk.Toplevel()
        popup.title("Выберите данные для осей")

        x_label = tk.Label(popup, text="Ось X:")
        x_label.pack()
        x_var = tk.StringVar(popup)
        x_var.set(df.columns[0])
        x_dropdown = tk.OptionMenu(popup, x_var, *df.columns)
        x_dropdown.pack()

        y_label = tk.Label(popup, text="Ось Y:")
        y_label.pack()
        y_var = tk.StringVar(popup)
        y_var.set(df.columns[1])
        y_dropdown = tk.OptionMenu(popup, y_var, *df.columns)
        y_dropdown.pack()

        def update_graph():
            x_data = df[x_var.get()]
            y_data = df[y_var.get()]

            Q1 = np.percentile(y_data, 25)
            Q3 = np.percentile(y_data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            filtered_indices = (y_data >= lower_bound) & (y_data <= upper_bound)
            filtered_x = x_data[filtered_indices]
            filtered_y = y_data[filtered_indices]

            # Создаем функцию для обновления графика с определенной частотой
            def animate(i):
                plt.cla()  # Очищаем текущий график
                plt.plot(filtered_x[:i], filtered_y[:i])  # Отображаем только часть данных
                plt.xlabel(x_var.get())
                plt.ylabel(y_var.get())
                plt.title("Динамическая отрисовка графика")

            # Создаем анимацию с обновлением каждые 100 миллисекунд
            anim = FuncAnimation(plt.gcf(), animate, frames=len(filtered_x), interval=15)
            plt.show()

        plot_button = tk.Button(popup, text="Динамическая отрисовка графика", command=update_graph)
        plot_button.pack()

        popup.mainloop()


file_button = tk.Button(root, text="Выбрать файл", command=open_file)
file_button.pack()

root.mainloop()
