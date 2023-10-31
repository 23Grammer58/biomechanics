import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import matplotlib.pyplot as plt

data = None

def load_csv():
    global data
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        data = pd.read_csv(file_path)
        x_columns = list(data.columns)
        combo_x['values'] = x_columns
        combo_x.current(0)  # Устанавливаем первый столбец по умолчанию для оси X
        update_y_columns()  # Обновляем список столбцов для оси Y
        update_histogram_columns()  # Обновляем список столбцов для гистограммы
        return data
    else:
        return None

def remove_outliers(data, y_axis, threshold=100):
    new_data = data.copy()
    for i in range(1, len(new_data)):
        if new_data.iloc[i][y_axis] > threshold * new_data.iloc[i - 1][y_axis]:
            new_data.at[i, y_axis] = new_data.at[i - 1, y_axis]
    return new_data

def polynomial_approximation(data, x_column, y_column, degree=1):
    x = data[x_column]
    y = data[y_column]
    coeffs = np.polyfit(x, y, degree)
    return np.polyval(coeffs, x)

def update_histogram_columns():
    global data
    if data is not None:
        columns = list(data.columns)
        combo_histogram['values'] = columns
        combo_histogram.current(0)  # Устанавливаем первый столбец по умолчанию для гистограммы

def update_y_columns():
    x_selected = combo_x.get()
    y_columns = [col for col in data.columns if col != x_selected]
    combo_y['values'] = y_columns
    combo_y.current(0)  # Устанавливаем первый столбец по умолчанию для оси Y

def plot_graph():
    global data
    if data is not None:
        x_axis = combo_x.get()
        y_axis = combo_y.get()

        filtered_data = data.copy()
        if filter_var.get():
            threshold = float(entry_threshold.get())
            filtered_data = remove_outliers(filtered_data, y_axis, threshold)

        plt.figure(figsize=(8, 6))
        plt.scatter(filtered_data[x_axis], filtered_data[y_axis], label="Filtered Data")
        plt.scatter(data[x_axis], data[y_axis], label="Original Data", alpha=0.5)

        if approx_var.get():
            plt.plot(filtered_data[x_axis], polynomial_approximation(filtered_data, x_axis, y_axis, degree=1), 'r',
                     label="Polynomial Approximation")

        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title("Data Visualization")
        plt.legend()
        plt.show()

        filtered_data.to_csv("StendVHB491012mm0-2.csv", index=False)

def save_csv():
    global data
    if data is not None:
        filtered_data = data.copy()
        if filter_var.get():
            threshold = float(entry_threshold.get())
            filtered_data = remove_outliers(data, combo_y.get(), threshold)

        filtered_data.to_csv("StendVHB491012mm0-2.csv", index=False)
        print("Сохранено успешно!")

def plot_histogram():
    global data
    if data is not None:
        plt.figure(figsize=(8, 6))
        column = combo_histogram.get()
        plt.hist(data[column], bins='auto')
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {column}")
        plt.show()

# Создание GUI
root = tk.Tk()
root.title("Выберите CSV файл, оси, фильтр и аппроксиматор")

frame = tk.Frame(root)
frame.pack()

button_load = tk.Button(frame, text="Загрузить CSV", command=load_csv)
button_load.pack(side=tk.LEFT, padx=5, pady=5)

label_x = tk.Label(frame, text="Выберите ось X:")
label_x.pack(side=tk.LEFT)

combo_x = ttk.Combobox(frame, width=20)
combo_x.pack(side=tk.LEFT, padx=5, pady=5)

label_y = tk.Label(frame, text="Выберите ось Y:")
label_y.pack(side=tk.LEFT)

combo_y = ttk.Combobox(frame, width=20)
combo_y.pack(side=tk.LEFT, padx=5, pady=5)

frame_buttons = tk.Frame(root)
frame_buttons.pack()

button_plot = tk.Button(frame_buttons, text="Отрисовать график", command=plot_graph)
button_plot.pack(side=tk.LEFT, padx=5, pady=5)

filter_var = tk.BooleanVar()
filter_var.set(True)
filter_checkbox = tk.Checkbutton(frame_buttons, text="Фильтровать выбросы", variable=filter_var)
filter_checkbox.pack(side=tk.LEFT, padx=5, pady=5)

label_threshold = tk.Label(frame_buttons, text="Пороговое значение:")
label_threshold.pack(side=tk.LEFT, padx=5, pady=5)

entry_threshold = tk.Entry(frame_buttons, width=10)
entry_threshold.pack(side=tk.LEFT, padx=5, pady=5)
entry_threshold.insert(tk.END, "100")

approx_var = tk.BooleanVar()
approx_var.set(False)
approx_checkbox = tk.Checkbutton(frame_buttons, text="Аппроксимировать данные", variable=approx_var)
approx_checkbox.pack(side=tk.LEFT, padx=5, pady=5)

button_save = tk.Button(frame_buttons, text="Сохранить новый файл", command=save_csv)
button_save.pack(side=tk.LEFT, padx=5, pady=5)

label_histogram = tk.Label(frame_buttons, text="Выберите столбец для гистограммы:")
label_histogram.pack(side=tk.LEFT, padx=5, pady=5)

combo_histogram = ttk.Combobox(frame_buttons, width=20)
combo_histogram.pack(side=tk.LEFT, padx=5, pady=5)

button_plot_histogram = tk.Button(frame_buttons, text="Отрисовать гистограмму", command=plot_histogram)
button_plot_histogram.pack(side=tk.LEFT, padx=5, pady=5)

root.mainloop()
