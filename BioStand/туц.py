import tkinter as tk
import os
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.interpolate import UnivariateSpline
import numpy as np


class CSVPlotterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Plotter")

        self.file_path = None
        self.x_axis = None
        self.y_axis = None
        self.filter_type = "Медианный фильтр"
        self.filter_window_size = 3
        self.smoothing_factor = 0.01

        self.label_file = tk.Label(root, text="Выберите CSV файл:")
        self.label_file.pack(pady=5)

        self.button_browse = tk.Button(root, text="Обзор", command=self.browse_file)
        self.button_browse.pack(pady=5)

        self.label_x_axis = tk.Label(root, text="Выберите ось X:")
        self.label_x_axis.pack(pady=5)

        self.x_axis_var = tk.StringVar()
        self.dropdown_x_axis = tk.OptionMenu(root, self.x_axis_var, "")
        self.dropdown_x_axis.pack(pady=5)

        self.label_y_axis = tk.Label(root, text="Выберите ось Y:")
        self.label_y_axis.pack(pady=5)

        self.y_axis_var = tk.StringVar()
        self.dropdown_y_axis = tk.OptionMenu(root, self.y_axis_var, "")
        self.dropdown_y_axis.pack(pady=5)

        self.label_filter_type = tk.Label(root, text="Выберите тип фильтра:")
        self.label_filter_type.pack(pady=5)

        self.filter_type_var = tk.StringVar()
        self.filter_type_var.set(self.filter_type)
        self.dropdown_filter_type = tk.OptionMenu(root, self.filter_type_var, "Медианный фильтр", "Простой Калман", "Savitzky-Golay", "Butterworth", command=self.update_filter_type)
        self.dropdown_filter_type.pack(pady=5)

        self.label_filter_window = tk.Label(root, text="Размер окна:")
        self.label_filter_window.pack(pady=5)

        self.filter_window_entry = tk.Entry(root)
        self.filter_window_entry.pack(pady=5)

        self.label_smoothing_factor = tk.Label(root, text="Коэффициент сглаживания (сплайн):")
        self.label_smoothing_factor.pack(pady=5)

        self.smoothing_factor_entry = tk.Entry(root)
        self.smoothing_factor_entry.pack(pady=5)

        self.button_plot = tk.Button(root, text="Построить график", command=self.plot_graph)
        self.button_plot.pack(pady=10)

        self.button_save_csv = tk.Button(root, text="Сохранить CSV", command=self.save_csv)
        self.button_save_csv.pack(pady=10)

    def get_script_directory(self):
        return os.path.dirname(os.path.realpath(__file__))

    def save_csv(self):
        if not self.x_axis or not self.y_axis:
            messagebox.showerror("Ошибка", "Выберите обе оси X и Y.")
            return

        try:
            self.filter_window_size = int(self.filter_window_entry.get())
            self.smoothing_factor = float(self.smoothing_factor_entry.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Неверные значения размера окна или коэффициента сглаживания.")
            return

        data = pd.read_csv(self.file_path)
        x_data = data[self.x_axis]
        y_data = data[self.y_axis]

        if self.filter_type == "Медианный фильтр":
            y_data_filtered = signal.medfilt(y_data, kernel_size=self.filter_window_size)
        elif self.filter_type == "Простой Калман":
            y_data_filtered = self.kalman_filter(y_data)
        elif self.filter_type == "Savitzky-Golay":
            poly_degree = self.filter_window_size
            y_data_filtered = signal.savgol_filter(y_data, window_length=poly_degree + 1, polyorder=poly_degree)
        elif self.filter_type == "Butterworth":
            nyquist = 0.5 * len(y_data)
            lowcut = float(self.filter_window_size)
            b, a = signal.butter(4, lowcut / nyquist, btype='low')
            y_data_filtered = signal.filtfilt(b, a, y_data)

        # Create a new DataFrame with filtered data
        filtered_data_df = pd.DataFrame({self.x_axis: x_data, self.y_axis: y_data_filtered})

        # Get the output folder path and create the 'filtered_data' folder if it doesn't exist
        output_folder = os.path.join(self.get_script_directory(), "filtered_data")
        os.makedirs(output_folder, exist_ok=True)

        # Get the output file path
        output_file_path = os.path.join(output_folder, "StendxVHB491012mm1-3.csv")

        try:
            # Save the DataFrame to the CSV file
            filtered_data_df.to_csv(output_file_path, index=False)
            print("File saved successfully.")
            messagebox.showinfo("Сохранение", "Файл успешно сохранен.")
        except Exception as e:
            print(f"Error: {e}")
            messagebox.showerror("Ошибка", f"Ошибка при сохранении файла: {e}")

    def browse_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if self.file_path:
            self.load_data()

    def load_data(self):
        data = pd.read_csv(self.file_path)
        self.columns = data.columns.tolist()
        self.update_dropdowns()

    def update_dropdowns(self):
        self.x_axis_var.set("")
        self.y_axis_var.set("")
        self.dropdown_x_axis['menu'].delete(0, 'end')
        self.dropdown_y_axis['menu'].delete(0, 'end')

        for column in self.columns:
            self.dropdown_x_axis['menu'].add_command(label=column, command=lambda col=column: self.select_x_axis(col))
            self.dropdown_y_axis['menu'].add_command(label=column, command=lambda col=column: self.select_y_axis(col))

    def select_x_axis(self, column):
        self.x_axis = column
        self.x_axis_var.set(column)

    def select_y_axis(self, column):
        self.y_axis = column
        self.y_axis_var.set(column)

    def update_filter_type(self, selected_filter):
        self.filter_type = selected_filter
        # Update the filter parameter entry fields based on the selected filter type
        if self.filter_type == "Медианный фильтр":
            self.label_filter_window.config(text="Размер окна медианного фильтра:")
        elif self.filter_type == "Простой Калман":
            self.label_filter_window.config(text="Не используется для Простого Калмана:")
        elif self.filter_type == "Savitzky-Golay":
            self.label_filter_window.config(text="Степень полинома:")
        elif self.filter_type == "Butterworth":
            self.label_filter_window.config(text="Нижняя частота:")

    def plot_graph(self):
        if not self.x_axis or not self.y_axis:
            messagebox.showerror("Ошибка", "Выберите обе оси X и Y.")
            return

        try:
            self.filter_window_size = int(self.filter_window_entry.get())
            self.smoothing_factor = float(self.smoothing_factor_entry.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Неверные значения размера окна или коэффициента сглаживания.")
            return

        data = pd.read_csv(self.file_path)
        x_data = data[self.x_axis]
        y_data = data[self.y_axis]

        try:
            self.filter_window_size = int(self.filter_window_entry.get())
            self.smoothing_factor = float(self.smoothing_factor_entry.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Неверные значения размера окна или коэффициента сглаживания.")
            return

        # Применение медианного фильтра только к столбцу 'y'
        #y_data_filtered = signal.medfilt(y_data, kernel_size=self.filter_window_size)

        # Гладкая аппроксимация данных после фильтрации с помощью сплайна
        #spline = UnivariateSpline(x_data, y_data_filtered, s=self.smoothing_factor)

        plt.figure(figsize=(8, 6))
        plt.plot(x_data, y_data, marker='x', label='GoreTexX')
        #plt.plot(x_data, y_data, marker='o', label='GoreTexY')
        #plt.plot(x_data, y_data_filtered, marker='x', label=f'Displacement Protocol ({self.filter_window_size})')
        #plt.plot(x_data, spline(x_data), label=f'Гладкая аппроксимация (сплайн)')
        plt.xlabel(self.x_axis)
        plt.ylabel(self.y_axis)
        plt.title(f"Протокол нагружения")
        plt.legend()
        plt.grid(False)
        plt.show()

        if self.filter_type == "Медианный фильтр":
            y_data_filtered = signal.medfilt(y_data, kernel_size=self.filter_window_size)
        elif self.filter_type == "Простой Калман":
            y_data_filtered = self.kalman_filter(y_data)
        elif self.filter_type == "Savitzky-Golay":
            poly_degree = self.filter_window_size
            y_data_filtered = signal.savgol_filter(y_data, window_length=poly_degree + 1, polyorder=poly_degree)
        elif self.filter_type == "Butterworth":
            nyquist = 0.5 * len(y_data)
            lowcut = float(self.filter_window_size)
            b, a = signal.butter(4, lowcut / nyquist, btype='low')
            y_data_filtered = signal.filtfilt(b, a, y_data)
    
    # def add_plot():
    #         file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    #         if file_path:
    #             title = title_entry.get()
    #             x_axis, y_axis = get_axes_from_csv(file_path)
    #             if x_axis and y_axis:
    #                 plot_csv(file_path, x_axis, y_axis, title)
    #                 canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = CSVPlotterApp(root)
    root.mainloop()
