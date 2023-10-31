import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

class DataVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSV Data Visualizer")
        self.geometry("800x500")

        self.file_path = None
        self.selected_columns = []
        self.filtered_data = None

        self.create_widgets()

    def create_widgets(self):
        self.select_file_button = tk.Button(self, text="Выбрать файл", command=self.select_file)
        self.select_file_button.pack(pady=10)

        self.select_x_label = tk.Label(self, text="Выберите ось X:")
        self.select_x_label.pack()
        self.x_var = tk.StringVar(self)
        self.x_var.set("")
        self.select_x_dropdown = tk.OptionMenu(self, self.x_var, "")

        self.select_y_label = tk.Label(self, text="Выберите ось Y:")
        self.select_y_label.pack()
        self.y_var = tk.StringVar(self)
        self.y_var.set("")
        self.select_y_dropdown = tk.OptionMenu(self, self.y_var, "")

        self.plot_button = tk.Button(self, text="Построить график", command=self.plot_graph)
        self.plot_button.pack(pady=10)

        self.animate_button = tk.Button(self, text="Анимация", command=self.animate_graph)
        self.animate_button.pack(pady=10)

        self.filter_button = tk.Button(self, text="Фильтр", command=self.filter_data)
        self.filter_button.pack(pady=10)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack()

        # Добавляем кнопки для выбора осей
        self.select_axes_button = tk.Button(self, text="Выбрать оси", command=self.select_axes)
        self.select_axes_button.pack(pady=10)

    def select_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Log Files", "*.log")])
        if self.file_path:
            self.load_data()

    def load_data(self):
        if self.file_path.lower().endswith('.log'):
            df = pd.read_csv(self.file_path, delimiter='\t')
        else:
            df = pd.read_csv(self.file_path)

        self.selected_columns = df.columns.tolist()
        self.x_var.set("")
        self.y_var.set("")
        self.select_x_dropdown.destroy()
        self.select_y_dropdown.destroy()
        self.select_x_dropdown = tk.OptionMenu(self, self.x_var, *self.selected_columns)
        self.select_x_dropdown.pack()
        self.select_y_dropdown = tk.OptionMenu(self, self.y_var, *self.selected_columns)
        self.select_y_dropdown.pack()

        # Сохраняем данные для фильтрации
        self.filtered_data = df.copy()

    def plot_graph(self):
        if not self.file_path:
            tk.messagebox.showerror("Ошибка", "Выберите файл!")
            return

        x_column = self.x_var.get()
        y_column = self.y_var.get()

        if not x_column or not y_column:
            tk.messagebox.showerror("Ошибка", "Выберите оси X и Y!")
            return

        df = pd.read_csv(self.file_path)

        x_data = df[x_column]
        y_data = df[y_column]

        self.ax.clear()
        self.ax.plot(x_data, y_data)
        self.ax.set_xlabel(x_column)
        self.ax.set_ylabel(y_column)
        self.ax.set_title("График")
        self.canvas.draw()

    def animate_graph(self):
        if not self.file_path:
            tk.messagebox.showerror("Ошибка", "Выберите файл!")
            return

        x_column = self.x_var.get()
        y_column = self.y_var.get()

        if not x_column or not y_column:
            tk.messagebox.showerror("Ошибка", "Выберите оси X и Y!")
            return

        df = pd.read_csv(self.file_path)

        def init():
            self.ax.clear()
            self.ax.set_xlabel(x_column)
            self.ax.set_ylabel(y_column)
            self.ax.set_title("Анимация графика")
            return self.ax,

        def animate(i):
            x = df[x_column][:i]
            y = df[y_column][:i]
            self.ax.clear()
            self.ax.plot(x, y)
            self.ax.set_xlabel(x_column)
            self.ax.set_ylabel(y_column)
            self.ax.set_title("Анимация графика")
            return self.ax,

        ani = FuncAnimation(self.fig, animate, init_func=init, frames=len(df), blit=True)
        self.canvas.draw()

    def select_axes(self):
        self.load_data()
        self.select_axes_button.config(state=tk.DISABLED)

    def filter_data(self):
        if not self.filtered_data.empty:
            for column in self.selected_columns:
                # Используем медиану и межквартильный размах для определения выбросов
                Q1 = self.filtered_data[column].quantile(0.25)
                Q3 = self.filtered_data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.filtered_data[column] = np.where(
                    (self.filtered_data[column] < lower_bound) | (self.filtered_data[column] > upper_bound),
                    self.filtered_data[column].shift(),
                    self.filtered_data[column]
                )
        else:
            tk.messagebox.showerror("Ошибка", "Выберите файл и оси перед фильтрацией!")

        # Построить график с отфильтрованными данными
        self.plot_graph()

if __name__ == "__main__":
    app = DataVisualizer()
    app.mainloop()
