import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog


def interpolate_and_evaluate(true_values, predicted_values):
    # Размеры выборок
    true_size = len(true_values)
    predicted_size = len(predicted_values)

    # Интерполируем
    if true_size != predicted_size:
        interpolation_function = interp1d(np.arange(predicted_size), predicted_values, kind='linear',
                                          fill_value='extrapolate')

        # Интерполируем до размера выборки true_values
        interpolated_predicted_values = interpolation_function(np.linspace(0, predicted_size - 1, true_size))
    else:
        interpolated_predicted_values = predicted_values

    # calcorr
    correlation_coefficient = np.corrcoef(true_values, interpolated_predicted_values)[0, 1]
    print(correlation_coefficient)
    plt.figure(figsize=(8, 6))
    plt.plot(true_values, label='DMA RSA-G2')
    plt.plot(interpolated_predicted_values, label='Biostand Y axis')
    plt.xlabel('Displacement (mme-01)')
    plt.ylabel('Force(N)')
    plt.legend()
    plt.title('DMA RSA-G2 VHB4010x12mm vs Biostand VHB4010x12mm')
    plt.show()
    # print(predicted_values)
    #----------------------------------------------------------------
    # R^2
    r_squared = r2_score(true_values, interpolated_predicted_values)

    # MAPE
    mape = np.mean(np.abs((true_values - interpolated_predicted_values) / true_values)) * 100
    print(mape)
    print(interpolated_predicted_values)
    print(true_values)
    return correlation_coefficient, r_squared, mape


def load_true_data():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        true_data_entry.delete(0, END)
        true_data_entry.insert(0, file_path)


def load_predicted_data():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        predicted_data_entry.delete(0, END)
        predicted_data_entry.insert(0, file_path)


def calculate_and_display_results():
    true_data_path = true_data_entry.get()
    predicted_data_path = predicted_data_entry.get()

    if not true_data_path or not predicted_data_path:
        result_label.config(text="Загрузите оба файла CSV.")
        return

    # Загрузка данных из CSV файлов
    true_data = pd.read_csv(true_data_path)
    predicted_data = pd.read_csv(predicted_data_path)

    # Извлечение значений из данных
    true_values = true_data['Value'].values
    predicted_values = predicted_data['Value'].values

    # Расчет коэффициентов
    correlation, r_squared, mape = interpolate_and_evaluate(true_values, predicted_values)

    # Отображение результатов на графике
    # plt.figure(figsize=(8, 6))
    # plt.plot(true_values, label='Ground Truth')
    # plt.plot(predicted_values, label='Predicted Data')
    # plt.xlabel('Sample')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.title('График Ground Truth и Predicted Data')
    # plt.show()

    # Вывод результатов
    result_label.config(
        text=f"Коэффициент корреляции: {correlation:.2f}\nКоэффициент детерминации: {r_squared:.2f}\nMAPE: {mape:.2f}%")


# Создание графического интерфейса
root = Tk()
root.title("CSV Interpolation and Evaluation")

# Загрузка данных
load_true_button = Button(root, text="Калибрант", command=load_true_data)
load_true_button.pack(pady=10)
true_data_entry = Entry(root, width=50)
true_data_entry.pack()

load_predicted_button = Button(root, text="Машина", command=load_predicted_data)
load_predicted_button.pack(pady=10)
predicted_data_entry = Entry(root, width=50)
predicted_data_entry.pack()

# Кнопка для расчета и отображения результатов
calculate_button = Button(root, text="Рассчитать и Отобразить Результаты", command=calculate_and_display_results)
calculate_button.pack(pady=20)

# Метка для вывода результатов
result_label = Label(root, text="")
result_label.pack()

root.mainloop()
