from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys
import os
import json
import glob
import re
import shutil
import csv


class NumpyEncoder(json.JSONEncoder):
    '''
    Класс для  записи в json, его лучше не трогать
    '''

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class DataProcessing:
    def __init__(self,path:str,name_protocol:str):
        self.path = path
        self.name_protocol = name_protocol
        self.path_log = self.__path_log()
        self.path_log_new =self.__path_log_new()
        self.paths = self.__sort_in_folder()


    def __path_log(self):
        logs = glob.glob(os.path.join(self.path,'*.log'))
        for val in logs:
            if not re.search(f"(new|{self.name_protocol})", os.path.split(val)[1]):
                print(val)
                return val
        # print(logs)
    def __path_log_new(self):
        log_file = list(os.path.split(self.path_log))
        # print(log_file)
        # log_name = log_file[1].split('.')
        # log_name[0]= log_name[0] +'_new'
        # path_log_new = '.'.join(log_name)
        # new_path_log = os.path.join(log_file[0],path_log_new)
        log_file[1] = re.sub(r'(.+).(log)', r'\1_new.\2', log_file[1])
        new_path_log = os.path.join(log_file[0],log_file[1])
        return new_path_log


    def __sort_by(self,file:str) -> int:
        '''
        функция для сортировки при считывания файлов
        '''
        # return int(re.search(r'\d{3,}', file)[0])
        # os.path.split(self.path)
        # return int(file.split('/')[-1].split('.')[0])
        # os.path.split(file)[1].split('.')[0])
        file = os.path.split(file)[1]
        name_file = re.sub(r'([\d+\.]?\d+).json|([\d+\.]?\d+).npy', r'\1\2', file)
        return float(name_file)

    def __sort_in_folder(self):
        '''
        функция для создания папочек и сортировки файлов по ним
        '''

        paths = []  # add paths all to folders
        extensions = {
            "npy": "Frame",
            "json": "Indication",
            "tif": "Images",
            " ": "Grayscale"
        }

        for extension, folder_image in extensions.items():
            files = glob.glob(os.path.join(self.path, f"*.{extension}"))
            print(f"[*] Найдено {len(files)} Файлов с раширением {extension}.")
            if not os.path.isdir(os.path.join(self.path, folder_image)):
                os.mkdir(os.path.join(self.path, folder_image))
                print(f"[+] Создана папка {folder_image}.")
            folder_location = os.path.join(self.path, folder_image)
            paths.append(folder_location)

            for file in files:
                # if 'exp_params'in file:
                #     continue
                if re.fullmatch(f'.*\d+\.?\d+?.{extension}|(?!.*exp_params.json)', file) is not None:
                    nowlocation = os.path.basename(file)
                    dst = os.path.join(self.path, folder_image, nowlocation)
                    print(f"[*] Перенесен файл '{file}' в {dst}")
                    # print(dst)
                    shutil.move(file, dst)
        return paths

    def read_tenzo_value_log_to_dictionary(self):
        self.sampling = []
        self.tenzo_data = []
        sum_sampling = 0
        file_lines = 0
        with open(f'{self.path_log}') as datalog:
            iter = 1
            self.tenzo_data.append({0: [], 1: [], 2: [], 3: []})
            for index, value in enumerate(datalog):
                data = value.split(',')
                for i,val in enumerate (data[-4:]):
                    res = re.findall(r'-?\d+', val)
                    if int(data[1]) == iter:
                        self.tenzo_data[iter-1][i].append(float(*res))
                    else:
                        #блок для вычисления дискретизации
                        if len(self.sampling) == 0:
                            self.sampling.append(index)
                        else:
                            sum_sampling += self.sampling[-1]
                            self.sampling.append(index - sum_sampling)
                        iter += 1

                        self.tenzo_data.append({0: [], 1: [], 2: [], 3: []})
                        if int(data[1]) == iter:
                            self.tenzo_data[iter-1][i].append(float(*res))
                file_lines = index
            datalog.close()
        sum_sampling += self.sampling[-1]
        self.sampling.append(file_lines - sum_sampling)

    @staticmethod
    def find_outliers_iqr(data):
        sorted_data = sorted(data)
        q1, q3 = np.percentile(sorted_data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        filtered_data = [x for x in data if (lower_bound < x < upper_bound)]
        return filtered_data
        # for i, x in enumerate(data):
        #     #     print(f'{data=}\n,{lower_bound=},{upper_bound=}')
        #     if x < lower_bound or x > upper_bound:
        #         # print(x)
        #         data.pop(i)
        # if abs(max(data)) >150000:
        #     print('posle', data)
        # return data

    @staticmethod
    def find_outliers_z_score(data):
        mean = np.mean(data)
        std = np.std(data)
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        filtered_data = [x for x in data if (lower_bound < x < upper_bound)]
        return filtered_data

    def search_outlier(self):
        for index, window in enumerate(self.tenzo_data):
            for key in window.keys():
                values = window[key]
                if len(values) > 3:
                    # values = DataProcessing.find_outliers_z_score(list(set(values)))
                    values = DataProcessing.find_outliers_iqr(list(set(values)))
                    # print(f'{values=},\t std = {np.std(values)},\t len = {len(values)}\n')
                window[key] = values

    def mean_value_tenzo(self):
        path_Indication = self.paths[1]
        self.json_files=glob.glob(os.path.join(path_Indication,'*.json'))
        # coef = [46745, 45731, 45731, 39700]
        self.json_files.sort(key=self.__sort_by)
        # print(self.json_files)
        self.json_files_average_strength = {}
        # print(self.tenzo_data)
        self.error_open = {}
        for index, window in enumerate(self.tenzo_data):
            with open(f'{self.json_files[index]}' ) as json_file:

                try:
                    file = eval(json.load(json_file))
                    file[f'{index + 1}']
                except Exception as e:
                    print(self.json_files[index])
                    print(file)
                    print(index+1)
                    self.error_open[index+1] = e
                    continue
                tenzo_mean_value = []
                for key in window.keys():
                    values = window[key]
                    # print(f'{window=},{values=},{key=}')
                    mean = sum(values)/len(values)
                    tenzo_mean_value.append(mean)
                # for i in range (len(tenzo_mean_value)):
                #     tenzo_mean_value[i]  = tenzo_mean_value[i]/coef[i]
                # self.file[index+1] = [self.move[index], tenzo_mean_value]
                file[f'{index + 1}'][1] = tenzo_mean_value
            self.json_files_average_strength.update(file)
        # print(self.json_files_average_strength)
                # print(f'{tenzo_mean_value=}')

    def rewriting_file_json(self):
        # print(self.error_open)
        for index, file in enumerate(self.json_files):
            with open(f'{file}','w') as json_file:
                if index+1 in self.error_open:
                    d = {index+1: f'Reading error  {self.error_open[index+1]}'}
                    ff = json.dump(f'{d}',json_file)
                    print(ff)
                    break
                    # print(json.dumps(f"{index + 1}:'Reading error  {self.error_open[index + 1]}'"))

                else:
                    values=self.json_files_average_strength[f'{index + 1}']
                    d = {f'{index + 1}':values}
                    json.dump(f'{d}',json_file)
                    # print(json.dumps(f"{index + 1}:{values}",indent=4))

    def rewriting_file_log(self):
        with open(f'{self.path_log}') as datalog:
            with open(f'{self.path_log_new}','w', encoding='utf-8') as new_datalog:
                iter = 1
                for index, value in enumerate(datalog):
                    data = value.split(',')
                    move_1_3 = np.array([])
                    move_2_4 =  np.array([])
                    for i, val in enumerate(data[2:6]):
                        res = re.findall(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)', val)
                        if i == 0 or i == 1:
                            move_1_3 = np.append(move_1_3,float(*res))
                        elif i == 2 or i == 3:
                            move_2_4 = np.append(move_2_4,float(*res))
                    if int(data[1]) == iter:
                        move_1_3 = move_1_3 / self.sampling[iter-1]
                        move_2_4 = move_2_4 / self.sampling[iter-1]
                    else:
                        iter += 1
                        if int(data[1]) == iter:
                            move_1_3 = move_1_3 / self.sampling[iter - 1]
                            move_2_4 = move_2_4 / self.sampling[iter - 1]
                        else:
                            print('dddddddd')
                    data_str_datalog = [tuple(move_1_3),tuple(move_2_4)]
                    res = re.sub(r'\(\([-+]?\d+\.\d+, [-+]?\d+\.\d+\), \([-+]?\d+\.\d+, [-+]?\d+\.\d+\)\)', f'({data_str_datalog[0]}, {data_str_datalog[1]})',value)
                    new_datalog.write(res)

            datalog.close()
            new_datalog.close()


    def create_csv(self):
        data = {}
        names = ['frame', 'axis_0_2', 'axis_1_3', 'tenzo_0', 'tenzo_1', 'tenzo_2', 'tenzo_3', 'mean_tenzo_0_2',
                 'mean_tenzo_1_3']

        for name in names:
                data[name] = []

        for id, x in enumerate(self.json_files_average_strength):
            data['frame'].append(x)
            value_key_x = self.json_files_average_strength[x]
            value_axes = value_key_x[0]
            value_tenzo = value_key_x[1]
            value_axis_0_2 = value_axes[0]
            value_axis_1_3 = value_axes[1]
            value_axis_0 = value_axis_0_2[0]
            value_axis_2 = value_axis_0_2[1]
            value_axis_1 = value_axis_1_3[0]
            value_axis_3 = value_axis_1_3[1]
            sum_0_2 = round(value_axis_0 + value_axis_2, 2)
            sum_1_3 = round(value_axis_1 + value_axis_3, 2)
            data['tenzo_1'].append(abs(value_tenzo[1]))
            data['tenzo_2'].append(abs(value_tenzo[2]))
            data['tenzo_0'].append(abs(value_tenzo[0]))
            data['tenzo_3'].append(abs(value_tenzo[3]))
            data['mean_tenzo_0_2'].append((abs(value_tenzo[0]) + abs(value_tenzo[2])) / 2)
            data['mean_tenzo_1_3'].append((abs(value_tenzo[1]) + abs(value_tenzo[3])) / 2)

            if id == 0:
                data['axis_0_2'].append(sum_0_2)
                data['axis_1_3'].append(sum_1_3)

            else:
                data['axis_0_2'].append(round(data['axis_0_2'][id - 1] + sum_0_2, 2))
                data['axis_1_3'].append(round(data['axis_1_3'][id - 1] + sum_1_3, 2))

        data['correlation_force_0_2'] = np.array(data['mean_tenzo_0_2']) * 0.96713
        data['correlation_force_1_3'] = np.array(data['mean_tenzo_1_3']) * 0.88

        with open(fr'{os.path.join(self.path, self.name_protocol)}.csv', 'w', newline='') as csvfile:
            fieldnames = list(data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(data['frame'])):
                row = {}
                for key in data:
                    # print(key)
                    row[key] = data[key][i]
                # print(row)
                writer.writerow(row)
        self.data = data
        print(fr'создан csv файл {os.path.join(self.path, self.name_protocol)}.csv')
    def draw_plot(self):
        x = self.data['axis_0_2']
        y = self.data['mean_tenzo_0_2']
        # print('x',len(x))
        # print('y',y)
        fig, ax = plt.subplots(figsize=(150, 30), dpi=150)
        # plt.ylim(0, 0.3)
        plt.plot(x, y)
        # plt.scatter(x, y)
        plt.xticks(x[::10], rotation=45)
        # plt.stem(frame_x,tenzo_3)
        # for row in tenzo_3:
        # for id, value in enumerate(y):
        #     # print(row.cty)
        #     ax.text(id, value, s=round(value, 2), horizontalalignment='center', verticalalignment='bottom', fontsize=8)
        # self.axis_labels(x, y)
        plt.show()
def main():
    # tt = DataProcessing('confa','ttt')
    tt = DataProcessing(r'C:\Users\Lenovo\PycharmProjects\MainBioStand\BioStand\files\SpecleTest','SpecleTest1')
    tt.read_tenzo_value_log_to_dictionary()
    tt.search_outlier()
    tt.mean_value_tenzo()
    tt.rewriting_file_json()
    tt.rewriting_file_log()
    tt.create_csv()
    tt.draw_plot()

if __name__ == "__main__":
    main()
    # data = np.array([10, 20, 300, 30, 40, 500, 50, 60, 700])
    # process_variance = 1.0  # Дисперсия процесса (варьируйте по необходимости)
    # measurement_variance = 100.0  # Дисперсия измерений (варьируйте по необходимости)
    #
    # filtered_data = kalman_filter(data, process_variance, measurement_variance)
    # print(filtered_data)
