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

    def __sort_by(self,file:str) -> int:
        '''
        функция для сортировки при считывания файлов
        '''
        # return int(re.search(r'\d{3,}', file)[0])
        return int(file.split('/')[-1].split('.')[0])

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
            files = glob.glob(os.path.join(self.path, fr"*.{extension}"))
            print(f"[*] Найдено {len(files)} Файлов с раширением {extension}.")
            if not os.path.isdir(os.path.join(self.path, folder_image)):
                # print(folder_image)
                os.mkdir(os.path.join(self.path, folder_image))
                print(f"[+] Создана папка {folder_image}.")
            folder_location = os.path.join(self.path, folder_image)
            paths.append(folder_location)

            for file in files:
                if re.fullmatch(f'.*\d*\.\d*\.{extension}|.*exp_params.json', file) is not None:
                    nowlocation = os.path.basename(file)
                    dst = os.path.join(self.path, folder_image, nowlocation)
                    print(f"[*] Перенесен файл '{file}' в {dst}")
                    # print(dst)
                    shutil.move(file, dst)
        return paths

    def read_tenzo_value_log_to_dictionary(self):
        self.tenzo = {}
        self.tenzo_time = []
        self.tenzo_data = []
        with open(f'{self.path}') as datalog:
            iter = 1
            for index, value in enumerate(datalog):
                self.tenzo_data.append({0: [], 1: [], 2: [], 3: []})
                data = value.split(',')
                # print(print(f'{motion=}'))
                # gg = []
                # for i in motion:
                #     res = re.findall(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)', i)
                #     gg.append(res[0])
                # self.move.append(gg)
                # self.tenzo_time.append(data[0][1:])
                # print(f'{data=}')
                print('=================================================')
                for i,val in enumerate (data[-4:]):

                    res = re.findall(r'-?\d+', val)
                    if int(data[1]) not in self.tenzo:
                        self.tenzo[int(data[1])] = []
                    if int(data[1]) == iter:
                        self.tenzo[iter].append(*res)
                    else:
                        iter += 1
                        if int(data[1]) == iter:
                            self.tenzo[iter].append(*res)
                    # print(f'{res=}')
            datalog.close()
        # print(self.tenzo_time)
    def parsing_tenzo_value(self):
        self.sampling =[]
        self.tenzo_data = []
        for index, data in enumerate(self.tenzo):
            self.tenzo_data.append({0:[],1:[],2:[],3:[]})
            sampling_i = int(len(self.tenzo[data]) / 4)
            self.sampling.append(sampling_i)
            i_0 = 0
            i_1 = 1
            i_2 = 2
            i_3 = 3
            for i in range(sampling_i):
                self.tenzo_data[index][0].append(abs(float(self.tenzo[data][i_0 + i * 4])))
                self.tenzo_data[index][1].append(abs(float(self.tenzo[data][i_1 + i * 4])))
                self.tenzo_data[index][2].append(abs(float(self.tenzo[data][i_2 + i * 4])))
                self.tenzo_data[index][3].append(abs(float(self.tenzo[data][i_3 + i * 4])))

    @staticmethod
    def find_outliers_iqr(data):
        sorted_data = sorted(data)
        q1, q3 = np.percentile(sorted_data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        for i, x in enumerate(data):
            # if x == 2106498.0:
            #     print(f'====================================================={x=}')
            #     print(f'{data=}\n,{lower_bound=},{upper_bound=}')
            if x < lower_bound or x > upper_bound:
                # print(x)
                data.pop(i)
        return data

    @staticmethod
    def find_outliers_z_score(data):
        mean = np.mean(data)
        std = np.std(data)
        # print(f'{std=}')
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        for i,x in enumerate(data):
            # if x == 2106498.0:
            #     print(f'====================================================={x=}')
            #
            #     z_score = (x - mean) / std
            #     print(f'{data=}\n{z_score=},{lower_bound=},{upper_bound=},{mean=},{std=}')
            z_score = (x - mean) / std
            if z_score < lower_bound or z_score > upper_bound:
                # print(x)
                data.pop(i)
        return data

    def search_outlier(self):
        for index, window in enumerate(self.tenzo_data):
            for key in window.keys():
                values = window[key]
                if len(values) > 3:
                    # print('до',len(values))
                    # values = DataProcessing.find_outliers_z_score(list(set(values)))
                    values = DataProcessing.find_outliers_iqr(list(set(values)))
                    # print(f'{values=},\t std = {np.std(values)},\t len = {len(values)}\n')
                window[key] = values

    def mean_value_tenzo(self):
        self.json_files=glob.glob('stend/VHB491012mmBiostand1-3/Indication/*.json')
        # coef = [46745, 45731, 45731, 39700]
        self.json_files.sort(key=self.__sort_by)
        # print(self.json_files)
        self.json_files_average_strength = {}
        # print(self.tenzo_data)
        self.error_open = []
        for index, window in enumerate(self.tenzo_data):
            with open(f'{self.json_files[index]}' ) as json_file:
                file = eval(json.load(json_file))
                # print(file)
                try:
                    pass
                    # print(file.keys())
                    file[f'{index+1}']
                    # # print(self.file[f'{index + 1}'][1])
                except KeyError:
                    # print('==================================================')
                    self.error_open.append(index+1)
                    # print('==================================================')
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
        for index, file in enumerate(self.json_files):
            with open(f'{file}','w') as json_file:
                if index+1 in self.error_open:
                    json.dump(f'{index+1:"Reading error"}',json_file )
                else:
                    values=self.json_files_average_strength[f'{index + 1}']
                    json.dump(f'{index + 1:{values}}',json_file)

    def rewriting_file_log(self):
        pass


    # def drow_plot(self):
    #     sss = []
    #     for i in self.file:
    #         mean0_2 = self.file[i][1][0] + self.file[i][1][2]
    #         sss.append(mean0_2)
    #     plt.plot(sss)
    #     plt.show()
def test():
    tt = DataProcessing('stend/VHB491012mmBiostand1-3/exp_params.log','ttt')
    tt.read_tenzo_value_log_to_dictionary()
    tt.parsing_tenzo_value()
    # tt.search_outlier()
    # tt.mean_value_tenzo()
    # tt.rewriting_file_json()
    # tt.drow_plot()

if __name__ == "__main__":
    test()
    # data = np.array([10, 20, 300, 30, 40, 500, 50, 60, 700])
    # process_variance = 1.0  # Дисперсия процесса (варьируйте по необходимости)
    # measurement_variance = 100.0  # Дисперсия измерений (варьируйте по необходимости)
    #
    # filtered_data = kalman_filter(data, process_variance, measurement_variance)
    # print(filtered_data)
