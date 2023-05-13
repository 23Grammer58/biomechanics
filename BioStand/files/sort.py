import os
import glob
import shutil
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt



def sort_in_folder(path):
    
    paths = [] #add paths all to folders
    
    extensions = {
    "npy"   : "Frame",
    "json"  : "Indication",
    "tif"    : "Images"
    }

    for extension,folder_name in extensions.items():
        files = glob.glob(os.path.join(path,f"*.{extension}"))
        print(f"[*] Найдено {len(files)} Файлов с раширением {extension}.")

        if not os.path.isdir(os.path.join(path,folder_name)):
            print(folder_name)
            os.mkdir(os.path.join(path,folder_name))
            print(f"[+] Создана папка {folder_name}.")
        folder_location = os.path.join(path,folder_name)
        #print (type(folder_location))
        print(folder_location)

        for file in files:
            nowlocation = os.path.basename(file)
            dst = os.path.join(path,folder_name,nowlocation)
            print(f"[*] Перенесен файл '{file}' в {dst}")
            #print(dst)
            shutil.move(file,dst)
        paths+=[folder_location]
    return paths

def transfer_npy(path0):
    path_Frame = path0 #path to npy
    list_npy = sorted(os.listdir(path_Frame)) #add elements from folder
    for id, x in enumerate(list_npy):
        os.chdir(paths[2]) #chahg the root directory
        path_npy = os.path.join(path_Frame, x)
        npy = np.load(path_npy)#read npy
        img = Image.fromarray(npy)
        img.save( f"{id}.tif")#transfer from npy to tif 
        os.chdir(os.pardir)

def write_json_to_dictionary(path1):
    path_Indication = path1  
    list_json =  sorted(os.listdir(path_Indication))
    all_json = {}
    id = 1
    #for id, x in enumerate(list_json):
    for x in list_json:
    #print(len(list_json))
        path_element_Inication = os.path.join(path_Indication, x)
        if "exp_params" not in path_element_Inication: 
            with open(path_element_Inication , 'r') as f:
                dictionary_string = eval(json.load(f ))
                #print(dictionary_string)
                value_key = dictionary_string[f'{id}'] 
                all_json[f"{id}"] = value_key
                id +=1
    #return all_json

#def dictionary_extraction(all_json):
    frame_x = []
    axis_0_2 = []
    axis_1_3 = []
    tenzo_0 = []
    tenzo_1 = []
    tenzo_2 = []
    tenzo_3 = []
    for id, x in enumerate(all_json):
    #for x in all_json:
        frame_x.append(x)
        value_key_x= all_json[x]
        value_axes = value_key_x[0]
        value_tenzo = value_key_x[1]
        value_axis_0_2 = value_axes[0]
        value_axis_1_3 = value_axes[1]
        value_axis_0 = value_axis_0_2[0]
        value_axis_2 = value_axis_0_2[1]
        value_axis_1 = value_axis_1_3[0]
        value_axis_3 = value_axis_1_3[1]
        sum_0_2 = round(value_axis_0 + value_axis_2,2)
        sum_1_3 = round(value_axis_1 +  value_axis_3,2)
        tenzo_0.append(value_tenzo[0])
        tenzo_1.append(value_tenzo[1])
        tenzo_2.append(value_tenzo[2])
        tenzo_3.append(value_tenzo[3])

        if id == 0:
            axis_0_2.append(sum_0_2)
            axis_1_3.append(sum_1_3)
        else:
            axis_0_2.append(round(axis_0_2[id-1] + sum_0_2,2))
            axis_1_3.append(round(axis_1_3[id-1] + sum_1_3,2))
    # print('0_2',axis_0_2)
    # print('1_3',axis_1_3)
    # print('x',frame_x)
    # print('tenzo 0',tenzo_0)
    # print('tenzo 1',tenzo_1)
    # print('tenzo 2',tenzo_2)
    # print('tenzo 3',tenzo_3)

        # print("0:",value_axis_0)
        # print("1:",value_axis_1)
        # print("2:",value_axis_2)
        # print("3:",value_axis_3)

    return(frame_x,axis_0_2,axis_1_3,tenzo_0,tenzo_1,tenzo_2,tenzo_3)    



def axis_labels(x,y):
    if x == data[0]:
        plt.xlabel('frame', fontsize=14)
    if y == data[1]:
        plt.ylabel('stretching by sensor 0 2', fontsize=14)
    if y == data[2]:
        plt.ylabel('stretching by sensor 1 3', fontsize=14)
    if y == data[3]:
        plt.ylabel('force reading of sensor 0', fontsize=14)
    if y == data[4]:
        plt.ylabel('force reading of sensor 1', fontsize=14)
    if y == data[5]:
        plt.ylabel('force reading of sensor 2', fontsize=14)
    if y == data[6]:
        plt.ylabel('force reading of sensor 3', fontsize=14)

def draw_plot(x,y):
    
    fig, ax = plt.subplots(figsize=(10,10), dpi = 150)
    plt.plot(x,y)
    # plt.scatter(x,y)
    plt.xticks(x[::5],rotation = 45)
    #plt.stem(frame_x,tenzo_3)
    #for row in tenzo_3:
    # for id, value in enumerate(y):
    #     #print(row.cty)
    #     ax.text(id, value,  s=round(value, 2), horizontalalignment= 'center', verticalalignment='bottom', fontsize=8)
    axis_labels(x,y)
    plt.show()


if __name__ == "__main__":

    path = "C:/Users/Lenovo/PycharmProjects/MainBioStand/BioStand/files/GoreTex7x7/Frame"
    paths = sort_in_folder(path)
    #transfer_npy(paths[0])
    data = write_json_to_dictionary(paths[1])
    draw_plot(data[0], data[1])
    draw_plot(data[0], data[2])
