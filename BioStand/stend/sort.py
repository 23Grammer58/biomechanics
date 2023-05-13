import glob
import json
import os
import shutil
import sys
import timeit
import cv2
import  re
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from CalibrationOpenCV.Calibration2 import calibration as calib

def sort_by(file: str) -> int:
    # print( int(re.search(r'\d{3,}',file)[0]))
    return int(re.search(r'\d{3,}',file)[0])

def sort_in_folder(path):
    paths = []  # add paths all to folders

    extensions = {
        "npy": "Frame",
        "json": "Indication",
        "tif": "Images",
        "": "Grayscale"
    }

    for extension, folder_image in extensions.items():
        files = glob.glob(os.path.join(path, f"*.{extension}"))
        print(f"[*] Найдено {len(files)} Файлов с раширением {extension}.")

        if not os.path.isdir(os.path.join(path, folder_image)):
            print(folder_image)
            os.mkdir(os.path.join(path, folder_image))
            print(f"[+] Создана папка {folder_image}.")
        folder_location = os.path.join(path, folder_image)
        # print (type(folder_location))
        # print(folder_location)

        for file in files:
            nowlocation = os.path.basename(file)
            dst = os.path.join(path, folder_image, nowlocation)
            print(f"[*] Перенесен файл '{file}' в {dst}")
            # print(dst)
            shutil.move(file, dst)
        paths.append(folder_location)
    return paths


def transfer_npy(path0,path2):
    start = timeit.default_timer()
    # print(path0)
    path_Frame = glob.glob(f'{path0}/*.npy')# path to npy
    path_Frame.sort(key=sort_by)
    # print(path_Frame)
    #list_npy = sorted(os.listdir(path_Frame))  # add elements from folder
    for id, x in enumerate(path_Frame):
        # print(x)
        #path_npy = os.path.join(path_Frame, x)
        npy = np.load(x)  # read npy
        npy = cv2.cvtColor(npy, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(npy)
        img.save(f"{path2}/{id}.tif")  # transfer from npy to tif
    end = timeit.default_timer()
    print('images created', f"Time taken is {end - start}s")

def conversion_to_grayscale(path0, path3, look_image = False ):
    #-----------------------------шляпа------------------------------
    # for file in sorted(os.listdir(path0)):
    #     img = np.load(file)  # <-- путь до изображения
    #     # convert to grayscale
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     # blur
    #     blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=33, sigmaY=33)
    #     # divide
    #     divide = cv2.divide(gray, blur, scale=255)
    #     # otsu threshold
    #     thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #     # apply morphology
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #     morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    #     # write result to disk
    #     # cv2.imwrite(r'3.jpg', divide)    # <-- куда сохраняет
    #     cv2.imwrite(f'{path3}/{id}.tif', divide)  # <-- куда сохраняет       # <-- куда сохраняет
    #     id += 1
    # os.chdir(os.pardir)
    # -----------------------------шляпа------------------------------
    start = timeit.default_timer()

    images = glob.glob(f'{path0}/*.npy')
    # print(images)
    folder_npy = 'grayscale_npy'

    if not os.path.isdir(os.path.join(path, folder_npy)):
        os.mkdir(os.path.join(path, folder_npy))
    folder_location = os.path.join(path, folder_npy)

    for el in images:
        # print( el)
        res = re.search(r'[0-9]+', el[-7:-3])
        # print(res[0])
        im = np.load(el)

        # im = np.load(el)
        # plt.imshow(imt)
        # plt.show()
        # plt.close()
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # plt.imshow(im)
        # plt.show()
        # plt.close()

        # img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        img = np.average(im, axis=2, weights=[0.144, 0.587, 0.299])
        img = img.astype(np.uint8)

        np.save(f'{folder_location}/{res[0]}', img)
        #img = np.average(im, axis=2, weights=[0.299, 0.587, 0.144]) в случае RBG


        # print(img)
        # print(img.shape)
        # plt.imshow(img)
        # plt.show()
        if look_image:
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            img.save(f'{path3}/{res[0]}.tif')
        # plt.imshow(img)
        # plt.show()
        # plt.imsave(f'{path3}/{id}.jpg', img)
        # plt.savefig(f'{path3}/{id}.tif')

        end = timeit.default_timer()
    # print('conversion to grayscale done ', f"Time taken is {end - start}s")

    return  folder_location



def write_json_to_dictionary(path1):
    path_Indication = path1
    list_json = sorted(os.listdir(path_Indication))
    all_json = {}
    id = 1
    for x in list_json:
        # print(len(list_json))
        path_element_Inication = os.path.join(path_Indication, x)
        if "exp_params" not in path_element_Inication:
            with open(path_element_Inication, 'r') as f:
                try:
                    dictionary_string = eval(json.load(f))
                except SyntaxError:
                    print('SyntaxError',id)
                    continue
                try:
                    value_key = dictionary_string[f'{id}']
                except KeyError:
                    print('skip:', id)
                all_json[f"{id}"] = value_key
                id += 1
    # return all_json

    frame_x = []
    axis_0_2 = []
    axis_1_3 = []
    tenzo_0 = []
    tenzo_1 = []
    tenzo_2 = []
    tenzo_3 = []
    mean_tenzo_0_2 = []
    mean_tenzo_1_3 = []
    for id, x in enumerate(all_json):
        # for x in all_json:
        frame_x.append(x)
        value_key_x = all_json[x]
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
        tenzo_0.append(value_tenzo[0])
        tenzo_1.append(value_tenzo[1])
        tenzo_2.append(value_tenzo[2])
        tenzo_3.append(value_tenzo[3])
        mean_tenzo_0_2.append((value_tenzo[0] + value_tenzo[2]) / 2)
        mean_tenzo_1_3.append((value_tenzo[1] + value_tenzo[3]) / 2)

        if id == 0:
            axis_0_2.append(sum_0_2)
            axis_1_3.append(sum_1_3)


        else:
            axis_0_2.append(round(axis_0_2[id - 1] + sum_0_2, 2))
            axis_1_3.append(round(axis_1_3[id - 1] + sum_1_3, 2))
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

    return (frame_x, axis_0_2, axis_1_3, tenzo_0, tenzo_1, tenzo_2, tenzo_3, mean_tenzo_0_2,mean_tenzo_1_3)


def axis_labels(x, y):
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


def draw_plot(x, y):
    fig, ax = plt.subplots(figsize=(150, 30), dpi=150)
    plt.plot(x, y)
    plt.scatter(x, y)
    plt.xticks(x[::5], rotation=45)
    # plt.stem(frame_x,tenzo_3)
    # for row in tenzo_3:
    # for id, value in enumerate(y):
    #     # print(row.cty)
    #     ax.text(id, value, s=round(value, 2), horizontalalignment='center', verticalalignment='bottom', fontsize=8)
    axis_labels(x, y)
    plt.show()


if __name__ == "__main__":
    start = timeit.default_timer()
    try:
        path = 'C:/Users/Lenovo/PycharmProjects/MainBioStand/BioStand/files/VHB4910NewTestForCal22mm'
    except:
        sys.stderr('папочка не найдена')
    paths = sort_in_folder(path)
    # transfer_npy(paths[0],paths[2])
    # path_calib = calib(path,False)
    # conversion_to_grayscale(path[0], paths[3],False)
    data = write_json_to_dictionary(paths[1])
    # draw_plot(data[1],data[7])
    # draw_plot(data[2], data[8])
    end = timeit.default_timer()
    print(f"program running time {int((end - start)//60)}:{int((end - start)%60)}min")
