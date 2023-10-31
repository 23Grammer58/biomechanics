import serial
from serial.tools import list_ports

import json
import time
import threading

import requests

#outdata = []
#prev_comm = ["1"]
#index_comm = 0

#pointx = 0
#pointy = 0

def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def init_ports():
    available_ports = list_ports.comports()
    print(f'Доступные порты: {[x.device for x in available_ports]}')
    print()
    '''
    Подключение двух устройств по СОМ-портам.

    '''
    #Получаем список портов
    ports = list(list_ports.comports())

    #Получаем список портов и ищем нужное устройство
    for p in ports:
        if str(p).split(' - ')[1][:16] == 'USB-SERIAL CH340':
            print(str(p).split(' - ')[0], ' имеется!')
            name2 = str(p).split(' - ')[0]
            #Подключаемся к нужному устройству
            mon = serial.Serial(
                port=name2, baudrate=230400, bytesize=8, timeout=3, stopbits=serial.STOPBITS_ONE
                )  # open serial port
            time.sleep(1)
            
    for p in ports:
        #print(str(p))
        if str(p).split(' - ')[1][:17] == 'Arduino Mega 2560':
            print(str(p).split(' - ')[0], ' имеется!')
            name1 = str(p).split(' - ')[0]
            #Подключаемся к нужному устройству
            ser = serial.Serial(
                port=name1, baudrate=230400, bytesize=8, timeout=3, stopbits=serial.STOPBITS_ONE
                )  # open serial port
            time.sleep(1)
    print()
    return(ser, mon)

def req (data):
    try:
        url = "/tests/endpoint"
        r = requests.post(url, json = json.loads(data), timeout = 1)
    except:
        print('Нет связи....')
    
    time.sleep(.5)
        

def init_hardware(ser, mon):
    print('\nЖмите ENTER для старта калибровки.', end="")
    #После первого включения измеритель ждет любого входного символа
    mon.write(b'.\n')
    input()

    #Активируем приводы
    ser.write(b'enable\n')
    time.sleep(1)

    #Устанавливаем значение ускорения
    #ser.write(b'accelerate(60000)\n')
    #time.sleep(1)

    #   Используем первую команду протокола для определения скорости
    #   mspeed = int(abs(protocol['1'][0][0]) * 1680 ) # базовая скорость подобрана опытным путем
    #com = 'speed(' + str(mspeed) + ')'
    #com = 'speed(1680)'
    com = 'speed(1000,1000)\n'
    ser.write(com.encode())
    time.sleep(1)

    # com = 'pos(-5500, -5500, -5500, -5500)\n' #Растягиваем калибровочные кольца
    # ser.write(com.encode())
    # time.sleep(10)
    # print('\nГотовимся к калибровке. Жмите ENTER для продолжения.', end="")
    # #После первого включения измеритель ждет любого входного символа
    # input()
    #
    # mon.write(b'\n') #Отправляем измерителю символ переноса строки. Измеритель калибруется.
    # time.sleep(1)
    # print('Ok\n')
    # print('Калибровка проведена. Жмите ENTER для продолжения.', end="")
    # input()
    #
    # mon.write(b'\n') #Отправляем измерителю символ переноса строки. Измеритель отвечает текущим значением по осям.
    # print('Ok\n')
    # time.sleep(1)
    # com = 'pos(5500, 5500, 5500, 5500)\n' #Возвращаемся на исходную позициюC
    # ser.write(com.encode())
    #
    # time.sleep(10)

    print('После возвращения в исходное состояние жмите ENTER для старта протокола.', end="")
    input()

    tele = []
    print('Стартовали...\n')
    time.sleep(1)


def start_hardware(ser, mon, protocol, ser_mon):
    global pointx , pointy, tele, outdata, i, epoh
    #print(protocol)
    print('Protocol started')
    i = 0
    for i in range(len(protocol)):
        start_time = time.time()
        epoh = protocol[i+1]
        ser_mon.epoh = epoh
        ser_mon.i = i
        m0 = sign(epoh[0][0])
        m2 = sign(epoh[0][1])
        m1 = sign(epoh[1][0])
        m3 = sign(epoh[1][1])
        
        #mspeed = int(abs(protocol['1'][0][0]) * 1680 ) # базовая скорость подобрана опытным путем
        mspeed0 = int(abs(epoh[0][0]) * 1000)  # помножить на 4
        mspeed2 = int(abs(epoh[0][1]) * 1000)
        mspeed1 = int(abs(epoh[1][0]) * 1000)
        mspeed3 = int(abs(epoh[1][1]) * 1000)

        #mspeed0 = int(epoh[0][0] * -1000)  # помножить на 4
        #mspeed2 = int(epoh[0][1] * -1000)
        #mspeed1 = int(epoh[1][0] * -1000)
        #mspeed3 = int(epoh[1][1] * -1000)

        # Magnitude 10, Stretch Magnitude X 100, Stretch Magnitude Y 100, Stretch Duration 10, Recovery Duration 10, Repetitions 5
        # 10  100  100  10  10   5
        #com = f'speed({mspeed})\n'
        #com = f'speed({mspeed0},{mspeed1},{mspeed2},{mspeed3})\n'
        com = f'speed({mspeed0},{mspeed1})\n'
        ##print('**', com)
        ser.write(com.encode())
        time.sleep(0.3) #Пауза основного цикла. Устанавливается скорость движения.
        
        # Стартуем двигатели с провышением дистанции. Следующий проход исправит.
        # NB: Не прерывать программу, иначе двигатели будут продолжать движение!
        #com = f'pos({-m0 * 10000}, {-m1 * 10000}, {-m2 * 10000}, {-m3 * 10000})\n'
        com = f'pos({-m0 * 10000}, {-m1 * 10000})\n'
        #com = f'pos({-10000}, {-10000}, {-10000}, {-10000})\n'

        # Стартуем тензодатчики. Ответ поступит через 0.1 - 0.3 секунды
        #mon.write(b'\n') #На данной позиции тестовый заход выдает стабильно >5 ошибок на сотню

        #print('**', com)
        ser.write(com.encode())
        time.sleep(0.3) #Пауза основного цикла. Двигатели двигаются.
        
        #Стартуем тензодатчики. Ответ поступит через 0.1 - 0.3 секунды
        #mon.write(b'\n')
        
        time.sleep(0.390) #Пауза основного цикла. Тензоданные готовятся.
        
        a = ser_mon.outdata[-1] # Используем последние полученные данные от тензодатчиков.
        print("{" + f'"{a[0]}" : [{a[1]}, [{a[2]}]]' + "}")
        
        #
        data = "{" + f'"{a[0]}" : [{a[1]}, [{a[2]}]]' + "}"
        
        #r = requests.post(url, json = json.loads(data), timeout = .2)
        t = threading.Thread(target=ser_mon.save_data, args=(data,))
        t.start()
        
        #print("--- %s seconds ---" % (time.time() - start_time))
        

def stop_hardware(ser, mon):
    print('{"-1" : [[[0,0], [0,0]], [0,0]]}') #Конечная посылка
    print('Завершили работу модуля управления.')

    #Останавливаем двигатели.
    ser.write(b'pos(1,1)\n')
    ser.write(b'break\n')
    time.sleep(1)
    ser.write(b'disable\n')
    time.sleep(1)

    ser.write(b'boot\n')
    ser.close()
    mon.close()
    global stop_threads
    stop_threads = True
    print("Завершили прослушивание СОМ-портов.")


def stop():
    global stop_threads
    return stop_threads

def thread_function_mon(name,ser_mon, stop):
    global pointx , pointy, tele, tron, i, epoh
    print("MON-listener стартовал")
    data1 = 0
    mon = ser_mon.mon
    log_file = open('two_gt.log', 'w')
    log_file.close()
    #log_file = open('two_gt.log', 'a')

    while(1):
        try:
            if mon.in_waiting == 0: continue
            #Читаем поток данных от источника
            while mon.in_waiting > 0:
                #Читаем префикс-байт и байт данных
                data1 = mon.read().decode()
                ser_mon.tele.append(data1)
                if data1=='\n':
                    #print([ser_mon.i + 1, ser_mon.epoh, "".join(ser_mon.tele[:-2])])
                    ser_mon.outdata.append([ser_mon.i+1, ser_mon.epoh, "".join(ser_mon.tele[:-2])])
                    if ser_mon.epoh != '':
                        data_bl = [time.time_ns(),ser_mon.i + 1, ser_mon.epoh, "".join(ser_mon.tele[:-2])]
                        log_file = open('two_gt.log', 'a')
                        #print("".join(str(data_bl)), end='')
                        log_file.write(str("".join(str(data_bl))))
                        log_file.write("\n")
                        #print("*")
                        log_file.close()
                    ser_mon.tele = []
                #if ser_mon.tron and ser_mon.epoh != '':
                    #print("\033[34m{}".format(data1), end='')
                    #print(data1, end='')
        except:
            pass

        #log_file.flush()

        if stop():
            #log_file.close()
            print("  Exiting loop.")
            break

def thread_function(name, ser_mon, stop):
    global pointx , pointy, tron, outdata

    print("COM-listener стартовал")
    data1 = 0 
    ser = ser_mon.ser
    mon = ser_mon.mon
    while(1):
        #try:
        #    if mon.in_waiting == 0:
        #        mon.write(b'\n')
        #except:
        #    pass
        try:
            if ser.in_waiting == 0: continue
            #Читаем поток данных от источника
            while ser.in_waiting > 0:
                #Читаем префикс-байт и байт данных
                data1 = ser.read()
                if ser_mon.tron:
                    print("\033[34m{}".format(data1.decode()), end='')
                    print("\033[30m", end='')
                    
        except:
            pass

        if stop():
            print("  Exiting loop.")
            break
            
def main_loop(protocol):
    ser = 0
    mon = 0
    tele = []
    outdata = []
    i = 0
    epoh = 0
    
    nomber = 0
    
    tron = True #Выводим в консоль сообщения исполнителей
    #tron = False #Не выводим в консоль сообщения исполнителей

    print("Стартую!")


    #Загружаем протокол
    # with open('exp_params.json', 'r') as f:
    #     protocol = json.loads(f.read())
    #     print("Протокол загружен.")

   
    ser, mon = init_ports()
    stop_threads = False
    #Запускаем параллельные подпрограммы прослушивания СОМ-портов
    x = threading.Thread(target=thread_function, args=(1,lambda: stop_threads,), daemon=True)
    x.start()
    m = threading.Thread(target=thread_function_mon, args=(1,lambda: stop_threads,), daemon=True)
    m.start()


    init_hardware(ser, mon)
    start_hardware(ser, mon, protocol)

    stop_threads = True
    

if __name__ == "__main__":
    #global outdata
    
    ser = 0
    mon = 0
    tele = []
    outdata = []
    i = 0
    epoh = 0
    
    nomber = 0
    
    tron = True #Выводим в консоль сообщения исполнителей
    #tron = False #Не выводим в консоль сообщения исполнителей

    print("Стартую!")


    #Загружаем протокол
    with open('exp_params.json', 'r') as f:
        protocol = json.loads(f.read())
        print("Протокол загружен.")


    ser, mon = init_ports()

    #Запускаем параллельные подпрограммы прослушивания СОМ-портов
    x = threading.Thread(target=thread_function, args=(1,), daemon=True)
    x.start()
    m = threading.Thread(target=thread_function_mon, args=(1,), daemon=True)
    m.start()


    init_hardware(ser, mon)
    start_hardware(ser, mon, protocol)
    stop_hardware(ser, mon)

