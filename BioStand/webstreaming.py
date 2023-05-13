from os import environ
from os import environ
from singlemotiondetector import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask_cors import CORS
from flask import render_template, flash, redirect,url_for, request, send_from_directory, jsonify
import threading
import argparse
import datetime
import imutils
import time
import cv2
import gxipy as gx
from flask_bootstrap import Bootstrap
import json
import requests
from src.data_classes import Sets
import numpy as np
import os
from src import ruler

class Cur_frame():
    frame = ''
    path = ''
    exp_name = ''
    exp_path = ''  
    frame_jpg = ''  
class Ser_Mon():
    ser =''
    mon =''
    tele = []
    outdata = []
    i = 0
    epoh = ''
    tron = True
    def save_data (self, data):
        ttime = time.time()
        np.save(os.path.join(Cur_frame.exp_path,str(ttime)),Cur_frame.frame)
        input_json = data
        with open (os.path.join(Cur_frame.exp_path,str(ttime))+'.json', 'w') as f:
            f.write(json.dumps(input_json))
        
        
# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
bootstrap = Bootstrap(app)
# enable CORS
CORS(app)

device_manager = gx.DeviceManager()
dev_num, dev_info_list = device_manager.update_device_list()
print(dev_info_list)
str_mac = dev_info_list[0].get("mac")
camera = device_manager.open_device_by_mac(str_mac)
# camera.stream_on()
time.sleep(7.0)
#np.load()
@app.route("/")
def index():
    Cur_frame.path =os.path.join(os.getcwd(), 'files')
    return render_template("index.html")

# def detect_motion(frameCount):
# 	# grab global references to the video stream, output frame, and
# 	# lock variables
# 	global camera, outputFrame, lock
# 	# initialize the motion detector and the total number of frames
# 	# read thus far
# 	md = SingleMotionDetector(accumWeight=0.01)
# 	total = 0
# 	# loop over frames from the video stream
# 	while True:
# 		# read the next frame from the video stream, resize it,
# 		# convert the frame to grayscale, and blur it
# 		#frame = vs.read()
# 		frame = camera.data_stream[0].get_image()
# 		frame = frame.convert("RGB")
# 		frame = frame.get_numpy_array()
# 		Cur_frame.frame = frame
# 		frame = imutils.resize(frame, width=600)
# 		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# 		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 		#gray = cv2.GaussianBlur(gray, (7, 7), 0)
# 		# grab the current timestamp and draw it on the frame
# 		timestamp = datetime.datetime.now()
# 		cv2.putText(frame, timestamp.strftime(
# 			"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
# 			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
# 		# if the total number of frames has reached a sufficient
# 		# number to construct a reasonable background model, then
# 		# continue to process the frame
# 		if total > frameCount:
# 			# detect motion in the image
# 			#motion = md.detect(gray)
# 			motion = None
			
# 			# check to see if motion was found in the frame
# 			if motion is not None:
# 				# unpack the tuple and draw the box surrounding the
# 				# "motion area" on the output frame
# 				(thresh, (minX, minY, maxX, maxY)) = motion
# 				cv2.rectangle(frame, (minX, minY), (maxX, maxY),
# 					(0, 0, 255), 2)
		
# 		# update the background model and increment the total number
# 		# of frames read thus far
# 		#md.update(gray)
# 		total += 1
# 		# acquire the lock, set the output frame, and release the
# 		# lock
# 		with lock:
# 			outputFrame = frame.copy()


def detect_motion(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	global camera, outputFrame, lock
	# initialize the motion detector and the total number of frames
	# read thus far
	md = SingleMotionDetector(accumWeight=0.01)
	total = 0
	# loop over frames from the video stream
	while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		#frame = vs.read()
		time.sleep((0.05))
		camera.stream_on()
		frame = camera.data_stream[0].get_image()
		camera.stream_off()
		#if frame != None:
		try:
			frame = frame.convert("RGB")
			frame = frame.get_numpy_array()
			Cur_frame.frame = frame
			frame = imutils.resize(frame, width=600)
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#gray = cv2.GaussianBlur(gray, (7, 7), 0)
		# grab the current timestamp and draw it on the frame
			timestamp = datetime.datetime.now()
			cv2.putText(frame, timestamp.strftime(
				"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
		# if the total number of frames has reached a sufficient
		# number to construct a reasonable background model, then
		# continue to process the frame
			if total > frameCount:
			# detect motion in the image
			#motion = md.detect(gray)
				motion = None
			
			# check to see if motion was found in the frame
				if motion is not None:
				# unpack the tuple and draw the box surrounding the
				# "motion area" on the output frame
					(thresh, (minX, minY, maxX, maxY)) = motion
					cv2.rectangle(frame, (minX, minY), (maxX, maxY),
						(0, 0, 255), 2)
		
		# update the background model and increment the total number
		# of frames read thus far
		#md.update(gray)
			total += 1
		# acquire the lock, set the output frame, and release the
		# lock
			with lock:
				outputFrame = frame.copy()   
		except:
			pass

def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock
	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue
			# encode the frame in JPEG format
			#outputFrame = cv2.cvtColor(outputFrame, cv2.COLOR_BGR2RGB)
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			# ensure the frame was successfully encoded
			if not flag:
				continue
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')
@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")
 
 
# if __name__ == '__main__':
#     HOST = environ.get('SERVER_HOST', 'localhost')
#     try:
#         PORT = int(environ.get('SERVER_PORT', '5555'))
#     except ValueError:
#         PORT = 5555
#     app.run(HOST, PORT,debug=False) 
@app.route('/post_user_coef/', methods=['POST'])
def post_user_coef():

    result_params = {}
    result = json.loads(request.data)
    result_params = result['data']['res']
    Cur_frame.exp_name = result['data']['res'][0]['name']
    Cur_frame.exp_path = os.path.join(Cur_frame.path,Cur_frame.exp_name)
    print(Cur_frame.exp_path)
    if not os.path.exists(Cur_frame.exp_path):
        os.mkdir(os.path.join(Cur_frame.exp_path))
    classes=[]
    for item in result_params:
        classes.append(Sets())
        classes[-1].cycle_count = int(item['cycle_count'])
        classes[-1].magnitude = int(item['magnitude'])
        classes[-1].magnitude_percent_X = int(item['magnitude_percent_X'])
        classes[-1].magnitude_percent_Y = int(item['magnitude_percent_Y'])
        classes[-1].stretch_phase_time = int(item['stretch_phase_time'])
        classes[-1].recover_phase_time = int(item['recover_phase_time'])
        if len(classes)>1:
            classes[-1].result_time = classes[-2].result_time
        classes[-1].prepare_data()
    data = json.dumps(classes[-1].result)
    #res = requests.post('http://localhost:5000/tests/endpoint', json=classes[-1].result)
    #print ('response from server:',res.text)
    #answer = res.json()
    
    
    
    
    with open (os.path.join(Cur_frame.exp_path,'exp_params.json'), 'w') as f:
        f.write(data)




    ruler.init_hardware(ser_mon.ser, ser_mon.mon)
    ruler.start_hardware(ser_mon.ser, ser_mon.mon, classes[-1].result, ser_mon)
    ruler.stop_hardware(ser_mon.ser, ser_mon.mon)

    return jsonify(result_params) 

@app.route('/tests/endpoint', methods=['POST'])
def my_test_endpoint():
    ttime = time.time()
    np.save(os.path.join(Cur_frame.exp_path,str(ttime)),Cur_frame.frame)
    input_json = request.get_json(force=True)
    with open (os.path.join(Cur_frame.exp_path,str(ttime))+'.json', 'w') as f:
        f.write(json.dumps(input_json))



# def my_test_endpoint(data):
#     ttime = time.time()
#     np.save(os.path.join(Cur_frame.exp_path,str(ttime)),Cur_frame.frame)
#     input_json = data
#     with open (os.path.join(Cur_frame.exp_path,str(ttime))+'.json', 'w') as f:
#         f.write(json.dumps(input_json))
        
    # force=True, above, is necessary if another developer 
    # forgot to set the MIME type to 'application/json'
    print ('data from client:', input_json)
    dictToReturn = {'answer':42}
    return jsonify(dictToReturn) 
@app.route('/test/save_frame',  methods=['POST'])
def save_frame():
    # with open ('test_raw', 'w') as f:
    #     f.write(cur_frame.frame)

    np.save(os.path.join(Cur_frame.exp_path,str(time.time())),Cur_frame.frame)
    
    return jsonify ({'res':'ok'})

# check to see if this is the main thread of execution
if __name__ == '__main__':
	ser_mon = Ser_Mon()
	outdata = []
	i = 0
	epoh = 0

	ser = 0
	mon = 0
	tele = []


    
	tron = True #Выводим в консоль сообщения исполнителей
    #tron = False #Не выводим в консоль сообщения исполнителей


	print("Стартую!")


    #Загружаем протокол
    # with open('exp_params.json', 'r') as f:
    #     protocol = json.loads(f.read())
    #     print("Протокол загружен.")


	ser_mon.ser, ser_mon.mon = ruler.init_ports()

    #Запускаем параллельные подпрограммы прослушивания СОМ-портов
	#m = threading.Thread(target=ruler.thread_function_mon, args=(1,ser_mon,), daemon=True)
	#m.start()
	#x = threading.Thread(target=ruler.thread_function, args=(1,ser_mon,), daemon=True)
	#x.start()
	stop_threads = False
    # Запускаем параллельные подпрограммы прослушивания СОМ-портов
	x = threading.Thread(target=ruler.thread_function, args=(1, ser_mon, lambda: stop_threads,), daemon=True)
	x.start()
	m = threading.Thread(target=ruler.thread_function_mon, args=(1, ser_mon, lambda: stop_threads,), daemon=True)
	m.start()

	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())
	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()
	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)
# release the video stream pointer

camera.stream_off()
camera.close_device()
stop_threads = True
