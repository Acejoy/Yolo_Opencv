'''the script used to run yolov3 algorithm on images ,image folders, live video with webcam and video file.

1)for live video: python .\yolo_opencv_video.py --config .\yolov3.cfg --weights .\yolov3.weights --classes .\yolov3.txt 

2)for video file:python .\yolo_opencv_video.py --config .\yolov3.cfg --weights .\yolov3.weights --classes .\yolov3.txt 
                then give the path like ./CarsDrivingUnderBridge.mp4

3)for image folder agiain:python .\yolo_opencv_video.py --config .\yolov3.cfg --weights .\yolov3.weights --classes .\yolov3.txt 
                then give the image folder path like : C:\\Users\\Legion\\Documents\\Project_ND_Sir\\Images_dataset\\EDITED OUTDOOR IMAGE DATA\\RGB

4)for an image:python .\yolo_opencv_video.py --config .\yolov3.cfg --weights .\yolov3.weights --classes .\yolov3.txt 
                and add path as: ./me.jpg




'''
import os
import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
#ap.add_argument('-i', '--image_folder', required=True,
#                help = 'path to input image folder')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()


#for scaling the images down
def scale_down(f,factor):
    dims = [f.shape[0],f.shape[1]]
    
    new_f = cv2.resize(f,(dims[1]//factor,dims[0]//factor),interpolation=cv2.INTER_AREA)
    
    return new_f


#get the layers that have no next layers 
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers



#draws the labels ,the boxes
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    #txt= label + "," + str(round(confidence,4))
    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
scale = 0.00392

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

def detect(img,Width,Height):
    #print('h1')
    blob = cv2.dnn.blobFromImage(img, scale, (416,416), (0,0,0), True, crop=False)
    #print('h2')
    net.setInput(blob)
    #print('h3')
    outs = net.forward(get_output_layers(net))
    #print('h4')
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    #print(outs)

    for out in outs:
        for detection in out:
            #print('h5')
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                #print('h7')
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                #print('h8')
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    #removes noises using NMS algorithm
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(img, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    return img

    
#create a menu that will decide whether to use webcam or video feed
option = input('''
                Menu\n
                1)choose live video feed
                2)choose a video
                3)choose image folder
                4)choose image
                Enter the option:''')

if(option=='1'):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    try:
        while True:
            ret,frame = cap.read()
            #print(type(frame))
            Width = frame.shape[1]
            Height = frame.shape[0]
            detected_img = detect(frame,Width,Height)
            #print('hey2')
            cv2.imshow('detected frame', detected_img)
            k=cv2.waitKey(30) & 0xff
            if k == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
                
    except Exception as e:
        print(e)
        cap.release()        
        cv2.destroyAllWindows()
elif(option=='2'):

    #get the video file path
    PATH_TO_FILE = input('Enter the file path for video:')

    cap = cv2.VideoCapture(PATH_TO_FILE)

    try:
        while True:
            ret,frame = cap.read()
            if ret == False:
                break
            #print(type(frame))
            Width = frame.shape[1]
            Height = frame.shape[0]
            detected_img = detect(frame,Width,Height)
            cv2.imshow('detected frame', detected_img)
            k = cv2.waitKey(30) & 0xff
            if  (k == ord('q')) :
                break
    
        cv2.destroyAllWindows()
        cap.release()

    except Exception as e:
        print(e)
        cap.release()        

elif(option=='3'):
    #PATH_TO_IMAGE_FOLDER = input('Enter the image folder path for video:')
    PATH_TO_IMAGE_FOLDER = r'C:\Users\Legion\Documents\Project_ND_Sir\Images_dataset\EDITED OUTDOOR IMAGE DATA\RGB'
    list_images=os.listdir(PATH_TO_IMAGE_FOLDER)

    list_images = sorted(list_images,key = lambda x: int(x[10:-4]))
    #print('============',len(list_images))
    for img_name in list_images:
        #print('.........',img_name)
        image_path=os.path.join(PATH_TO_IMAGE_FOLDER,img_name)
        image=scale_down(cv2.imread(image_path),4)
        #print('........',image_path)

        Width = image.shape[1]
        Height = image.shape[0]

        detected_img = detect(image,Width,Height)

        cv2.imshow("object detection", image)
        k = cv2.waitKey(250) & 0xff
        
        if k==ord('q'):
            break
        
        #cv2.destroyAllWindows()

    cv2.destroyAllWindows()
elif option == '4' :
    PATH_TO_IMAGE = input('Enter the path to image:')

    image=cv2.imread(PATH_TO_IMAGE)
    Width = image.shape[1]
    Height = image.shape[0]
    scaled_img = scale_down(image,10)
    detected_img = detect(scaled_img, Width, Height)
    cv2.imshow('Detected image',detected_img)
    cv2.waitKey() 
    cv2.destroyAllWindows()



else:
    print('Enter valid option\nExiting......')
