import os
import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image_folder', required=True,
                help = 'path to input image folder')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()

def scale_down(f,scale):
    dims = [f.shape[0],f.shape[1]]
    
    new_f = cv2.resize(f,(dims[1]//scale,dims[0]//scale),interpolation=cv2.INTER_AREA)
    
    return new_f

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    txt= label + "," + str(round(confidence,4))
    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
Width = None
Height = None
scale = 0.00392

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

def detect(img):

    blob = cv2.dnn.blobFromImage(img, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4




    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


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

    

img_folder_path = args.image_folder
list_images=os.listdir(img_folder_path)

list_images = sorted(list_images,key = lambda x: int(x[10:-4]))
print('============',len(list_images))
for img_name in list_images:
    #print('.........',img_name)
    image_path=os.path.join(img_folder_path,img_name)
    image=scale_down(cv2.imread(image_path),4)
    print('........',type(image))

    Width = image.shape[1]
    Height = image.shape[0]

    detected_img = detect(image)

    cv2.imshow("object detection", image)
    k = cv2.waitKey(250) & 0xff
    
    if k==ord('q'):
        break
    
    #cv2.destroyAllWindows()

cv2.destroyAllWindows()
