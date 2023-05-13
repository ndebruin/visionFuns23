import numpy as np
import cv2
from math import asin

# step 1 - load the model

net = cv2.dnn.readNet('best.onnx')

# step 2 - feed a 640x640 image to get predictions

num_of_markers = -1

def format_yolov5(frame):

    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result

def find_dist(pixels):
    return 3105/pixels


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)


while True:
    ret, frame = cap.read()

    #frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameSmooth = cv2.edgePreservingFilter(frame, flags=1, sigma_s=60, sigma_r=0.4)
    smoothHSV = cv2.cvtColor(frameSmooth, cv2.COLOR_BGR2HSV)


    input_image = format_yolov5(frame) # making the image square
    blob = cv2.dnn.blobFromImage(input_image , 1/255.0, (640, 640), swapRB=True)
    net.setInput(blob)
    predictions = net.forward()

    # step 3 - unwrap the predictions to get the object detections 

    class_ids = []
    confidences = []
    boxes = []

    output_data = predictions[0]

    image_width, image_height, _ = input_image.shape
    x_factor = image_width / 640
    y_factor =  image_height / 640

    for r in range(25200):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    class_list = []
    with open("classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    if(len(result_class_ids) != num_of_markers):
        print(f"{len(result_class_ids)} Expo Markers found.")
        num_of_markers = len(result_class_ids)

    for i in range(len(result_class_ids)):
        
        box = result_boxes[i]
        class_id = result_class_ids[i]

        # calculate color
        centerX = box[0] + round(box[2]/2)
        centerY = box[1] + round(box[3]/2)

        xStart = box[0]
        yStart = box[1]
        
        highestSat = 0
        highHue = 0
        highVal = 0

        # find the highest saturation thing in the detection box
        for xDiff in range(box[2]):
            for yDiff in range(box[3]):
                h,s,v = smoothHSV[yStart+yDiff][xStart+xDiff]
                if(s > highestSat):
                    highestSat = s
                    highHue = h
                    highVal = v

        # convert hsv to bgr
        hsv = np.uint8([[[highHue,highestSat,highVal]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        b,g,r = bgr[0][0]

        # distance
        largest = box[2]
        if (box[3] > box[2]):
            largest = box[3]
        
        # weird math
        dist = find_dist(largest)
        dist = round(dist, 2)

        # calculate angles
        
        # center of image
        imgCenterX = 640/2
        imgCenterY = 480/2

        # diffs
        # xDelta = abs(centerX - imgCenterX)
        # yDelta = abs(centerY - imgCenterY)
        
        # xDelta = find_dist(xDelta)
        # YDelta = find_dist(yDelta)

        # # calculate angles
        # xTheta = xDelta/dist
        # yTheta = yDelta/dist

        # print(f"{xTheta} {yTheta}")
        

        cv2.rectangle(frame, box, (int(b), int(g), int(r)), 2)
        # label boxes
        cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (int(b), int(g), int(r)), -1)
        cv2.putText(frame, f"{dist}\"", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))

    cv2.imshow("output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()