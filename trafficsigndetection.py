from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("best.pt")

classNames = ['bus_stop', 'do_not_enter', 'do_not_stop', 'do_not_turn_l', 'do_not_turn_r', 'do_not_u_turn', 'enter_left_lane', 'green_light', 'left_right_lane', 'no_parking', 'parking', 'ped_crossing', 'ped_zebra_cross', 'railway_crossing', 'red_light', 'stop', 't_intersection_l', 'traffic_light', 'u_turn', 'warning', 'yellow_light'
              ]

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255), 3)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0]*100))/100

            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)


    cv2.imshow("Image", img)
    cv2.waitKey(1)
