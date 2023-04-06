from ultralytics import YOLO
import cv2
import cvzone
import math

#cap = cv2.VideoCapture(0)  # For Webcam

#cap.set(3, 1920)
#cap.set(4, 1080)
cap = cv2.VideoCapture("../Videos/video-cpfl.mp4")  # For Video

model = YOLO("ppe.pt")

classNames = ['Capacete', '-', 'Sem Capacete', '-', 'Sem Vestuario de Seguranca', 'Pessoa', 'Cone de Protecao',
              'Vestuario de Seguranca', '-', '-']
myColor = (0, 0, 255)
#x = 0
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)

            if conf > 0.5:
                if currentClass =='Sem-Capacete' or currentClass =='Sem Vestuario de Seguranca':
                    myColor = (0, 0,255)
                elif currentClass =='Capacete' or currentClass =='Vestuario de Seguranca':
                    myColor =(0,255,0)
                else:
                    myColor = (255, 0, 0)

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1.5, thickness=1,colorB=myColor,
                                   colorT=(255,255,255),colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    
    cv2.imshow("Deteccao de EPI", img)
    #cv2.imwrite('hola{}.jpg'.format(x),img)
    #x+=1
    cv2.waitKey(1)
