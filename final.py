import cv2
import torch
from IPython.display import display,Image,clear_output
from twilio import *
import time
from datetime import datetime 
from twilio.rest import Client

fire_detected = False
blast_detected = False
smoke_detected = False
count = 0
recording= False
fps = 5
# desired_fps = 30

model = torch.hub.load('ultralytics/yolov5','custom',path='D:\courses\main_project\\best.pt')
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
out = cv2.VideoWriter(f'D:/courses/main_project/saved_fire_video/{datetime.now().strftime("%H-%M-%S")}.avi',fourcc,fps,(600,400))
# cap = cv2.VideoCapture('D:\courses\main_project\Explosion2.mp4')
cap = cv2.VideoCapture(0)


# def twiliomsg(output1):



#     account_sid = ''
#     auth_token = ''
#     client = Client(account_sid, auth_token)

#     message = client.messages.create(
#     from_='+12294412109',
#     body=output1,
#     to='+919373631532'
#     )


#     print(message.sid)






while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break


    frame = cv2.resize(frame, (640, 480))
    # Perform inference on the frame
    results = model(frame)
    cv2.putText(frame,f'{datetime.now().strftime("%D-%H-%M-%S")}',(50,50),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,0),1)
    # Extract information from results
    predictions = results.xyxy[0].cpu().numpy()

    # Process and visualize predictions
    for pred in predictions:
        x1, y1, x2, y2, confidence, class_index = pred
        class_label = model.names[int(class_index)]

        if class_label == 'fire' or 'blast' and confidence > 0.65 :
            fire_detected = True
            blast_detected = True
            output = 'Blast or Fire Detected'
            print(output)
            # twiliomsg(output)
            start_recording_time = time.time()
            recording=True



        if class_label == 'smoke' and confidence > 0.65 :
            count += 1
            if count == 5:
                time.sleep(3)
                smoke_detected = True
                output = 'Smoke Detected'
                print(output)
                # twiliomsg(output)
                start_recording_time = time.time()
                recording=True
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_label}: {confidence:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame with predictions
    display(Image(data=cv2.imencode('.jpg', frame_rgb)[1]))
    # time.sleep(1 / desired_fps)
    # Clear the previous output for a smooth video display
    clear_output(wait=True)
    if recording==True:
        out.write(frame)

        if time.time() - start_recording_time >= 10:  # Adjust the duration as needed
            recording = False
            out.release()
            print("Recording stopped.")
    cv2.imshow('video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






