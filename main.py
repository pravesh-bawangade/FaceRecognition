'''

To execute simply run:
main.py

To input new user:
main.py --mode "input"

'''
import os
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import cv2
from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
from tf_graph import FaceRecGraph
import argparse
import sys
import json
import time
import numpy as np
import credentials as cr
from imutils.video import FPS

TIMEOUT = 10 #10 seconds


def email_send(img):
    img = rescale_frame(img, percent=50)
    path_image = cr.PATH
    cv2.imwrite(os.path.join(path_image, 'intruder.jpg'), img)
    fromaddr = cr.FROMADDR  # Type your email address
    toaddr = cr.TOADDR  # Type email address to whom you want to send
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "Intruder Detected"  # Type subject of your mail

    body = "It is recommended to check surveillance feed."  # Type your message body
    msg.attach(MIMEText(body, 'plain'))

    filename = "intruder.jpg"
    attachment = open(cr.PATH + "/intruder.jpg", "rb")

    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

    msg.attach(part)
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, cr.PASSWORD)  # Type your email password
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def main(args):
    mode = args.mode
    camera_type = args.type
    if(mode == "camera"):
        camera_recog(camera_type)
    elif mode == "input":
        create_manual_data(camera_type)
    else:
        raise ValueError("Unimplemented mode")
'''
Description:
Images from Video Capture -> detect faces' regions -> crop those faces and align them 
    -> each cropped face is categorized in 3 types: Center, Left, Right 
    -> Extract 128D vectors( face features)
    -> Search for matching subjects in the dataset based on the types of face positions. 
    -> The preexisitng face 128D vector with the shortest distance to the 128D vector of the face on screen is most likely a match
    (Distance threshold is 0.6, percentage threshold is 70%)
    
'''
def camera_recog(camera_type = 0):
    print("[INFO] camera sensor warming up...")
    vs = cv2.VideoCapture(camera_type) #get input from webcam
    detect_time = time.time()
    tracker = cv2.TrackerKCF_create()
    rects = []
    recog_data = None
    key = None
    fps = None
    diff = 0
    while True:

        _,frame = vs.read();

        if (not (len(rects) == 0)) and (not (diff) >= 5.0):
            (success, box) = tracker.update(frame)

            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0)) #draw bounding box for the face
                cv2.putText(frame,recog_data[i][0]+" - "+str(recog_data[i][1])+"%",(x,y),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)

            fps.update()
            fps.stop()
            cv2.putText(frame, str(fps.fps()), (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

        else:

            rects, landmarks = face_detect.detect_face(frame, 80)  # min face size is set to 80x80
            aligns = []
            positions = []

            for (i, rect) in enumerate(rects):
                aligned_face, face_pos = aligner.align(160, frame, landmarks[:, i])
                if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
                    aligns.append(aligned_face)
                    positions.append(face_pos)
                else:
                    print("Align face failed")  # log
            if (len(aligns) > 0):
                features_arr = extract_feature.get_features(aligns)
                recog_data = findPeople(features_arr, positions, frame)

            print(rects)

            if not (len(rects) == 0):
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, (rects[0][0], rects[0][1], rects[0][2]-rects[0][0], rects[0][3]-rects[0][1]))
            detect_time = time.time()
        fps = FPS().start()

        if key == ord("q"):
            break
        done = time.time()
        diff = (done - detect_time)


'''
facerec_128D.txt Data Structure:
{
"Person ID": {
    "Center": [[128D vector]],
    "Left": [[128D vector]],
    "Right": [[128D Vector]]
    }
}
This function basically does a simple linear search for 
^the 128D vector with the min distance to the 128D vector of the face on screen
'''
def findPeople(features_arr, positions, frame, thres = 0.6, percent_thres = 70):
    '''
    :param features_arr: a list of 128d Features of all faces on screen
    :param positions: a list of face position types of all faces on screen
    :param thres: distance threshold
    :return: person name and percentage
    '''
    f = open('./facerec_128D.txt','r')
    data_set = json.loads(f.read())
    returnRes = []
    for (i,features_128D) in enumerate(features_arr):
        result = "Unknown"
        smallest = sys.maxsize
        for person in data_set.keys():
            person_data = data_set[person][positions[i]]
            for data in person_data:
                distance = np.sqrt(np.sum(np.square(data-features_128D)))
                if(distance < smallest):
                    smallest = distance
                    result = person
        percentage =  min(100, 100 * thres / smallest)
        if percentage <= percent_thres :
            result = "Unknown"
            #email_send(frame)    # uncomment this line to enable email feature
        returnRes.append((result,percentage))
    return returnRes    

'''
Description:
User input his/her name or ID -> Images from Video Capture -> detect the face -> crop the face and align it 
    -> face is then categorized in 3 types: Center, Left, Right 
    -> Extract 128D vectors( face features)
    -> Append each newly extracted face 128D vector to its corresponding position type (Center, Left, Right)
    -> Press Q to stop capturing
    -> Find the center ( the mean) of those 128D vectors in each category. ( np.mean(...) )
    -> Save
    
'''
def create_manual_data(camera_type= 0):
    vs = cv2.VideoCapture(camera_type) #get input from webcam
    print("Please input new user ID:")
    new_name = input(); #ez python input()
    f = open('./facerec_128D.txt','r')
    data_set = json.loads(f.read())
    person_imgs = {"Left" : [], "Right": [], "Center": []}
    person_features = {"Left" : [], "Right": [], "Center": []}
    print("Please start turning slowly. Press 'q' to save and add this new user to the dataset")
    while True:
        _, frame = vs.read()
        rects, landmarks = face_detect.detect_face(frame, 80)  # min face size is set to 80x80
        for (i, rect) in enumerate(rects):
            aligned_frame, pos = aligner.align(160,frame,landmarks[:,i])
            if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:
                person_imgs[pos].append(aligned_frame)
                cv2.imshow("Captured face", aligned_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    for pos in person_imgs: #there r some exceptions here, but I'll just leave it as this to keep it simple
        person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]),axis=0).tolist()]
    data_set[new_name] = person_features
    f = open('./facerec_128D.txt', 'w')
    f.write(json.dumps(data_set))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="Run camera recognition", default="camera")
    parser.add_argument("--type", type=str, help="camera type", default=0)
    args = parser.parse_args(sys.argv[1:])
    FRGraph = FaceRecGraph()
    MTCNNGraph = FaceRecGraph()
    aligner = AlignCustom()
    extract_feature = FaceFeature(FRGraph)
    face_detect = MTCNNDetect(MTCNNGraph, scale_factor=2) #scale_factor, rescales image for faster detection
    main(args)
