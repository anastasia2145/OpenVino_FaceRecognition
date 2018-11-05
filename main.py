#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function

import datetime
import sys
from threading import Thread
from time import time

import cv2
import logging as log
# import telegram
# from telegram.utils.request import Request
from utils.drawing import drawrect, draw_text_on_image, SCALAR_GREEN, SCALAR_RED
from utils.openvino_face_database import OpenvinoFaceDataBase
from utils.phantom_sort_tracker import PhantomSortTracker

token = "633139202:AAH9a8-GacSVjBPj88qjY9NOLjFigL2mQCw"
# openvino_recognitions = -1001331011999
# openvino_people = -1001317607609
openvino_people = -230348484
openvino_recognitions = -317434647
# pp = Request(proxy_url="socks5h://51.15.115.236:1080",
             # urllib3_proxy_kwargs={"username": "susr", "password": "password"})
# bot = telegram.Bot(token, request=pp)


def send_recognition(text):
    while True:
        try:
            bot.send_message(openvino_recognitions, text=text)
            break
        except Exception as e:
            print(e)


def send_person(text):
    while True:
        try:
            bot.send_message(openvino_people, text=text)
            break
        except Exception as e:
            print(e)


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    face_database = OpenvinoFaceDataBase(log, "./data/ms_faces", train=False)

    frames = 0

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('./output.mp4', fourcc, 30.0, (1800, 1200))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1800,1200))

    # Start sync inference
    log.info("Starting inference for video")

    start = time()

    # input_video = cv2.VideoCapture("rtsp://admin:admin@80.254.49.38:49464/out.h264")
    input_video = cv2.VideoCapture("./GrowFood_video/79.mp4")

    # tracker = PhantomSortTracker(0, 0, 2304, 1296, 0.5, 0.3, 3, 0.5, 10, 30, 3, 0.75, 4, 0, 0, 0, 0, 0.5, 0.55)
    tracker = PhantomSortTracker(0, 0, 1800, 1200, 0.5, 0.3, 5, 0.5, 5, 10, 3, 0.75, 4, 0, 0, 0, 0, 0.5, 0.55)
    recognized_tracks = set()

    # aligned_counter = 0
    while True:
        # Capture frame-by-frame
        ret, frame = input_video.read()
        
        # if not ret:
            # input_video = cv2.VideoCapture("rtsp://admin:admin@80.254.49.38:49464/out.h264")
            # continue
            # break
        
        frames += 1
        # if frames < 2000:
            # continue
        
        if frames % 1000 == 0:
            print("Handled {} frames in {} s".format(frames, time() - start))
        print("frames", frames)

        if frames % 1 == 0:
            #ПОСМОТРИМ, ЧТО ЗА КАДР
            # cv2.imshow('frame', frame)
            # cv2.waitKey(1)
            print("start")
            fr = cv2.resize(frame, (128, 128))
            check = face_database.face_to_vector(fr)
            print("check", check)
            aligned_faces = face_database.detect_and_identify(frame)
            # И ЧТО ДЛЯ НЕГО НАЙДЕТ 
            print("aligned_faces", aligned_faces)
            boxes = []
            for left, right, top, bottom, name, confidence, aligned_face, age, gender in aligned_faces:
                # filename = "./data/aligned/" + str(aligned_counter) + ".png"
                # cv2.imwrite(filename, aligned_face)
                # aligned_counter += 1
                # thread = Thread(target=send_recognition,
                # args =(datetime.datetime.now().strftime("%H:%M:%S ")
                # + ": " + str(name) + " " + str(right - left) + " " + str(dist), ))
                # thread.start()
                if confidence >= 0:
                    box = (left, top, right, bottom, confidence, name)
                    boxes.append(box)                    
                    draw_text_on_image(frame, gender + " " + str(age), (left, bottom + 60))
            tracker.update_all([boxes], frames)
            
            recognized = []
            for track in tracker.deleted_tracks:
                if track.counted and track.track_id not in recognized_tracks:
                    recognized.append(track)
            for track in tracker.tracks:
                if track.counted and track.track_id not in recognized_tracks:
                    recognized.append(track)
            for track in recognized:
                for box in track.boxes:
                    if not box.phantom and (box.right - box.left) >= 70:
                        recognized_tracks.add(track.track_id)
                        names = {}
                        for b in track.boxes:
                            if b.person not in names:
                                names[b.person] = b.confidence
                            else:
                                names[b.person] = max(names[b.person], b.confidence)

                        track.person = max(names, key=names.get)
                        log.info(track.person)
                        # thread = Thread(target=send_person,
                        # args=(datetime.datetime.now().strftime("%H:%M:%S ") + track.person,))
                        # thread.start()
                        break

            for track in tracker.tracks:
                box = track.boxes[-1]
                style = "dotted"
                if box.phantom:
                    style = "dashed"
                color = SCALAR_GREEN
                name = box.person
                if track.track_id in recognized_tracks:
                    color = SCALAR_RED
                    name = track.person

                drawrect(frame, (box.left, box.bottom), (box.right, box.top), color, thickness=3, style=style)
                draw_text_on_image(frame, name, (box.left, box.bottom))
                draw_text_on_image(frame, str(float(box.confidence)), (box.left, box.bottom + 30))

            tracker.deleted_tracks.clear()
            
        # Display the resulting frame
        # cv2.imshow('frame', frame)
        # cv2.waitKey(1)
        # frame = cv2.resize(frame, (1800, 1200))
        out.write(frame)

    print("Handled {} frames in {} s".format(frames, time() - start))
    print("123")

    # When everything done, release the capture
    input_video.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)
