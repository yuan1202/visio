from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse

from .utils import to_planar
from .pipeline import make_pipeline


class Visio:

    def __init__(self, **args):
        
        # set attributes
        for k, v in args.items():
            setattr(self, k, v)

    def run(self):
        
        # setup bluetooth stuff
        if self.bluetooth is True:
            import serial
            btSerial = serial.Serial("/dev/ttyAMA0", baudrate=9600, timeout=0.5)
            is_rpi = platform.machine().startswith('arm') or platform.machine().startswith('aarch64')

        # main loop
        with dai.Device(make_pipeline()) as device:

            # Start the pipeline
            device.startPipeline()

            vid_cap = cv2.VideoCapture(self.video)
            framerate = vid_cap.get(cv2.CAP_PROP_FPS)
            tracklets = device.getOutputQueue("tracklets", 4, False)

            startTime = time.monotonic()
            counter = 0
            fps = 0
            frame = None
            writer = None

            seq_num = 0

            while(True):

                # get video frame
                read_correctly, origFrame = vid_cap.read()

                if not read_correctly:
                    break

                if self.image_size_visual is not None:
                    H = W = self.image_size_visual
                    frame = cv2.resize(origFrame, (W, H))
                else:
                    H, W, _ = origFrame.shape
                    frame = origFrame

                # create video recorder
                # do it here t0 access image size conveniently
                if (self.record is not None) and (writer is None):
                    writer = cv2.VideoWriter(self.record, cv2.VideoWriter_fourcc(*'MJPG'), 10, (W,  H))


                # process for detection network
                nnFrame = dai.ImgFrame()
                nnFrame.setType(dai.RawImgFrame.Type(8))
                nnFrame.setSequenceNum(seq_num)
                nnFrame.setWidth(self.image_size_nn)
                nnFrame.setHeight(self.image_size_nn)
                nnFrame.setData(to_planar(origFrame, (self.image_size_nn, self.image_size_nn)))
                device.getInputQueue("nn_in").send(nnFrame)

                # process for tracker
                if self.visual_frame_tracking:
                    trackerFrame = dai.ImgFrame()
                    trackerFrame.setType(dai.RawImgFrame.Type(8))
                    trackerFrame.setSequenceNum(seq_num)
                    trackerFrame.setWidth(W)
                    trackerFrame.setHeight(H)
                    trackerFrame.setData(to_planar(origFrame, (W, H)))
                    device.getInputQueue("tracker_in").send(trackerFrame)

                seq_num += 1
                track = tracklets.get()

                counter+=1
                current_time = time.monotonic()
                if (current_time - startTime) > 1 :
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                color = (255, 0, 0)
                trackletsData = track.tracklets

                for t in trackletsData:
                    
                    if t.status == dai.Tracklet.TrackingStatus.LOST:
                        continue

                    roi = t.roi.denormalize(W, H)
                    x1 = int(roi.topLeft().x)
                    y1 = int(roi.topLeft().y)
                    x2 = int(roi.bottomRight().x)
                    y2 = int(roi.bottomRight().y)

                    # bluetooth interfacing
                    if self.bluetooth is True:
                        area = (x2 - x1) * (y2 - y1)
                        if t.status == dai.Tracklet.TrackingStatus.TRACKED:
                            #If the object is still tracked compare with the previous frame and check if is closer
                            #if id in previous_frame_dict and (previous_frame_dict[id]['area'] < current_frame_dict[id]['area']) and current_frame_dict[id]['area'] > 450:
                            if area > 10000:
                                print("ALERT ID {} IS CLOSE INMINENT IMPACT".format(t.id))
                                btSerial.write("a".encode())
                            #elif id in previous_frame_dict and (previous_frame_dict[id]['area'] < current_frame_dict[id]['area']) and current_frame_dict[id]['area'] > 100:
                            elif area > 8000:
                                print("Warning ID {} is getting closer".format(t.id))
                                btSerial.write("w".encode())
                            else:
                                btSerial.write("s".encode())

                    try:
                        label = self.labels[t.label]
                    except:
                        label = t.label

                    statusMap = {dai.Tracklet.TrackingStatus.NEW : "NEW", dai.Tracklet.TrackingStatus.TRACKED : "TRACKED", dai.Tracklet.TrackingStatus.LOST : "LOST"}
                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, statusMap[t.status], (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

                cv2.imshow("tracker", frame)

                # write record frame
                if self.record is not None:
                    writer.write(frame)

                if cv2.waitKey(1) == ord('q'):
                    # close video recorder
                    if self.record is not None:
                        writer.release()
                    vid_cap.release()
                    break