import os
from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import platform

from .utils import to_planar
from .pipeline import make_pipeline
from config import pipeline_config


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
        with dai.Device(make_pipeline(self.video is None)) as device:

            # Start the pipeline
            device.startPipeline()

            if self.video is not None:
                frame_source = cv2.VideoCapture(self.video)
                framerate = frame_source.get(cv2.CAP_PROP_FPS)
            else:
                frame_source = device.getOutputQueue("video", 4, False)
            
            tracklets = device.getOutputQueue("tracklets", 4, False)
            
            startTime = time.monotonic()
            counter = 0
            fps = 0
            frame = None
            writer = None

            seq_num = 0

            while(True):

                # get input frame
                if self.video is not None:
                    read_correctly, origFrame = frame_source.read()
                    if not read_correctly:
                        break
                else:
                    origFrame = frame_source.get().getCvFrame()
                
                # resize image frame for visualisation if configured
                H, W, _ = origFrame.shape
                if type(pipeline_config.visualisation_resize) == float:
                    H = np.round(H * pipeline_config.visualisation_resize).astype(int)
                    W = np.round(W * pipeline_config.visualisation_resize).astype(int)
                    frame = cv2.resize(origFrame, (W, H))
                else:
                    frame = origFrame
                
                # create video recorder
                # do it here t0 access image size conveniently
                if self.record and (writer is None):
                    output_name = time.strftime('visio_recording_GMT-%Y%b%d-%HH%MM%SS.avi', time.gmtime())
                    if self.video is not None:
                        output_name = os.path.basename(self.video).split('.')[0] + '_' + output_name

                    writer = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'MJPG'), 10, (W,  H))

                # send image frame to detection network if input is video
                if self.video is not None:
                    nnFrame = dai.ImgFrame()
                    nnFrame.setType(dai.ImgFrame.Type.BGR888p)
                    nnFrame.setSequenceNum(seq_num)
                    nnFrame.setWidth(pipeline_config.image_size_nn)
                    nnFrame.setHeight(pipeline_config.image_size_nn)
                    nnFrame.setData(to_planar(origFrame, (pipeline_config.image_size_nn, pipeline_config.image_size_nn)))
                    device.getInputQueue("nn_in").send(nnFrame)

                # get tracking results
                seq_num += 1
                track = tracklets.get()
                
                # some basic stuff
                counter+=1
                current_time = time.monotonic()
                if (current_time - startTime) > 1 :
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                color = (255, 0, 0)

                # process tracking results
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
                        label = pipeline_config.labels[t.label]
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
                    # close down
                    if self.record is not None:
                        writer.release()
                    if self.video is not None:
                        frame_source.release()
                    break