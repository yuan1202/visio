import os
from pathlib import Path
import numpy as np
import depthai as dai
from config import pipeline_config


# Pipeline definition
def make_pipeline(use_camera):

    pipeline = dai.Pipeline()

    # --------------------------------------------
    # detection model
    detectionNetwork = pipeline.createYoloDetectionNetwork()
    detectionNetwork.setConfidenceThreshold(pipeline_config.detection_confidence)
    detectionNetwork.setNumClasses(80)
    detectionNetwork.setCoordinateSize(4)
    detectionNetwork.setAnchors(np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]))
    detectionNetwork.setAnchorMasks({"side26": np.array([1, 2, 3]), "side13": np.array([3, 4, 5])})
    detectionNetwork.setIouThreshold(0.5)

    model_path = os.path.join(Path(os.path.dirname(__file__)).parent, 'blobs', pipeline_config.model)
    detectionNetwork.setBlobPath(model_path)
    detectionNetwork.setNumInferenceThreads(2)
    detectionNetwork.input.setBlocking(False)

    # --------------------------------------------
    # object tracker
    objectTracker = pipeline.createObjectTracker()
    objectTracker.setDetectionLabelsToTrack(pipeline_config.tracking_labels)  # track vehicle and bicycle [0, 1, 2, 3, 5, 7]
    # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS
    objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
    objectTracker.setTrackerIdAssigmentPolicy(dai.TrackerIdAssigmentPolicy.UNIQUE_ID)

    trackerOut = pipeline.createXLinkOut()
    trackerOut.setStreamName("tracklets")
    objectTracker.out.link(trackerOut.input)

    # --------------------------------------------
    # setup input source
    if use_camera:
        colorCam = pipeline.createColorCamera()
        colorCam.setPreviewSize(pipeline_config.image_size_nn, pipeline_config.image_size_nn)
        colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        colorCam.setInterleaved(False)
        colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        colorCam.setFps(pipeline_config.fps)

        colorCam.preview.link(detectionNetwork.input)
        colorCam.video.link(objectTracker.inputTrackerFrame)

        camOut = pipeline.createXLinkOut()
        camOut.setStreamName("video")
        objectTracker.passthroughTrackerFrame.link(camOut.input)

    else:
        nn_in = pipeline.createXLinkIn()
        nn_in.setStreamName("nn_in")
        nn_in.out.link(detectionNetwork.input)

        detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

    # --------------------------------------------
    # link-up detection model and object tracker
    detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
    detectionNetwork.out.link(objectTracker.inputDetections)

    return pipeline
