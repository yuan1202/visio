from collections import namedtuple


Configuration = namedtuple(
    'Pipeline_Configuration', 
    [
        'visualisation_resize',     # visualisation/recording resize ratio
        'image_size_nn',            # image size for NN inference
        'fps',                      # camera FPS
        'model',                    # detection model file name
        'labels',                   # detection model prediction labels
        'tracking_labels',          # labels to track
        'detection_confidence',     # detection model confidence threshold
        'fast_tracking',            # use nn passthrough for tracking instead of orignal video
    ]
)

pipeline_config = Configuration(
    visualisation_resize=0.6,
    image_size_nn=416,
    fps=30,
    model='tiny-yolo-v4_openvino_2021.2_6shave.blob',
    labels = [
        "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
        "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
        "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
        "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
        "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
        "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
        "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
        "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
        "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
        "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
        "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
        "teddy bear",     "hair drier", "toothbrush"
    ],
    tracking_labels=[1, 2, 3, 5, 7], # bicycle, car, motorbike, bus, truck
    detection_confidence=0.4,
    fast_tracking=False,
)