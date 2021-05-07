from collections import namedtuple


Configuration = namedtuple(
    'Pipeline_Configuration', 
    [
        'image_size_visual',
        'image_size_nn',
        'visual_frame_tracking',
        'model',
        'labels',
        'tracking_labels',
    ]
)

pipeline_config = Configuration(
    image_size_visual=None,
    image_size_nn=416,
    visual_frame_tracking=False, # this doesn't work yet
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
    tracking_labels=[0, 1, 2, 3, 5, 7],
)