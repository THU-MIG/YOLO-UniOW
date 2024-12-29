import itertools

__all__ = ["VOC_COCO_CLASS_NAMES", "VOC_CLASS_NAMES_COCOFIED", "BASE_VOC_CLASS_NAMES", "UNK_CLASS"]


#OWOD splits
VOC_CLASS_NAMES_COCOFIED = [
    "airplane",  "dining table", "motorcycle",
    "potted plant", "couch", "tv"
]

BASE_VOC_CLASS_NAMES = [
    "aeroplane", "diningtable", "motorbike",
    "pottedplant",  "sofa", "tvmonitor"
]
UNK_CLASS = ["unknown"]

VOC_COCO_CLASS_NAMES={}


T1_CLASS_NAMES = [
    "aeroplane","bicycle","bird","boat","bus","car",
    "cat","cow","dog","horse","motorbike","sheep","train",
    "elephant","bear","zebra","giraffe","truck","person"
]

T2_CLASS_NAMES = [
    "traffic light","fire hydrant","stop sign",
    "parking meter","bench","chair","diningtable",
    "pottedplant","backpack","umbrella","handbag",
    "tie","suitcase","microwave","oven","toaster","sink",
    "refrigerator","bed","toilet","sofa"
]

T3_CLASS_NAMES = [
    "frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","banana","apple","sandwich",
    "orange","broccoli","carrot","hot dog","pizza","donut","cake"
]

T4_CLASS_NAMES = [
    "laptop","mouse","remote","keyboard","cell phone","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush",
    "wine glass","cup","fork","knife","spoon","bowl","tvmonitor","bottle"
]

VOC_COCO_CLASS_NAMES["SOWODB"] = tuple(itertools.chain(T1_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))


VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

T2_CLASS_NAMES = [
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator"
]

T3_CLASS_NAMES = [
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake"
]

T4_CLASS_NAMES = [
    "bed", "toilet", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl"
]
VOC_COCO_CLASS_NAMES["MOWODB"] = tuple(itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))

T1_CLASS_NAMES = [
        'vehicle.bicycle',
        'vehicle.motorcycle',
        'vehicle.car',
        'vehicle.bus.bendy',
        'vehicle.bus.rigid',
        'vehicle.truck',
        'vehicle.emergency.ambulance',
        'vehicle.emergency.police',
        'vehicle.construction',
        'vehicle.trailer'
]

T2_CLASS_NAMES = [
        'human.pedestrian.adult',
        'human.pedestrian.child',
        'human.pedestrian.wheelchair',
        'human.pedestrian.stroller',
        'human.pedestrian.personal_mobility',
        'human.pedestrian.police_officer',
        'human.pedestrian.construction_worker'
]

T3_CLASS_NAMES = [
        'movable_object.barrier',
        'movable_object.trafficcone',
        'movable_object.pushable_pullable',
        'movable_object.debris',
        'static_object.bicycle_rack',
        'animal'
]

VOC_COCO_CLASS_NAMES["nuOWODB"] = tuple(itertools.chain(T1_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, UNK_CLASS))
