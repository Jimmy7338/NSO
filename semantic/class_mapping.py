"""
YOLO COCO类别到自定义语义标签集的映射配置
支持更多室内物体类别（桌子、茶杯、枕头等）
"""

# YOLOv8 COCO 80类类别名称（按索引顺序）
COCO_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# 自定义语义标签集（重点关注室内物体）
CUSTOM_SEMANTIC_LABELS = [
    # 家具类
    'chair',           # 椅子
    'couch',           # 沙发
    'bed',             # 床
    'dining table',    # 餐桌
    'bench',           # 长椅/凳子
    # 餐具类
    'cup',             # 茶杯/杯子
    'bottle',          # 瓶子
    'bowl',            # 碗
    'wine glass',      # 酒杯
    'fork',            # 叉子
    'knife',           # 刀
    'spoon',           # 勺子
    # 床上用品
    'pillow',          # 枕头（需要从其他类别映射，COCO没有直接类别）
    # 电子设备
    'tv',              # 电视
    'laptop',          # 笔记本
    'mouse',           # 鼠标
    'keyboard',        # 键盘
    'cell phone',      # 手机
    'remote',          # 遥控器
    # 家电
    'microwave',       # 微波炉
    'oven',            # 烤箱
    'toaster',         # 烤面包机
    'refrigerator',    # 冰箱
    'sink',            # 水槽
    # 其他室内物品
    'book',            # 书
    'clock',           # 时钟
    'vase',            # 花瓶
    'potted plant',    # 盆栽
    'toilet',          # 马桶
    'scissors',        # 剪刀
    'teddy bear',      # 泰迪熊
    'hair drier',      # 吹风机
    'toothbrush',      # 牙刷
    # 个人物品
    'backpack',        # 背包
    'umbrella',        # 雨伞
    'handbag',         # 手提包
    'tie',             # 领带
    'suitcase',        # 行李箱
    # 食物（可能在室内）
    'banana',          # 香蕉
    'apple',           # 苹果
    'sandwich',        # 三明治
    'orange',          # 橙子
    'broccoli',        # 西兰花
    'carrot',          # 胡萝卜
    'hot dog',         # 热狗
    'pizza',           # 披萨
    'donut',           # 甜甜圈
    'cake',            # 蛋糕
    # 人物和动物
    'person',          # 人
    'cat',             # 猫
    'dog',             # 狗
    'bird',            # 鸟
]

# YOLO类别ID到自定义语义标签ID的映射
# 如果YOLO类别不在自定义标签集中，映射到-1（忽略）
YOLO_TO_CUSTOM_MAPPING = {}

# 初始化映射表
for yolo_id, yolo_name in enumerate(COCO_CLASS_NAMES):
    if yolo_name in CUSTOM_SEMANTIC_LABELS:
        YOLO_TO_CUSTOM_MAPPING[yolo_id] = CUSTOM_SEMANTIC_LABELS.index(yolo_name)
    else:
        YOLO_TO_CUSTOM_MAPPING[yolo_id] = -1  # 忽略此类别

# 特殊映射：将某些类别映射到"pillow"（COCO没有pillow，可以用couch/bed等近似）
# 这里我们暂时不映射，如果检测到枕头相关的物体，可以通过扩展COCO类别或使用其他模型
# 如果需要，可以添加：
# YOLO_TO_CUSTOM_MAPPING[COCO_CLASS_NAMES.index('couch')] = CUSTOM_SEMANTIC_LABELS.index('pillow')  # 示例

# 反向映射：自定义标签ID到YOLO类别ID列表（一个自定义标签可能对应多个YOLO类别）
CUSTOM_TO_YOLO_MAPPING = {}
for custom_id, custom_name in enumerate(CUSTOM_SEMANTIC_LABELS):
    CUSTOM_TO_YOLO_MAPPING[custom_id] = [
        yolo_id for yolo_id, yolo_name in enumerate(COCO_CLASS_NAMES)
        if yolo_name == custom_name
    ]


def map_yolo_to_custom(yolo_class_ids: list) -> list:
    """
    将YOLO类别ID列表映射到自定义语义标签ID列表
    返回: 映射后的类别ID列表（忽略的类别会被过滤掉）
    """
    mapped_ids = []
    for yolo_id in yolo_class_ids:
        custom_id = YOLO_TO_CUSTOM_MAPPING.get(yolo_id, -1)
        if custom_id >= 0:  # 只保留在自定义标签集中的类别
            mapped_ids.append(custom_id)
    return mapped_ids


def get_num_custom_classes() -> int:
    """返回自定义语义类别数量"""
    return len(CUSTOM_SEMANTIC_LABELS)


def get_custom_class_name(class_id: int) -> str:
    """根据自定义类别ID获取类别名称"""
    if 0 <= class_id < len(CUSTOM_SEMANTIC_LABELS):
        return CUSTOM_SEMANTIC_LABELS[class_id]
    return "unknown"


# 室内场景类别白名单（过滤掉不合理的室外物体）
# 注意：COCO数据集没有"衣柜"类别，但我们可以通过其他方式识别
# 例如：大的矩形物体可能是衣柜、书柜等
INDOOR_CLASS_WHITELIST = [
    # 人物和动物（可能在室内）
    'person', 'cat', 'dog', 'bird',
    # 家具（注意：COCO没有衣柜，但可以检测其他家具）
    'chair', 'couch', 'bed', 'dining table', 'bench',
    # 餐具和容器
    'cup', 'bottle', 'bowl', 'wine glass', 'fork', 'knife', 'spoon',
    # 电子设备
    'tv', 'laptop', 'mouse', 'keyboard', 'cell phone', 'remote',
    # 家电
    'microwave', 'oven', 'toaster', 'refrigerator', 'sink',
    # 其他室内物品
    'book', 'clock', 'vase', 'potted plant', 'toilet',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
    # 食物（可能在室内）
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake',
    # 运动用品（可能在室内）
    'sports ball', 'tennis racket', 'baseball bat', 'baseball glove',
    'skateboard', 'frisbee',
]


def is_indoor_class(yolo_class_id: int) -> bool:
    """判断YOLO类别ID是否属于室内场景类别"""
    if 0 <= yolo_class_id < len(COCO_CLASS_NAMES):
        class_name = COCO_CLASS_NAMES[yolo_class_id]
        return class_name in INDOOR_CLASS_WHITELIST
    return False


def filter_indoor_classes(yolo_class_ids: list, scores: list = None) -> tuple:
    """
    过滤YOLO类别，只保留室内场景相关类别
    返回: (过滤后的类别ID列表, 过滤后的分数列表)
    """
    filtered_ids = []
    filtered_scores = []
    for i, cls_id in enumerate(yolo_class_ids):
        if is_indoor_class(int(cls_id)):
            filtered_ids.append(cls_id)
            if scores is not None:
                filtered_scores.append(scores[i])
    if scores is not None:
        return filtered_ids, filtered_scores
    return filtered_ids

