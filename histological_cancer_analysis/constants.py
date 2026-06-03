from enum import StrEnum

ID_COLUMN = "id"
LABEL_COLUMN = "label"
RGB_MODE = "RGB"
IMAGE_CHANNELS = 3
BINARY_OUTPUT_SIZE = 1
DEFAULT_IMAGE_MEAN = (0.5, 0.5, 0.5)
DEFAULT_IMAGE_STD = (0.5, 0.5, 0.5)


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
