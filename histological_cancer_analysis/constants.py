from enum import StrEnum

ID_COLUMN = "id"
LABEL_COLUMN = "label"
RGB_MODE = "RGB"


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
