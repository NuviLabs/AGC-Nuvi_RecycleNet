from .coco import CocoDataset
from .builder import DATASETS
### IMPORT YOUR DATASET FORMAT AND PASS IT IN MYDATASET FUNC

@DATASETS.register_module
class Taco_dataset(CocoDataset):

    CLASSES = ('combustion', 'paper', 'steel', 'glass', 'plastic', 'plasticbag', 'styrofoam', 'food',) ## ADD ALL YOUR CLASSES
