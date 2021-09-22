
class TrainConfig:
    DATA_DIR = 'resources/dataset/'
    MODEL_DIR = 'resources/model/mobilenet_v2-pretrained-ImageNet-1000cl.pth'
    SAVE_FINAL_MODEL = True
    CHECKPOINT_DIR = 'resources/checkpoints'

    EPOCHS = 10
    SCHED_STEP = 3
    BATCH_SIZE = 500
    LR = 0.1
    SHUFFLE_TRAIN = True
