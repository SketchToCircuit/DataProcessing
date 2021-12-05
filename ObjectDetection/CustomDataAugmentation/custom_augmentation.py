class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

def augment(image, boxes):
    '''
    image: Tensor("", shape=(None, None, 3), dtype=float32)
    boxes: Tensor("", shape=(None, 4), dtype=float32)
    '''
    return (image, boxes)