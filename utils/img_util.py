import cv2

def convert_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (WIDTH, HEIGHT), cv2.INTER_AREA)
    return img