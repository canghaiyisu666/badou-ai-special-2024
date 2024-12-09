from mask_rcnn import MASK_RCNN
from PIL import Image

mask_rcnn = MASK_RCNN()


img = input('img/street.jpg')
try:
    image = Image.open('img/street.jpg')
except:
    print('Open Error! Try again!')
else:
    mask_rcnn.detect_image(image)
mask_rcnn.close_session()
    
