from pdf2image import convert_from_path
from roboflow import Roboflow
import cv2
import pytesseract
from google.colab.patches import cv2_imshow

pages = convert_from_path('/content/Deepanshi_Sharma_Resume (2) (1).pdf', 500)
for count, page in enumerate(pages):
    page.save(f'out{count}.jpg', 'JPEG')

rf = Roboflow(api_key="ANxK7EzOKpkouhcLBQ4J")
project = rf.workspace().project("cvv-dhjla")
model = project.version(1).model
image_path = f'out{count}.jpg'
print(model.predict(image_path, confidence=40, overlap=30).json())
thisdict = model.predict(image_path, confidence=40, overlap=30)
#print(thisdict)

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

img = cv2.imread(image_path)
predictions = thisdict.predictions
for prediction in predictions:
    class_name = prediction['class']
    X = prediction['x']
    Y = prediction['y']
    W = prediction['width']
    H = prediction['height']
    print(f"Class: {class_name}")
    print(f"X: {X}")
    print(f"Y: {Y}")
    print(f"W: {W}")
    print(f"H: {H}")
    block_coordinates = (X ,Y, W, H)
    # Extract the block from the image
    x, y, w, h = block_coordinates
    block = img[y:y+h, x:x+w]
    # Display the extracted block
    cv2_imshow(block)
    cv2.imwrite("block_image1.png", block)
    gray_block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("block_image_gray1.png",gray_block)
    # Apply OCR on the block
    text = pytesseract.image_to_string('/content/block_image_gray1.png', lang='eng')
    # Print the extracted text
    print("Extracted Text:")
    print(text)
