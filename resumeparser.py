from pdf2image import convert_from_path
from PIL import Image
from roboflow import Roboflow
import pytesseract
import cv2
from google.colab.patches import cv2_imshow
import os

pytesseract.pytesseract.tesseract_cmd= '/usr/bin/tesseract'

path = r'/content/drive/MyDrive/data/ACCOUNTANT'

out = r'/content'

pdf_files = [f for f in os.listdir(path) if f.endswith(".pdf")]
#print(pdf_files)

GREEN = '\033[92m'
RESET = '\033[0m'

for pdf_file in pdf_files:
  # Construct the full path for the PDF file
  pdf_path = os.path.join(path, pdf_file)
  pages = convert_from_path(pdf_path)
  for i, image in enumerate(pages):
      pic = os.path.join(out, f"{pdf_file}_page_{i + 1}.png")
      image.save(pic)
      #print(pic)

      print(GREEN+'==========================================================================================================================='+RESET)
      print(GREEN+"RESUME ID:", pdf_file+RESET)
      print(GREEN+'==========================================================================================================================='+RESET)
      rf = Roboflow(api_key="VNchzCIrEh3cIH92zeY8")
      project = rf.workspace("kkkk-mlls6").project("resum")
      model = project.version(1).model
      result = model.predict(pic)
      #print(result)

      # Read the image
      img = cv2.imread(pic)
      block_coordinates = []
      predictions = result.predictions
      for prediction in predictions:
        class_name = prediction['class']
        x = prediction['x']
        y = prediction['y']
        w = prediction['width']
        h = prediction['height']
        print(f"Class: {class_name}")
        print(f"x: {x}")
        print(f"y: {y}")
        print(f"w: {w}")
        print(f"h: {h}")

      # Extract the block from the image
        block = img[y:y+h, x:x+w]

      # Display the extracted block
        cv2_imshow(block)
        cv2.imwrite("output.png",block)
        gray_block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("block_image_gray1.png",gray_block)

      # Apply OCR on the block
        text = pytesseract.image_to_string('/content/block_image_gray1.png', lang='eng')

      # Print the extracted text
        print("Extracted Text:")
        print(text)

