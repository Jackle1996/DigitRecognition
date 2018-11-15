try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\ch1011587\AppData\Local\Tesseract-OCR\tesseract"
print(pytesseract.image_to_string(Image.open('testpic.png')))