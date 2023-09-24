# Using flask to make an api
# import necessary libraries and functions
from flask import Flask, request
from flask_cors import CORS, cross_origin
import numpy as np
import cv2
import pytesseract
from pytesseract import Output
from deep_translator import GoogleTranslator
import imutils

# creating a Flask app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# on the terminal type: curl http://127.0.0.1:5000/
# returns hello world when we use GET.
# returns the data that we send when we use POST.
@app.route('/api/uploadfile', methods=['POST'])
@cross_origin()
def home():
    image = cv2.imdecode(np.fromstring(request.files['myFile'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # do some OSD -> make sure orientation is okay
    results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)

    # rotate the image to correct the orientation
    rotated = imutils.rotate_bound(image, angle=results["rotate"])

    # use Tesseract to OCR the image, then replace newline characters
    # with a single space
    text = pytesseract.image_to_string(rotated)

    text = text.replace("\n", " ")

    language = request.args.get('language')

    # translate the text to a different language
    translated = (GoogleTranslator(source='auto', target=language)
                  .translate(text))
    return {'original_text': text, 'translated_to': translated}


# This route gets a list of languages we can translate to
@app.route('/api/supported_languages', methods=['GET'])
@cross_origin()
def supported_languages():
    languages_dict = GoogleTranslator().get_supported_languages(as_dict=True)
    parsed_dict = [{'value': v, 'label': k} for k,v in languages_dict.items()]
    return parsed_dict


# driver function
if __name__ == '__main__':
    app.run(debug=True)
