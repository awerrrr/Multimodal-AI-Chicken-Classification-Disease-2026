import cv2
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224
MAX_LEN = 150

# ============================================================
# IMAGE PREPROCESS
# ============================================================
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype(np.float32)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# ============================================================
# TEXT PREPROCESS
# ============================================================
def preprocess_text(text, tokenizer):

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding='post'
    )

    return padded