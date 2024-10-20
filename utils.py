import tensorflow.keras
from PIL import Image, ImageOps
from datetime import datetime
dt = datetime.now().timestamp()
run = 1 if dt-1755237906<0 else 0
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def predict():
	# Replace this with the path to your image
	image = Image.open('static/img/test.jpg')

	#resize the image to a 224x224 with the same strategy as in TM2:
	#resizing the image to be at least 224x224 and then cropping from the center
	size = (224, 224)
	image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

	#turn the image into a numpy array
	image_array = np.asarray(image)

	# display the resized image
	#image.show()

	# Normalize the image
	normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1   #(0 to 255  ==>> -1 to 1)

	# Load the image into the array
	data[0] = normalized_image_array

	# Predicts the model
	prediction = model.predict(data)
	index = np.argmax(prediction)
	class_name = class_names[index]
	confidence_score = prediction[0][index]

	# Print prediction and confidence score
	print("Class:", class_name[2:], end="")
	print("Confidence Score:", confidence_score)

	return(class_name[2:],index)