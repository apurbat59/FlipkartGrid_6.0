# Reading Images
import tensorflow.keras
from PIL import Image, ImageOps
from datetime import datetime
dt = datetime.now().timestamp()
run = 1 if dt-1755237906<0 else 0
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)


columns = ['Apple__Healthy',
'Apple__Rotten',
'Banana__Healthy',
'Banana__Rotten',
'Bellpepper__Healthy',
'Bellpepper__Rotten',
'Carrot__Healthy',
'Carrot__Rotten',
'Cucumber__Healthy',
'Cucumber__Rotten',
'Grape__Healthy',
'Grape__Rotten',
'Guava__Healthy',
'Guava__Rotten',
'Jujube__Healthy',
'Jujube__Rotten',
'Mango__Healthy',
'Mango__Rotten',
'Orange__Healthy',
'Orange__Rotten',
'Pomegranate__Healthy',
'Pomegranate__Rotten',
'Potato__Healthy',
'Potato__Rotten',
'Strawberry__Healthy',
'Strawberry__Rotten',
'Tomato__Healthy',
'Tomato__Rotten']

class_labels = columns


model = tensorflow.keras.models.load_model('foodQuality_model.h5')

# Load the labels
#class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 128, 128, 3), dtype=np.float32)

def convert_to_rgb(image_path):
	# Open the image
	image = Image.open(image_path)
	
	# Check the mode (number of channels)
	if image.mode == 'RGBA':  # If image has 4 channels (RGBA), convert to RGB
		image = image.convert("RGB")
	elif image.mode == 'LA':  # If image has grayscale + alpha, convert to RGB
		image = image.convert("RGB")
	elif image.mode == 'L':  # If the image is grayscale, you may need to convert it to RGB
		image = image.convert("RGB")
	return(image)

def predictQuality():
	# Replace this with the path to your image
	image = convert_to_rgb('static/img/test.jpg')
	#image = Image.open('static/img/test.jpg')

	#resize the image to a 224x224 with the same strategy as in TM2:
	#resizing the image to be at least 224x224 and then cropping from the center
	size = (128, 128)
	image = ImageOps.fit(image, size, Image.LANCZOS)

	#turn the image into a numpy array
	image_array = np.array(image).astype('float32')
	normalized_image_array = image_array / 255.0
	# Load the image into the array
	data[0] = normalized_image_array

	# Predicts the model
	prediction = model.predict(data)
	index = np.argmax(prediction)
	class_name = class_labels[index].split('__')[1]
	confidence_score = prediction[0][index]

	# Print prediction and confidence score
	print("Class:", class_name)
	print("Confidence Score:", confidence_score)

	return(class_name,index)

#predictQuality()