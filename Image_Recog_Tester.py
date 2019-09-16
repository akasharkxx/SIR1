from PIL import Image
import numpy as np
from keras.models import load_model #for loading trained model from file

# not required in Image_Recog_Trainer
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

model = load_model('trained_model.h5')

input_path = input('Enter image file pathname: ')
input_img = Image.open(input_path)
input_img = input_img.resize((32, 32), resample=Image.LANCZOS)

image_array = np.array(input_img)
image_array = image_array.astype('float32')
image_array /= 255.0
image_array = image_array.reshape(1, 32, 32, 3)
answer = model.predict(image_array)
input_img.show()
print(labels[np.argmax(answer)])