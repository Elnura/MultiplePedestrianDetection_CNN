from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims

model = VGG16()

ixs = [2, 5, 9, 13, 17]
outputs = [model.layers[i].output for i in ixs]
model = Model(inputs=model.inputs, outputs=outputs)

img = load_img('D:/PROJECTS/MOT_CNN_DATAASSOCIATION/DATASET/OTB50/David/img/0001.jpg', target_size=(224, 224))

img = img_to_array(img)

img = expand_dims(img, axis=0)

img = preprocess_input(img)

feature_maps = model.predict(img)

square = 8
for fmap in feature_maps:

	ix = 1
	for _ in range(square):
		for _ in range(square):

			ax = pyplot.subplot(square, square, ix)
			ax.set_xticks([])
			ax.set_yticks([])

			pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
			ix += 1
	# show the figure
	pyplot.show()