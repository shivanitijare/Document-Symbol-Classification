import os
import argparse
import PIL
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import re
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

#extract the arguments 
parser = argparse.ArgumentParser(description=
'Augment images, fit the classifier, evaluate accuracy or predict class of symbol')

parser.add_argument('--task', type=str, default='pass',
                    help="""
                    task to perform: 
                    augment_images-->augment a set of images to extend dataset
                    fit-->fit the classifier (and optionaly save) the classifier; 
                    evaluate-->­­­calculate the accuracy on a given set of images
                    classify-->predict the probability that the image is from the possible classes
                    """)

parser.add_argument('--aug', type=str, default=None,
                    help="""
                    Path of the image when we want to perform data augmentation
                    """)

parser.add_argument('--evaluate_directory', type=str, default='test',
                    help="""
                    If we want to evaluate accuracy on images in "train", "val" or "test"
                    """)

parser.add_argument('--img', type=str, default=None,
                    help="""
                    Path of the image when we want to predict its class
                    """)

args = parser.parse_args()

#checking the format of given arguments
if args.task not in ['augment_img', 'fit', 'evaluate', 'classify']:
    print('Task not supported!')
    args.task = 'pass'

if args.task == 'augment_img':    
    if os.path.exists(args.aug):
        aug_path = args.aug
    else:
        print('Unknown path!')
        args.task = 'pass'

if args.task == 'evaluate_directory':    
    if args.evaluate_directory not in ['train', 'val', 'test']:
        print('evaluate_directory has to be train, val or test')
        args.task = 'pass'

if args.task == 'classify':    
    if os.path.exists(args.img):
        img_path = args.img
    else:
        print('Unknown path!')
        args.task = 'pass'


# function to preprocess the image
def preprocess_image(image):


# Read images from dataset
data, labels = [], []
main = "test-subset/"
folder = [os.path.join(main, folder) for folder in os.listdir(main)]
symbols = [os.path.join(d,f) for d in folder for f in os.listdir(d)]

for symbol in symbols:
    image = cv2.imread(symbol)
    image = cv2.resize(image, (32, 32)).flatten()
    data.append(image)
    label = symbol.split(os.path.sep)[-2].split(".")[0]
    label = re.sub('\_\d*', '', label)
    labels.append(label)


# Preprocessing - uniform dimensions? image enhancement?
## scale the raw pixel intensities to the range [0, 1]
data = np.array(data) / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# consult_ifu [1 0 0 0]
# do_not_resterilize [0 1 0 0]
# skeep_dry [0 0 1 0] 
# sterile [0 0 0 1] 



model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))

# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.01
EPOCHS = 10
# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the neural network
H = model.fit(x=trainX, y=trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=32)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))


# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")