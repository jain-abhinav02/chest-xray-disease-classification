from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import cv2
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from model import resnet_v2
from training import weighted_loss

IMG_SIZE = (128,128)
all_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
multi_disease_model = resnet_v2((128,128,1),(9*6)+2,len(all_labels))
multi_disease_model.compile(optimizer = 'adam', loss = weighted_loss)
multi_disease_model.load_weights('../model_weights/weights_epoch_20.h5')

test_df = pd.read_csv('../dataset/test_df.csv')
test_idg = ImageDataGenerator(rescale=1.0/255.0)
test_gen = test_idg.flow_from_dataframe(
        dataframe=test_df,
        directory=None,
        x_col='path',
        y_col=all_labels,
        target_size=(128, 128),
        batch_size=64,
        classes=all_labels,
        class_mode='raw',
        color_mode = "grayscale",
        shuffle=False)

pred_test = multi_disease_model.predict(test_gen, verbose = 1)

img_idx = [] #list of random indices
img_path = [] # list of random image paths corresponding to the above indices
all_idx = list(range(test_df.shape[0])) 
for i in range(len(all_labels)):
    idx = np.random.choice(np.argsort(pred_test[:,i])[-100:])
    print(idx,end=" ")
    img_idx.append(idx)
    img_path.append(test_df.path.iloc[idx])

img_array = []
for i in range(len(all_labels)) :
  cur_path = img_path[i]
  img = cv2.imread(cur_path,0)
  img = cv2.resize(img,IMG_SIZE)
  img_array.append(img_to_array(img,dtype='float32'))
img_array = np.array(img_array)
print(img_array.shape)

def find_target_layer():
  for layer in reversed(multi_disease_model.layers):
    if len(layer.output_shape) == 4:
      return layer.name

def gen_heatmap(input_image,target_class):
    target_layer = multi_disease_model.get_layer(find_target_layer())
    gradModel = Model(inputs = [multi_disease_model.inputs],
                      outputs = [target_layer.output, multi_disease_model.output])
    with tf.GradientTape() as tape:
        convOutputs, pred = gradModel(input_image)
        loss = pred[:,target_class]
    grads = tape.gradient(loss, convOutputs)           
    # use automatic differentiation to compute the gradients
    
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads  
    # compute the guided gradients
    epsilon = 1e-7
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
    heatmap = cv2.resize(cam.numpy(), IMG_SIZE)
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + epsilon  # normalize the heatmap such that all values lie in the range
    heatmap = numer / denom                            # [0, 1], scale the resulting values to the range [0, 255]
    heatmap = (heatmap * 255).astype("uint8")          # and then convert to an unsigned 8-bit integer
    return heatmap

fig, axs = plt.subplots(4, 4,figsize=(16,16))
i = 0
j = 0
for k in range(len(all_labels)):
    idx = img_idx[k]
    target = np.argmax(pred_test[idx,:]) # select target class as the one with highest probability
    heatmap = gen_heatmap(img_array[k:k+1],target)
    axs[i,j].imshow(img_array[k,:,:,0].astype("uint8"))
    axs[i,j].imshow(heatmap,alpha=0.5)
    axs[i,j].set_title(all_labels[target]+" : "+str(pred_test[idx,target]))
    j+=1
    if j==4:
        i += 1
        j = 0
fig.savefig("xray_samples.png")