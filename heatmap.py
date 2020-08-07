from tensorflow.keras.preprocessing.image import img_to_array
import cv2
from tensorflow.keras import Model

multi_disease_model.load_weights('../input/resnet-weights/weights_epoch_20.h5')
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