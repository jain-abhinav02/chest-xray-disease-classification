from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from model import resnet_v2

#creating an Image Data generator
IMG_SIZE = (128, 128)
core_idg = ImageDataGenerator(rescale=1.0/255.0, validation_split = 0.06)

# obtaing the training images using the above generator
train_gen = core_idg.flow_from_dataframe(
        dataframe=train_df,
        directory=None,
        x_col='path',
        y_col=all_labels,
        target_size=(128, 128),
        batch_size=64,
        class_mode='raw',
        classes=all_labels,
        shuffle=True,
        color_mode = "grayscale",
        subset='training')

# obtaing the validation images using the above generator
valid_gen = core_idg.flow_from_dataframe(
        dataframe=train_df,
        directory=None,
        x_col='path',
        y_col=all_labels,
        target_size=(128, 128),
        batch_size=64,
        class_mode='raw',
        classes=all_labels,
        shuffle=False,
        color_mode = "grayscale",
        subset='validation')

def compute_freq(ground_labels):
    num_samples = ground_labels.shape[0]
    pos_samples = np.sum(ground_labels,axis=0)
    neg_samples = num_samples-pos_samples
    pos_samples = pos_samples/float(num_samples)
    neg_samples = neg_samples/float(num_samples)
    return pos_samples, neg_samples

freq_pos , freq_neg = compute_freq(train_gen.labels)
print(freq_pos)
print(freq_neg)

# assign negative class frequency as positive class weights and vice versa
weights_pos = K.constant(freq_neg,dtype='float32')
weights_neg = K.constant(freq_pos,dtype='float32')
epsilon=1e-7

# weighted loss function to handle class imbalance
def weighted_loss(y_true, y_pred):
    loss = 0.0
    loss_pos = -1 * K.sum( weights_pos * y_true * K.log(y_pred + epsilon), axis=-1)
    loss_neg = -1 * K.sum( weights_neg * (1 - y_true) * K.log(1 - y_pred + epsilon) ,axis=-1)
    return (loss_pos+loss_neg)/len(all_labels)

multi_disease_model = resnet_v2((128,128,1),(9*6)+2,len(all_labels))
multi_disease_model.compile(optimizer = 'adam', loss = weighted_loss)
multi_disease_model.summary()

checkpoint = ModelCheckpoint("resnet_weights_latest{epoch:02d}.h5", verbose=1, save_weights_only = True)
initial_epoch = 0
epochs = 30

# start the training
hist = multi_disease_model.fit(train_gen, validation_data = valid_gen, initial_epoch = initial_epoch, epochs = epochs, callbacks = [checkpoint])

# save the loss on training set for all epochs
np.save('loss'+str(epochs)+'.npy',np.array(hist.history['loss']))
np.save('loss'+str(epochs)+'.npy',np.array(hist.history['val_loss']))