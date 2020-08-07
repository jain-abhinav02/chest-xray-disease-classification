from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd 
from model import resnet_v2
from training import weighted_loss
from visualize_training import print_roc_curve

# from the above graph, we can infer that the model generalises well at epoch 20
# hence, we use model weights corresponding to epoch 20 for testing
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

multi_disease_model.load_weights("../model_weights/weights_epoch_20.h5")
auc_test = print_roc_curve(test_gen, multi_disease_model,'test_set_roc_curve.png')

np.save("test_set_auc.npy",auc_test)
print(auc_test)

print("Model performance on Test data")
print("min auc =",np.min(auc_test))
print("max auc =",np.max(auc_test))
print("mean auc = ",np.mean(auc_test))
print("median auc = ",np.median(auc_test))