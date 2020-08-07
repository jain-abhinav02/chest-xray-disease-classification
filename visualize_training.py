import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

val_loss = []
for i in range(30):
    multi_disease_model.load_weights(os.path.join(stored_data_path,'weights_epoch_%02d.h5'%(i+1)))
    cur_loss = multi_disease_model.evaluate(valid_gen, verbose = 1)
    val_loss.append(cur_loss)
np.save('validation_loss.npy',np.array(val_loss))

training_loss = np.load(os.path.join(stored_data_path,'training_loss.npy'))
val_loss = np.load(os.path.join(stored_data_path,'validation_loss.npy'))

# plot loss on training and validation set
fig , axis = plt.subplots(1,1,figsize = (9,6))
axis.plot(range(1,training_loss.shape[0]+1),training_loss,label="training loss")
axis.plot(range(1,val_loss.shape[0]+1),val_loss,label="validation loss")
axis.legend()
plt.xticks(range(1,training_loss.shape[0]+1))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss')
plt.show()
fig.savefig('loss.png')

def print_roc_curve(gen, model,filename):
    pred_labels = model.predict(gen, verbose = 1)
    fig, ax = plt.subplots(1,1, figsize = (9, 9))
    auc_list = []
    for (idx, class_label) in enumerate(all_labels):
        fpr, tpr, thresholds = roc_curve(gen.labels[:,idx].astype(int), pred_labels[:,idx])
        cur_auc = auc(fpr,tpr)
        auc_list.append(cur_auc)
        ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (class_label, cur_auc))
    ax.legend()
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    fig.savefig(filename)
    auc_list = np.array(auc_list)
    return auc_list

auc_epochs = []

print("Epoch 5")
multi_disease_model.load_weights(os.path.join(stored_data_path,"weights_epoch_05.h5"))
auc_5 = print_roc_curve(valid_gen, multi_disease_model,'val_set_roc_curve_epoch_05.png')
auc_epochs.append(auc_5)

print("Epoch 10")
multi_disease_model.load_weights(os.path.join(stored_data_path,"weights_epoch_10.h5"))
auc_10 = print_roc_curve(valid_gen, multi_disease_model,'val_set_roc_curve_epoch_10.png')
auc_epochs.append(auc_10)

print("Epoch 15")
multi_disease_model.load_weights(os.path.join(stored_data_path,"weights_epoch_15.h5"))
auc_15 = print_roc_curve(valid_gen, multi_disease_model,'val_set_roc_curve_epoch_15.png')
auc_epochs.append(auc_15)

print("Epoch 20")
multi_disease_model.load_weights(os.path.join(stored_data_path,"weights_epoch_20.h5"))
auc_20 = print_roc_curve(valid_gen, multi_disease_model,'val_set_roc_curve_epoch_20.png')
auc_epochs.append(auc_20)

print("Epoch 25")
multi_disease_model.load_weights(os.path.join(stored_data_path,"weights_epoch_25.h5"))
auc_25 = print_roc_curve(valid_gen, multi_disease_model,'val_set_roc_curve_epoch_25.png')
auc_epochs.append(auc_25)

print("Epoch 30")
multi_disease_model.load_weights(os.path.join(stored_data_path,"weights_epoch_30.h5"))
auc_30 = print_roc_curve(valid_gen, multi_disease_model,'val_set_roc_curve_epoch_30.png')
auc_epochs.append(auc_30)

auc_epochs =np.array(auc_epochs)
x_labels = [5,10,15,20,25,30]
fig , axis = plt.subplots(1,1,figsize = (9,6))
axis.plot(x_labels,np.mean(auc_epochs,axis = 1),label="mean")
axis.plot(x_labels,np.median(auc_epochs,axis = 1),label="median")
axis.plot(x_labels,np.min(auc_epochs,axis = 1),label="min")
axis.plot(x_labels,np.max(auc_epochs,axis = 1),label="max")
axis.legend()
plt.xticks(x_labels)
plt.show()  

print(np.mean(auc_epochs,axis = 1))
print(np.median(auc_epochs,axis = 1))
print(np.min(auc_epochs,axis = 1))
print(np.max(auc_epochs,axis = 1))