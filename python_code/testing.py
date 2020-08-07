# from the above graph, we can infer that the model generalises well at epoch 20
# hence, we use model weights corresponding to epoch 20 for testing

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

multi_disease_model.load_weights(os.path.join(stored_data_path,"weights_epoch_20.h5"))
auc_test = print_roc_curve(test_gen, multi_disease_model,'test_set_roc_curve.png')

np.save("test_set_auc.npy",auc_test)
print(auc_test)

print("Model performance on Test data")
print("min auc =",np.min(auc_test))
print("max auc =",np.max(auc_test))
print("mean auc = ",np.mean(auc_test))
print("median auc = ",np.median(auc_test))