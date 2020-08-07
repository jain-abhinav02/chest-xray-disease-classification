from sklearn.model_selection import train_test_split
import os
import pandas as pd

input_data_path=os.path.join('..','input','data')
stored_data_path = os.path.join('..','input','resnet-weights')
csv_filename='Data_Entry_2017.csv'
all_xray_df = pd.read_csv(os.path.join(stored_data_path,csv_filename))

# 15% of the data will be used for testing of model performance
# random state is set so as to get the same split everytime
train_df, test_df = train_test_split(all_xray_df,test_size = 0.15, random_state = 2020)
print('Number of training examples:', train_df.shape[0])
print('Number of validation examples:', test_df.shape[0])

# execute just once
train_df.to_csv('train_df.csv',index=False)
test_df.to_csv('test_df.csv',index=False)

# once saved , use the following statements to load train and test dataframes subsequently
train_df = pd.read_csv(os.path.join(stored_data_path,'train_df.csv'))
test_df = pd.read_csv(os.path.join(stored_data_path,'test_df.csv'))