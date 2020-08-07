import os
import pandas as pd
from glob import glob

input_data_path=os.path.join('..','input','data')
stored_data_path = os.path.join('..','input','resnet-weights')
csv_filename='Data_Entry_2017.csv'
all_xray_df = pd.read_csv(os.path.join(input_data_path,csv_filename))

mask = all_xray_df['Finding Labels']!='No Finding' # set all 'No Finding' labels to 0 and rest to 1
ctr = 0
for i in range(mask.shape[0]):
    if mask[i]==0:
        ctr+=1
    if ctr%5==0:
        mask[i]=1  # select every 5th 'No Finding' label
# No Finding class reduced to 20%

all_xray_df = all_xray_df[mask].copy(deep=True)

all_image_paths = {os.path.basename(f): f  for f in glob(os.path.join(input_data_path,'images*','*','*.png'))   }  
# create a dict mapping image names to their path

print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
print(all_xray_df.sample(3))
all_xray_df.to_csv(os.path.join(stored_data_path,csv_filename),index=False)
