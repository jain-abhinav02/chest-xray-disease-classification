import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

input_data_path=os.path.join('..','dataset')
stored_data_path = os.path.join('..','dataset')
csv_filename='Data_Entry_2017.csv'
all_xray_df = pd.read_csv(os.path.join(stored_data_path,csv_filename))

# print the 15 most common labels in the column 'Finding Labels'
label_counts = all_xray_df['Finding Labels'].value_counts()[:15]
print((label_counts))

fig, ax = plt.subplots(1,1,figsize = (12, 8))
ax.bar(np.arange(len(label_counts))+0.5, label_counts)
ax.set_xticks(np.arange(len(label_counts))+0.5)
_ = ax.set_xticklabels(label_counts.index, rotation = 90)