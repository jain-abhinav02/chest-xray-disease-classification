import os
import pandas as pd

input_data_path=os.path.join('..','input','data')
stored_data_path = os.path.join('..','input','resnet-weights')
csv_filename='Data_Entry_2017.csv'
all_xray_df = pd.read_csv(os.path.join(stored_data_path,csv_filename))

all_labels=set()
# create a set of the names of all diseases

def sep_diseases(x):
    list_diseases=x.split('|')
    for item in list_diseases:
        all_labels.add(item)
    return list_diseases

# Since the image may contain multiple disease labels
# Create a list of all disesases and append a new column named output to the x_ray dataframe
all_xray_df['disease_vec']=all_xray_df['Finding Labels'].apply(sep_diseases)

all_labels=list(all_labels)
all_labels.remove('No Finding')
all_labels.sort()

disease_freq={}
for sample in all_xray_df['disease_vec']:
    for disease in sample:
        if disease in disease_freq:
            disease_freq[disease]+=1
        else:
            disease_freq[disease]=1
print(disease_freq)

fig,ax=plt.subplots(1,1,figsize=(12,8))
plt.xticks(range(15),list(disease_freq.keys()), rotation = 90)
freq = np.array(list(disease_freq.values()))
percent = freq/np.sum(freq)
ax.bar(range(15),list(percent))

for label in all_labels:
    all_xray_df[label]=all_xray_df['disease_vec'].apply(lambda x: float(label in x))

# Glimpse of the pre-processed dataframe
all_xray_df.loc[:,'disease_vec':]

all_xray_df.to_csv(os.path.join(stored_data_path,csv_filename),index=False)