import pandas as pd
import numpy as np
import os

# Change the directory
os.chdir("C://UCD/__Practicum/ImageTagging/tianchi_data/")

# Get labels
labels1 = pd.read_csv("round1_train/Annotations/label.csv", names=["path", "folder", "code"])
labels2 = pd.read_csv("round2_train/Annotations/label.csv", names=["path", "folder", "code"])
labels = pd.concat([labels1, labels2], ignore_index=True)
labels.head()

''' # Extract the directory and the image name
for i in range(len(labels)):
    split = os.path.split(labels.loc[i,'path'])
    labels.loc[i,'directory'] = split[0]
    labels.loc[i,'filename'] = split[1]
labels.head()
'''
''' # Check if no-y exists
for i in range(len(labels)):
    if "y" not in labels.loc[i,'code']:
        print("yes", i)
        break
#  NO SUCH SITUATION - NO OUTPUT IN THE TERMINAL
'''

# Subset annotations
coat_data = labels[labels['folder'] == 'coat_length_labels'].reset_index(drop=True)
collar_data = labels[labels['folder'] == 'collar_design_labels'].reset_index(drop=True)
lapel_data = labels[labels['folder'] == 'lapel_design_labels'].reset_index(drop=True)
neck_data = labels[labels['folder'] == 'neck_design_labels'].reset_index(drop=True)
neckline_data = labels[labels['folder'] == 'neckline_design_labels'].reset_index(drop=True)
pant_data = labels[labels['folder'] == 'pant_length_labels'].reset_index(drop=True)
skirt_data = labels[labels['folder'] == 'skirt_length_labels'].reset_index(drop=True)
sleeve_data = labels[labels['folder'] == 'sleeve_length_labels'].reset_index(drop=True)


# Create lists for each hierarchy - lapel design, neck design
coat = ['Invisible','High Waist Length','Regular Length','Long Length','Micro Length','Knee Length','Midi Length','Anckle&Floor Length']
collar = ['Invisible', 'Shirt Collar', 'Peter Pan', 'Puritan Collar', 'Rib Collar']
lapel = ["Invisible", "Notched", "Collarless", "Shawl Collar", "Plus Size Shawl"]
neck = ["Invisible", "Turtle Neck", "Ruffle Semi-High Collar", "Low Turtle Neck", "Draped Collar"]
neckline = ['Invisible','Strapless Neck','Deep V Neckline','Straight Neck','V Neckline','Square Neckline','Off Shoulder','Round Neckline','Swear Heart Neck','One Shoulder Neckline']
pant = ['Invisible', 'Short Pant', 'Mid Length', '3/4 Length', 'Cropped Pant', 'Full Length']
skirt = ['Invisible', 'Short Length', 'Knee Length', 'Midi Length', 'Ankle Length', 'Floor Length']
sleeve = ['Invisible','Sleeveless','Cup Sleeves','Short Sleeves','Elbow Sleeves','3/4 Sleeves','Wrist Length','Long Sleeves','Extra Long Sleeves']

categories = ['coat','collar','lapel','neck','neckline','pant','skirt','sleeve']
datasets = []
for i in categories:
    datasets.append(f'{i}_data')

## Decode label 
def decode(data, list):
    for i in range(len(data)):
        pos = data.loc[i,'code'].find('y')
        data.loc[i,'label'] = list[pos]

# Decode labels for each category
for i,j in zip(datasets,categories):
    decode(globals()[i],globals()[j])

lapel_data.head()

# Create folders
os.getcwd()
os.chdir("train/Images")
os.listdir()
for i in neck:
    os.mkdir(f"neck_design_labels/{i}")

os.chdir("..")
os.chdir("neck_design_labels")
# Group images to corresponding folders
def group(data):
    for i in range(len(data)):
        split = os.path.split(data.loc[i,'path'])
        oldpath = split[1]
        newpath = data.loc[i,'label']+'/'+oldpath
        os.rename(oldpath, newpath)

group(neck_data)

