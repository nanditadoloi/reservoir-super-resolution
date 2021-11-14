import numpy as np
import pandas as pd

f = open("../same_geo/model_phi_57.INC")
poro_list = []
lines = f.readlines()
for i in range(1,len(lines)-1):
    # print(lines[i].rstrip())
    poro_list.append(float(lines[i].rstrip()))
SR_poro = np.reshape(np.array(poro_list),(30,110,5))

f = open("../up_same_geo/model_phi_57.INC")
poro_list = []
lines = f.readlines()
for i in range(1,len(lines)-2):
    # print(lines[i].rstrip())
    poro_list.append(float(lines[i].rstrip()))
LR_poro = np.reshape(np.array(poro_list),(15,55,5))

# df=pd.read_csv("/home/kimsk/delete/SR/SR_NEW_MODEL_0.csv",sep=';')
# SR_OS=np.zeros((30,110,5))
# row_df = df.loc[df['DAYS'] == 7100]
# for j in range(30):
#     for k in range(110):
#         for l in range(5):
#             part_1='BOSAT:'
#             part_2=str(j+1)
#             part_3=','+str(k+1)
#             part_4=','+str(l+1)
#             layer_name=part_1+ part_2+ part_3+part_4
#             data = row_df[layer_name]
#             value = data.to_numpy()[0]
#             SR_OS[j,k,l] = value

# df=pd.read_csv("/home/kimsk/delete/LR/LR_NEW_MODEL_0.csv",sep=';')
# LR_OS=np.zeros((15,55,5))
# row_df = df.loc[df['DAYS'] == 7100]
# for j in range(15):
#     for k in range(55):
#         for l in range(5):
#             part_1='BOSAT:'
#             part_2=str(j+1)
#             part_3=','+str(k+1)
#             part_4=','+str(l+1)
#             layer_name=part_1+ part_2+ part_3+part_4
#             data = row_df[layer_name]
#             value = data.to_numpy()[0]
#             LR_OS[j,k,l] = value

# df=pd.read_csv("/home/kimsk/delete/SR/SR_NEW_MODEL_0.csv",sep=';')
# SR_BPR=np.zeros((30,110,5))
# row_df = df.loc[df['DAYS'] == 7100]
# for j in range(30):
#     for k in range(110):
#         for l in range(5):
#             part_1='BPR:'
#             part_2=str(j+1)
#             part_3=','+str(k+1)
#             part_4=','+str(l+1)
#             layer_name=part_1+ part_2+ part_3+part_4
#             data = row_df[layer_name]
#             value = data.to_numpy()[0]
#             SR_BPR[j,k,l] = value

# df=pd.read_csv("/home/kimsk/delete/LR/LR_NEW_MODEL_0.csv",sep=';')
# LR_BPR=np.zeros((15,55,5))
# row_df = df.loc[df['DAYS'] == 7100]
# for j in range(15):
#     for k in range(55):
#         for l in range(5):
#             part_1='BPR:'
#             part_2=str(j+1)
#             part_3=','+str(k+1)
#             part_4=','+str(l+1)
#             layer_name=part_1+ part_2+ part_3+part_4
#             data = row_df[layer_name]
#             value = data.to_numpy()[0]
#             LR_BPR[j,k,l] = value

np.savez("sample_data.npz", SR_poro=SR_poro, LR_poro=LR_poro)