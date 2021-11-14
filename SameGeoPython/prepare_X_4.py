from ecl.summary import EclSum
import numpy as np
import pandas as pd
import multiprocessing
import os.path
from os import path


def process_id(i):
    skip_data = False
    a_part_1='../up_same_geo/NEW_MODEL_'
    a_part_2='0'
    a_part_3='.UNSMRY'
    a_part_4='.csv'
    a_part_2=str(i)
    filename = a_part_1+ a_part_2+ a_part_3
    common_csv_file = a_part_1+ a_part_2+ a_part_4

    if not path.exists(filename):
        return

    if path.exists('prepared_data/data_x_'+a_part_2+'.npy'):
        print("Data exists")
        return

    if path.exists(common_csv_file):
        print(common_csv_file, "Reading")
    else:
        print(common_csv_file, "Writing New File")
        summary = EclSum(filename)
        summary.export_csv(common_csv_file)

    df=pd.read_csv(common_csv_file,sep=';')
    data_list = []
    BPR_data_list = []
    new_data=np.zeros((15,55,5))
    new_data_BPR=np.zeros((15,55,5))
    row_df = df.loc[df['DAYS'] == 7100]
    for j in range(15):
        for k in range(55):
            for l in range(5):
                part_1='BOSAT:'
                part_2=str(j+1)
                part_3=','+str(k+1)
                part_4=','+str(l+1)
                layer_name=part_1+ part_2+ part_3+part_4
                if layer_name not in df.columns:
                    skip_data = True
                if row_df.shape[0] != 1:
                    skip_data = True
                    print("no data")
                if not skip_data:
                    data = row_df[layer_name]
                    new_data_BPR[j,k,l] = row_df['BPR:'+ part_2+ part_3+part_4].to_numpy()[0]
                    value = data.to_numpy()[0]
                    new_data[j,k,l] = value
                else:
                    break
            if skip_data:
                break
        if skip_data:
            break
    data_list.append(new_data)
    BPR_data_list.append(new_data_BPR)
    new_data=np.zeros((15,55,5))
    new_data_BPR=np.zeros((15,55,5))
    row_df = df.loc[df['DAYS'] == 5500]
    for j in range(15):
        for k in range(55):
            for l in range(5):
                part_1='BOSAT:'
                part_2=str(j+1)
                part_3=','+str(k+1)
                part_4=','+str(l+1)
                layer_name=part_1+ part_2+ part_3+part_4
                if layer_name not in df.columns:
                    skip_data = True
                if row_df.shape[0] != 1:
                    skip_data = True
                    print("no data")
                if not skip_data:
                    data = row_df[layer_name]
                    value = data.to_numpy()[0]
                    new_data_BPR[j,k,l] = row_df['BPR:'+ part_2+ part_3+part_4].to_numpy()[0]
                    new_data[j,k,l] = value
                else:
                    break
            if skip_data:
                break
        if skip_data:
            break
    data_list.append(new_data)
    BPR_data_list.append(new_data_BPR)
    if not skip_data:
        #pass
        np.save('prepared_data/data_x_'+a_part_2+'.npy',data_list)
        np.save('prepared_data/data_BPR_x_'+a_part_2+'.npy',BPR_data_list)
    else:
        print("Skipping")


ids = list(range(0,6000))
pool = multiprocessing.Pool(20)
dictionary_list = pool.map(process_id, ids)

# process_id(3000)