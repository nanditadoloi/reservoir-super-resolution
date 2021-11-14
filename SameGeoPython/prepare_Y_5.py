from ecl.summary import EclSum
import numpy as np
import pandas as pd
import multiprocessing
import os.path
from os import path
import time 

list_layer_names = ['DAYS']

def process_id(i):
    global list_layer_names
    skip_data = False
    a_part_1='../same_geo/NEW_MODEL_'
    a_part_2='0'
    a_part_3='.UNSMRY'
    a_part_4='.csv'
    a_part_2=str(i)
    filename = a_part_1+ a_part_2+ a_part_3
    common_csv_file = a_part_1+ a_part_2+ a_part_4

    if not path.exists(filename):
        return
    
    if path.exists('prepared_data/data_y_'+a_part_2+'.npy'):
        print("Data exists")
        return    
    
    if path.exists(common_csv_file):
        print(common_csv_file, "Reading")
    else:
        print(common_csv_file, "Writing New File")
        summary = EclSum(filename)
        summary.export_csv(common_csv_file)

    df = None
    # t1 = time.time()
    df=pd.read_csv(common_csv_file,sep=';',usecols=["DAYS"])
    id7100 = df[df['DAYS']==7100].index.values
    id5500 = df[df['DAYS']==5500].index.values
    if(len(id5500)==0 or len(id7100)==0):
        print("Rows not found")
        return
    id7100 = id7100[0]
    id5500 = id5500[0]
    rows_to_keep = [0,id5500+1, id7100+1]
    rows_to_skip = [j for j in range(df.shape[0]+1) if j not in rows_to_keep]
    # t3 = time.time()
    try:
        df=pd.read_csv(common_csv_file,sep=';',
                       usecols=list_layer_names,
                       skiprows = rows_to_skip)
    except:
        print("No coumns to read")
        return
    # t2 = time.time()
    # print("optimizes:", t2-t1, t3-t1)

    data_list = []
    BPR_data_list = []
    new_data_BPR=np.zeros((30,110,5))
    new_data=np.zeros((30,110,5))
    row_df = df.loc[df['DAYS'] == 7100]
    for j in range(30):
        for k in range(110):
            for l in range(5):
                part_1='BOSAT:'
                part_2=str(j+1)
                part_3=','+str(k+1)
                part_4=','+str(l+1)
                layer_name=part_1+ part_2+ part_3+part_4
                # list_layer_names.append(layer_name)
                if layer_name not in df.columns:
                    skip_data = True
                if row_df.shape[0] != 1:
                    skip_data = True
                    print("no data")
                if not skip_data:
                    data = row_df[layer_name]
                    value = data.to_numpy()[0]
                    new_data[j,k,l] = value
                    new_data_BPR[j,k,l] = row_df['BPR:'+ part_2+ part_3+part_4].to_numpy()[0]
                else:
                    break
            if skip_data:
                break
        if skip_data:
            break
    data_list.append(new_data)
    new_data=np.zeros((30,110,5))
    BPR_data_list.append(new_data_BPR)
    new_data_BPR=np.zeros((30,110,5))
    row_df = df.loc[df['DAYS'] == 5500]
    for j in range(30):
        for k in range(110):
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
                    new_data[j,k,l] = value
                    new_data_BPR[j,k,l] = row_df['BPR:'+ part_2+ part_3+part_4].to_numpy()[0]
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
        np.save('prepared_data/data_y_'+a_part_2+'.npy',data_list)
        np.save('prepared_data/data_BPR_y_'+a_part_2+'.npy',BPR_data_list)
    else:
        print("Skipping")

for j in range(30):
    for k in range(110):
        for l in range(5):
            part_1='BOSAT:'
            part_2=str(j+1)
            part_3=','+str(k+1)
            part_4=','+str(l+1)
            layer_name=part_1+ part_2+ part_3+part_4
            list_layer_names.append(layer_name)
            list_layer_names.append('BPR:'+ part_2+ part_3+part_4)

ids = list(range(0,6000))
pool = multiprocessing.Pool(20)
dictionary_list = pool.map(process_id, ids)

# # process_id(1000)
# process_id(2498)