import numpy as np 
import glob, os


def giveDensity(pressure):
    pvdo_table = np.array([[300,1.05],[800,1.02],[8000,1.01]])
    pl = None
    ph = None
    for i in range(pvdo_table.shape[0]):
        if pvdo_table[i,0]>=pressure:
            pl = i-1
            ph = i
            break
    if pl is None:
        # Data beyond table. Cannot Extrapolate
        return pvdo_table[pvdo_table.shape[0]-1,1]
    elif pl < 0:
        # Data beyond table. Cannot Extrapolate
        return pvdo_table[0,1]
    else:
        # Linearly interpolate bo between two table entries
        dbo_dp = (pvdo_table[ph,1]-pvdo_table[pl,1])/(pvdo_table[ph,0]-pvdo_table[pl,0])
        Dp = pressure-pvdo_table[pl,0]
        return pvdo_table[pl,1]+(dbo_dp*Dp)


data = np.load("Data/sample_data.npz")
SR_poro = data['SR_poro']
LR_poro = data['LR_poro']


# High res
vol_cell = 10*20*2
poro = SR_poro

# Low res
vol_cell = 20*40*2 
poro = LR_poro             

# Common
rho_So = 53

rho_SR*SR_poro*vol_cell_SR*SR_OS

os.chdir("/media/kimsk/DATA/Nandita_Data/super-resolution/Models/data_files_2/test/input_BPR")
for file in glob.glob("*.npy"):
    BPR = np.load(file)
    BO = np.zeros(BPR.shape)
    for i in range(BPR.shape[0]):
        for j in range(BPR.shape[1]):
            for k in range(BPR.shape[2]):
                BO[i,j,k] = giveDensity(BPR[i,j,k])
    C = poro
    
