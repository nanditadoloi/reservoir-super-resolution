from os import path
import numpy as np

list_data_X = []
list_data_Y = []
list_data_BPR_X = []
list_data_BPR_Y = []
found_indices = []

data = np.load("sample_data.npz")
SR_poro = data['SR_poro']
LR_poro = data['LR_poro']
vol_cell_SR = 10*20*2
vol_cell_LR = 20*40*2 

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

def getC(BPR_2, poro, vol):
    rho_So = 53
    BO_2 = np.zeros(BPR_2.shape)
    for l in range(BPR_2.shape[0]):
        BPR = BPR_2[l,:,:,:]
        for i in range(BPR.shape[0]):
            for j in range(BPR.shape[1]):
                for k in range(BPR.shape[2]):
                    BO_2[l,i,j,k] = giveDensity(BPR[i,j,k])
        BO_2[l,:,:,:] = BO_2[l,:,:,:]*poro*vol*rho_So
    return BO_2


for i in range(6000):
    y_filename='/media/kimsk/DATA/Nandita_Data/super-resolution/Models/SameGeoPython/prepared_data/data_y_'+str(i)+'.npy'
    x_filename='/media/kimsk/DATA/Nandita_Data/super-resolution/Models/SameGeoPython/prepared_data/data_x_'+str(i)+'.npy'
    if path.exists(y_filename) and path.exists(x_filename):
        found_indices.append(i)

print(found_indices)

for i in found_indices:
    filename='/media/kimsk/DATA/Nandita_Data/super-resolution/Models/SameGeoPython/prepared_data/data_y_'+str(i)+'.npy'
    new_BPR_data = np.array(np.load('/media/kimsk/DATA/Nandita_Data/super-resolution/Models/SameGeoPython/prepared_data/data_BPR_y_'+str(i)+'.npy'))
    new_BPR_data = getC(new_BPR_data,SR_poro,vol_cell_SR)
    new_data = np.array(np.load(filename))
    if new_data.shape != (2,30,110,5):
        print(new_data.shape)
    else:
        list_data_Y.append(new_data[:,:,:,:])
        list_data_BPR_Y.append(new_BPR_data[:,:,:,:])
        print(filename)

for i in found_indices:
    # filename_y='prepared_data/data_y_'+str(i)+'.npy'
    # new_BPR_data_y = np.array(np.load('prepared_data/data_BPR_y_'+str(i)+'.npy'))
    # new_BPR_data_y = getC(new_BPR_data_y,SR_poro,vol_cell_SR)
    # new_data_Y = np.array(np.load(filename_y))
    # if new_data_Y.shape != (2,30,110,5):
    #     print(new_data_Y.shape)
    # else:
    #     list_data_Y.append(new_data_Y[0,:,:,:])
    #     list_data_Y.append(new_data_Y[1,:,:,:])
    #     list_data_BPR_Y.append(new_BPR_data_y[0,:,:,:])
    #     list_data_BPR_Y.append(new_BPR_data_y[1,:,:,:])
    #     print(filename_y)

    filename='/media/kimsk/DATA/Nandita_Data/super-resolution/Models/SameGeoPython/prepared_data/data_x_'+str(i)+'.npy'
    new_data = np.array(np.load(filename))
    new_BPR_data = np.array(np.load('/media/kimsk/DATA/Nandita_Data/super-resolution/Models/SameGeoPython/prepared_data/data_BPR_x_'+str(i)+'.npy'))
    new_BPR_data = getC(new_BPR_data,LR_poro,vol_cell_LR)
    if new_data.shape != (2,15,55,5):
        print(new_data.shape, filename)
    else:
        list_data_X.append(new_data[:,:,:,:])
        list_data_BPR_X.append(new_BPR_data[:,:,:,:])
        print(filename)

    # m1 = np.sum(new_data_Y[0,:,:,:]*new_BPR_data_y[0,:,:,:]) 
    # m2 = np.sum(new_data[0,:,:,:]*new_BPR_data[0,:,:,:])
    # print(m1, m2, (m1-m2)*100.0/m2)

data_Y = np.array(list_data_Y)
data_X = np.array(list_data_X)
data_BPR_Y = np.array(list_data_BPR_Y)
data_BPR_X = np.array(list_data_BPR_X)
print(data_Y.shape)


np.save("data_Y_3.npy", data_Y)
np.save("data_X_3.npy", data_X)
np.save("data_Y_BPR_3.npy", data_BPR_Y)
np.save("data_X_BPR_3.npy", data_BPR_X)
