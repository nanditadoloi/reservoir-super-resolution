sid = 15079
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from PIL import Image
import numpy as np

array_lr = np.array(Image.open(f"/media/kimsk/DATA/Nandita_Data/super-resolution/SRGAN-PyTorch/tests/lr_im{sid}.npy.png"))
array_sr = np.array(Image.open(f"/media/kimsk/DATA/Nandita_Data/super-resolution/SRGAN-PyTorch/tests/sr_im{sid}.npy.png"))
array_hr = np.array(Image.open(f"/media/kimsk/DATA/Nandita_Data/super-resolution/SRGAN-PyTorch/tests/hr_im{sid}.npy.png"))

print(np.max(array_lr))

#figure(figsize=(8, 24), dpi=100)
fig, axs = plt.subplots(4,figsize=(16,14), dpi=100)
fig.tight_layout()
axs[0].imshow(array_lr[:,:,0],cmap='jet',vmin=0,vmax=255)
#plt.tight_layout()
#plt.savefig("lr.png")
#plt.show()

#figure(figsize=(8, 6), dpi=100)
axs[1].imshow(np.sum(array_sr[:,:,:],axis=2)/3.0,cmap='jet',vmin=0,vmax=255)
# plt.tight_layout()
# plt.savefig("sr.png")
# plt.show()

#figure(figsize=(8, 6), dpi=100)
axs[2].imshow(array_hr[:,:,0],cmap='jet',vmin=0,vmax=255)
# plt.tight_layout()
# plt.savefig("hr.png")
# plt.show()

#figure(figsize=(8, 6), dpi=100)
axs[3].imshow(np.abs(np.sum(array_sr[:,:,:],axis=2)/3.0-array_hr[:,:,0]),cmap='gray',vmin=0,vmax=255)
plt.tight_layout()
plt.savefig("diff.png")
plt.show()

print(np.max(np.abs(np.sum(array_sr[:,:,:],axis=2)/3.0-array_hr[:,:,0]))/255.0)