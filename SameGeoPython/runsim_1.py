"""R_TOPS:file name from 0-67
R_PHI:
R_PERM:
R_DEPTH: depth of the top layer 
R_OWCDEPTH: depth of the last layer
R_INJ_X: 1-30
R_INJ_Y:1-110
R_P1_X:
R_P1_Y:
R_INJ_RESV:max reservoir volume injection rate(rtb/d),default=5000
R_INJ_BHP:default=10000
R_P1_WRAT: default=4000"""


import numpy as np
import os

template_file = open('MODEL.DATA','r')
data = template_file.read()
template_file.close()

num_points = 6000

for i in range(num_points):
	px=np.random.randint(1,30+1)
	py=np.random.randint(1,110+1)
	found=False
	ix=0
	iy=0
	while not found:
		ix=np.random.randint(1,30+1)
		iy=np.random.randint(1,110+1)
		if not (ix==px and iy==py):
			found=True
	new_data = data.replace('R_INJ_X', str(ix))
	new_data = new_data.replace('R_INJ_Y', str(iy))

	new_data = new_data.replace('R_P1_X', str(px))
	new_data = new_data.replace('R_P1_Y', str(py))

	mode_id=57 #np.random.randint(0,68)
	new_data = new_data.replace('R_PHI','model_phi_'+ str(mode_id) + '.INC')
	new_data = new_data.replace('R_PERM','model_perm_'+ str(mode_id) + '.INC')
	new_data = new_data.replace('R_TOPS','model_tops_'+ str(mode_id) + '.INC')

	new_data = new_data.replace('R_INJ_RESV', '5000')
	new_data = new_data.replace('R_INJ_BHP', '10000')
	new_data= new_data.replace('R_P1_WRAT', '4000')

	initial_tops=12000
	tops=0
	counter=0
	for m in range(int(60/30)):
		for j in range(int(220/110)):
			for k in range(int(85/5)):
				if mode_id==counter:
					tops=initial_tops+((k*5))*2
					break
				counter+=1	
	new_data = new_data.replace('R_DEPTH', str(tops))
	new_data = new_data.replace('R_OWCDEPTH', str(tops+2*5))

	new_file= open('same_geo/NEW_MODEL_'+ str(i)+'.DATA', 'w')
	new_file.write(new_data)
	new_file.close()













