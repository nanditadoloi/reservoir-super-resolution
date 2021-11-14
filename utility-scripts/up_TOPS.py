import numpy as np

initial_tops=12000
file_count=0
for i in range (int(60/30)):
	for j in range(int(220/110)):
		for k in range(int(85/5)):
			new_file=open('upscaled_MPD/model_tops_'+ str(file_count) + '.INC','w')
			print(file_count)
			new_file.write('TOPS\n')
			for l in range(5):
				tops=initial_tops+((k*5)+l)*2
				new_file.write(str(15*55)+'*'+str(tops)+'\n')
			file_count=file_count+1
			new_file.write('/')
			new_file.close()











			