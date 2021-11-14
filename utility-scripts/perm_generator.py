import numpy as np

x=open('SPE10MODEL2_PERM.INC','r')
list_of_lines=x.readlines()
selected_lines=list_of_lines[10:187010]
list_of_numbers=[]
for i in range(len(selected_lines)):
	parts=selected_lines[i].split()
	for j in parts:
		list_of_numbers.append(float(j))

array_of_numbers=np.array(list_of_numbers)
array_of_numbers=np.reshape(array_of_numbers,(60,220,85))
list_of_smallarray_permx=[]
for i in range(int(60/30)):
	for j in range(int(220/110)):
		for k in range(int(85/5)):
			small_array=array_of_numbers[i*30:(i+1)*30,j*110:(j+1)*110,k*5:(k+1)*5]
			list_of_smallarray_permx.append(small_array)



selected_lines=list_of_lines[187013:374013]
list_of_numbers=[]
for i in range(len(selected_lines)):
	parts=selected_lines[i].split()
	for j in parts:
		list_of_numbers.append(float(j))

array_of_numbers=np.array(list_of_numbers)
array_of_numbers=np.reshape(array_of_numbers,(60,220,85))
list_of_smallarray_permy=[]
for i in range (int(60/30)):
	for j in range(int(220/110)):
		for k in range(int(85/5)):
			small_array=array_of_numbers[i*30:(i+1)*30,j*110:(j+1)*110,k*5:(k+1)*5]
			list_of_smallarray_permy.append(small_array)





selected_lines=list_of_lines[374016:561016]
list_of_numbers=[]
for i in range(len(selected_lines)):
	parts=selected_lines[i].split()
	for j in parts:
		list_of_numbers.append(float(j))

array_of_numbers=np.array(list_of_numbers)
array_of_numbers=np.reshape(array_of_numbers,(60,220,85))
list_of_smallarray_permz=[]
for i in range (int(60/30)):
	for j in range(int(220/110)):
		for k in range(int(85/5)):
			small_array=array_of_numbers[i*30:(i+1)*30,j*110:(j+1)*110,k*5:(k+1)*5]
			list_of_smallarray_permz.append(small_array)






for i in range(len(list_of_smallarray_permx)):
	new_file=open('MPD/model_perm_'+ str(i) + '.INC','w')
	small_array=list_of_smallarray_permx[i]
	small_array=np.reshape(small_array,(16500,1))
	new_file.write('PERMX')
	for j in range(16500):
		new_file.write('\n'+ str(small_array[j,0]))

	small_array=list_of_smallarray_permy[i]
	small_array=np.reshape(small_array,(16500,1))
	new_file.write('\n/\nPERMY')
	for j in range(16500):
		new_file.write('\n'+ str(small_array[j,0]))

	small_array=list_of_smallarray_permz[i]
	small_array=np.reshape(small_array,(16500,1))
	new_file.write('\n/\nPERMZ')

	for j in range(16500):
		new_file.write('\n'+ str(small_array[j,0]))



	new_file.write('\n' + '/')
	new_file.close()







			