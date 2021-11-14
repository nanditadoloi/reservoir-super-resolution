
new_file=open('MPD/BPR' + '.txt','w')
new_file.write('BPR\n')
for i in range(30):
	for j in range(110):
		for k in range(5):
			new_file.write(str(i+1)+' '+str(j+1)+' '+str(k+1)+' /\n')
new_file.write('/\n')			
