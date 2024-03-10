import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
import math
from math import*
import numpy as np
import matplotlib.pyplot as plt
import mir_eval
from scipy.spatial.distance import euclidean
import csv
from sync_ import event_gen as generate 
import sync_score as sync

beat = np.array([3.48459961134357, 4.2362458037, 4.9734793638, 5.7165179126, 6.4595564613, 7.1967900214, 7.9340235815, 
        8.6712571416, 9.4084907017, 10.1341142844, 10.8625405221564, 11.6109847022822, 12.3251979863693, 
        13.0652342014257, 13.7738421756, 14.5168807244, 15.2713295717129, 15.9913478445, 16.7141687723436, 
        17.4570076423564, 18.1884362138564, 18.9256697738564, 19.6542956903, 20.3973342391, 21.1287628105, 
        21.8601913819, 22.6032299307, 23.3346585021, 24.0660870736, 24.814930611, 25.5463591824, 26.2487628105, 
        26.9857966919129, 27.7116199534])
path = '/home/olfa/attachments/RAPiD/App_Sync_Hamza/sync_anal/frames/result_gen/'
audio = 'country.00056.wav'
shift_percentage = [0.1,0.2,0.3,0.5]
std = [0.1,0.2,0.6,0.8]
mid_10 = int (len(beat)*10/100)
print(mid_10)
num_values_to_select = [mid_10,10,15,int(len(beat)/2)]
print("*************",len(beat))

file_path = ["original"+'.txt',"off_beat"+'.txt',"double"+'.txt', "half_odd"+'.txt' ,"half_even"+'.txt']
vectors = generate._get_reference_beat_variations(beat)
for i in range(0,len(vectors),1): 
    file_path_ = path + file_path[i]
    np.savetxt(file_path_,vectors[i], fmt='%.5f')


for i in range(0,len(shift_percentage),1): 
    var =str(int(shift_percentage[i]*100))
    generate.generate_shifted_vector(beat,shift_percentage[i],'befor_shift_'+var+'_%')
    generate.generate_shifted_vector_after(beat,shift_percentage[i],'after_shift_'+var+'_%')


for i in range(0,len(num_values_to_select),1): 
    var =str(int(num_values_to_select[i]))
    
    generate.shift_array_by_indices(beat,num_values_to_select[i],'not_match_'+var)


for i in range(0,len(std),1): 
    var =str(int(std[i]))
    generate.random_vec (beat,0, std[i],'random_std_'+ str(i))


for i in range(0,len(num_values_to_select),1): 
    var =str(int(num_values_to_select[i]))
    generate.missed_motion(beat, num_values_to_select[i],'missed_'+var)

import os 

# List all files in the folder
file_list = sorted(os.listdir(path))

# Create an empty dictionary to store the arrays with file names as keys
all_arrays = {}

# Loop through each file and read the array
for file_name in file_list:
    file_path = os.path.join(path, file_name)
    
    # Check if the file is a text file (you can adjust the condition based on your file type)
    if file_name.endswith('.txt'):
        # Load the array from the text file using genfromtxt
        loaded_array = np.genfromtxt(file_path)
        
        # Use the file name as the key in the dictionary
        all_arrays[file_name] = loaded_array


for file_name, array in all_arrays.items():
    print(f"Array from file '{file_name}'")
    #score = scores_motion_beat(beat,array,file_name)
    plt.figure(figsize=(18, 4))
    
    generate.plot_with_beats(beat,array,file_name,audio)
    sync.scores_motion_beat(beat,array,file_name)
    plt.show()


############################Scoresss############################


