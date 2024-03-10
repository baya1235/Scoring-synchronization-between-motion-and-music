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
import librosa.display
class event_gen:
    #@staticmethod
    def generate_shifted_vector(time_vector, percentage,name):
        std_vec = np.diff(time_vector)
    
        shifted_vector = []
        for i in range(0,len(time_vector)-1,1):
            #shift_fraction = percentage / 100.0
            val = time_vector[i] 
            
            shift_amount = std_vec [i] * percentage
            #print('******* shift and value : ',val,shift_amount,std_vec)
            shifted_vector.append(val + shift_amount)
            file_path = name+'.txt'
            path = '/home/olfa/attachments/RAPiD/App_Sync_Hamza/sync_anal/frames/result_gen/'
            all_path = path+file_path
            np.savetxt(all_path,shifted_vector, fmt='%.5f')
        
        return np.sort(shifted_vector)
    #@staticmethod
    def generate_shifted_vector_after(time_vector, percentage,name):
        std_vec = np.diff(time_vector)
        
        shifted_vector = []
        for i in range(0,len(time_vector)-1,1):
            #shift_fraction = percentage / 100.0
            val = time_vector[i] 
            
            shift_amount = std_vec[i] * percentage
            #print('******* shift and value : ',val,shift_amount,std_vec)
            shifted_vector.append(val - shift_amount)
            file_path = name+'.txt'
            path = '/home/olfa/attachments/RAPiD/App_Sync_Hamza/sync_anal/frames/result_gen/'
            all_path = path+file_path
            np.savetxt(all_path,shifted_vector, fmt='%.5f')
        return np.sort(shifted_vector)
    #@staticmethod
    def shift_array_by_indices(input_array,num_values_to_select,name):
        """
        Shifts values in the input array based on the provided indices and shift values.

        Parameters:
        - input_array (numpy array): The input array to be shifted.
        - shift_indices (list): A list of indices to be shifted.
        - shift_values (list): A list of shift values corresponding to each index.

        Returns:
        - numpy array: The shifted array.
        """
        vector_1_to_30 = np.arange(0, len(input_array))

        # Specify the number of values to select
        
        shift_indices = np.random.choice(vector_1_to_30, num_values_to_select)
        #shift_indices = [2, 8, 14, 20, 18]
        shift_values = np.random.uniform(-0.2, 0.2, size=num_values_to_select)
        if len(shift_indices) != len(shift_values):
            raise ValueError("Number of indices and values should be the same.")

        shifted_array = np.array(input_array)

        for index, shift_value in zip(shift_indices, shift_values):
            shifted_array[index] += shift_value
        file_path = name+'.txt'
        path = '/home/olfa/attachments/RAPiD/App_Sync_Hamza/sync_anal/frames/result_gen/'
        all_path = path+file_path
        np.savetxt(all_path,shifted_array, fmt='%.5f')
        return shifted_array
    #@staticmethod
    def random_vec ( input_array,average, std,name) :
        random_array = np.sort(np.array(input_array) + np.random.normal(average,std,len(input_array)))
        #print(random_array)
        file_path = name+'.txt'
        path = '/home/olfa/attachments/RAPiD/App_Sync_Hamza/sync_anal/frames/result_gen/'
        all_path = path+file_path
        np.savetxt(all_path,random_array, fmt='%.5f')
        
        return random_array
    #@staticmethod
    def missed_motion(input_array, num_values_to_select, name):
        vector_1_to_30 = np.arange(0, len(input_array))
        shift_indices = np.random.choice(vector_1_to_30, num_values_to_select, replace=False)

        # Create a boolean mask to exclude specified indices
        mask = np.ones_like(input_array, dtype=bool)
        mask[shift_indices] = False

        # Use the mask to extract values
        modified_array = input_array[mask]
        file_path = name+'.txt'
        path = '/home/olfa/attachments/RAPiD/App_Sync_Hamza/sync_anal/frames/result_gen/'
        all_path = path+file_path
        np.savetxt(all_path,modified_array, fmt='%.5f')
        #print(modified_array)

        return modified_array
    def plot_with_beats(beat ,array, title,audio):
        x, sr = librosa.load(audio)
        values_from_5_to_20 = array[5:20]
        values_from_5_to_20_2 = beat[5:20]
        #plt.subplot(2,1,1)
        plt.vlines(array, ymin=-0.5, ymax=0.5, color='g', label='motion',linewidth=1)
        plt.grid()
        #librosa.display.waveplot(x, sr=sr, alpha=0.2)
        #plt.subplot(2,1,2)
        plt.vlines(beat, ymin=-0.5, ymax=0.5, color='red', label='beat', linestyle='dashed', linewidth=1)
        plt.title(title)
        #plt.legend()
        plt.grid()
    #@staticmethod
    def _get_reference_beat_variations(reference_beats):
        """Return metric variations of the reference beats

        Parameters
        ----------
        reference_beats : np.ndarray
            beat locations in seconds

        Returns
        -------
        reference_beats : np.ndarray
            Original beat locations
        off_beat : np.ndarray
            180 degrees out of phase from the original beat locations
        double : np.ndarray
            Beats at 2x the original tempo
        half_odd : np.ndarray
            Half tempo, odd beats
        half_even : np.ndarray
            Half tempo, even beats

        """

        # Create annotations at twice the metric level
        interpolated_indices = np.arange(0, reference_beats.shape[0]-.5, .5)
        original_indices = np.arange(0, reference_beats.shape[0])
        double_reference_beats = np.interp(interpolated_indices,
                                        original_indices,
                                        reference_beats)
        # Return metric variations:
        # True, off-beat, double tempo, half tempo odd, and half tempo even
        return (reference_beats,
                double_reference_beats[1::2],
                double_reference_beats,
                reference_beats[::2],
                reference_beats[1::2])
    
