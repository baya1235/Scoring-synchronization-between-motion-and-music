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

class event_gen:
    #@staticmethod
    def generate_shifted_vector(reference_beats, percentage,name):
        """
        Generates a shifted beats vector from a reference beats vector.

        Args:
            reference_beats (array_like): A reference beats array.
            percentage (float): The percentage of shift to apply to each beat.
            name (str): The name of the file to save the shifted beats vector.

        Returns:
            array_like: A sorted shifted beats array.

        Note:
            The percentage shift should be between 0 and 100 inclusive.
        
        """
        inter_beat = np.diff(reference_beats)
    
        shifted_vector = []
        for i in range(0,len(reference_beats)-1,1):
            val = reference_beats[i] 
            shift_amount = inter_beat [i] * percentage
            shifted_vector.append(val + shift_amount)
            file_path = name+'.txt'
            path = './'
            all_path = path+file_path
            np.savetxt(all_path,shifted_vector, fmt='%.5f')
        
        return np.sort(shifted_vector)
    #@staticmethod
    def generate_shifted_vector_after(reference_beats, percentage,name):
        """
        Generates a shifted beats vector by shifting each beat backwards after the reference beats.

        Args:
            reference_beats (array_like): A reference beats array.
            percentage (float): The percentage of shift to apply to each inter-beat interval.
            name (str): The name of the file to save the shifted beats vector.

        Returns:
            array_like: A sorted shifted beats array.
        Note:
            The percentage shift should be between 0 and 100 inclusive.
        """
        inter_beat = np.diff(reference_beats)
        
        shifted_vector = []
        for i in range(0,len(reference_beats)-1,1):
            val = reference_beats[i] 
            shift_amount = inter_beat[i] * percentage
            shifted_vector.append(val - shift_amount)
            file_path = name+'.txt'
            path = './'
            all_path = path+file_path
            np.savetxt(all_path,shifted_vector, fmt='%.5f')
        return np.sort(shifted_vector)
    #@staticmethod
    def matching(reference_beats,num_values_to_select,name):
        """
        Shifts values in the input array based on the provided indices and shift values.

        Parameters:
        - reference_beats (numpy array): The input array to be shifted.
        - shift_indices (list): A list of indices to be shifted.
        - shift_values (list): A list of shift values corresponding to each index.

        Returns:
        - numpy array: The shifted array.
        """
        vector_1_to_30 = np.arange(0, len(reference_beats))

        # Specify the number of values to select
        
        shift_indices = np.random.choice(vector_1_to_30, num_values_to_select)
        shift_values = np.random.uniform(-0.2, 0.2, size=num_values_to_select)
        if len(shift_indices) != len(shift_values):
            raise ValueError("Number of indices and values should be the same.")

        shifted_array = np.array(reference_beats)

        for index, shift_value in zip(shift_indices, shift_values):
            shifted_array[index] += shift_value
        file_path = name+'.txt'
        path = './'
        all_path = path+file_path
        np.savetxt(all_path,shifted_array, fmt='%.5f')
        return shifted_array
    #@staticmethod
    def randomized ( reference_beats,average, std,name) :
        """
        Generates a random shifted array based on the reference beats array.

        Parameters:
            reference_beats (array_like): The reference beats array.
            average (float): The average of the normal distribution for random values.
            std (float): The standard deviation of the normal distribution for random values.
            name (str): The name of the file to save the random shifted array.

        Returns:
            numpy array: The random shifted array.
        """
        
        random_array = np.sort(np.array(reference_beats) + np.random.normal(average,std,len(reference_beats)))
        file_path = name+'.txt'
        path = './'
        all_path = path+file_path
        np.savetxt(all_path,random_array, fmt='%.5f')
        
        return random_array
    #@staticmethod
    def missed_motion(reference_beats, num_values_to_select, name):
        """
        Generates an array with missed motion based on the provided reference beats array.

        Parameters:
            reference_beats (array_like): The reference beats array.
            num_values_to_select (int): The number of values to select for missed motion.
            name (str): The name of the file to save the observation array.

        Returns:
            numpy array: The array with missed motion.
        """
        
        vector_1_to_30 = np.arange(0, len(reference_beats))
        shift_indices = np.random.choice(vector_1_to_30, num_values_to_select, replace=False)

        # Create a boolean mask to exclude specified indices
        mask = np.ones_like(reference_beats, dtype=bool)
        mask[shift_indices] = False

        # Use the mask to extract values
        observation = reference_beats[mask]
        file_path = name+'.txt'
        path = './'
        all_path = path+file_path
        np.savetxt(all_path,observation, fmt='%.5f')
        #print(observation)

        return observation
    #@staticmethod
    def plot_with_beats(reference_beats ,observation, title):
        """
        Plots reference beats and motion observations.

        Parameters:
            reference_beats (array_like): Array containing reference beat positions.
            observation (array_like): Array containing observed motion positions.
            title (str): Title for the plot.
        """
        plt.vlines(observation, ymin=-0.5, ymax=0.5, color='g', label='motion',linewidth=1)
        plt.grid()
        plt.vlines(reference_beats, ymin=-0.5, ymax=0.5, color='red', label='beat', linestyle='dashed', linewidth=1)
        plt.title(title)
        plt.legend()    
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
    
