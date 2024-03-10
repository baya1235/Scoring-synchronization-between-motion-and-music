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
#@staticmethod
 
def calculate_score_gauss(x,alpha):
    #x = p-g
    expo = np.exp(-x**2 / (2 * alpha**2))
    #score_i = 2 * (1 - sigmoid_term)
    return expo 
#@staticmethod
def calculate_score(p, g,alpha):
    
    sigmoid_term = 1 / (1 + np.exp(-(alpha*(abs(p - g)))))
    score_i = 2 * (1 - sigmoid_term)
    return score_i


#@staticmethod
def normalizeAngle( angle) : 

        angle %= 360
        if (angle > 180.0) : 
            return angle - 360
        if (angle <= -180):
            return angle + 360
        return angle
#@staticmethod
def phase_relative (b1,b2) : 
    angle_ = []
    r_1 = []
    count = 0
    cos = []
    for f in  b2 :
            #print('first loop',f)
            for i in range (0,len(b1)-1,1) :
                #print('beat second',
                if f  == b1[i+1] : 
                    phi_app = 0 
                    angle_.append(phi_app)
                    r_1.append(i)
                if (f< b1[i+1] and f > b1[i] ) : 
                    phi_app=( 360 * (f-b1[i]) / (b1[i+1]- b1[i]))
                    cos.append(np.cos(phi_app))
                    angles_rad = np.deg2rad(phi_app)
                    phi_app = normalizeAngle(phi_app)
                    angle_.append(phi_app)
                    count =count+1
                    r_1.append(i)
                

    # Calculate median direction
    median_direction = np.median(cos)

    # Calculate angular variation
    angular_variation = np.max(cos) - np.min(cos)

    # Calculate angular deviation
    angular_deviation = np.std(cos)
    # Calculate the mean direction
    
    mean_direction = np.arctan2(np.mean(np.sin(angle_ )), np.mean(np.cos(angle_ )))

    # Calculate the mean vector length
    mean_vector_length = np.sqrt(np.mean(np.sin(angle_ )) ** 2 + np.mean(np.cos(angle_ )) ** 2)
    # Create a circular plot
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='polar')
    ax.plot(angle_ , np.ones(len(angle_ )), 'ro')  # Plot data points
    ax.quiver(0, 0, np.cos(mean_direction), np.sin(mean_direction), angles='xy', scale_units='xy', scale=1, color='blue', width=0.02)
    ax.set_yticklabels([])  # Hide radial tick labels
    plt.title(f"Mean Direction: {np.degrees(mean_direction):.5f}°, Mean Vector Length: {mean_vector_length:.5f}")
    """
    
    return angle_, r_1, median_direction, angular_variation, angular_deviation,mean_direction,mean_vector_length
#@staticmethod
def normalize_to_length(value, length):
    min_value = 0
    max_value = length
    normalized_value = (value - min_value) / (max_value - min_value)
    return normalized_value
#@staticmethod

def normalize_values(original_values, new_min=0, new_max=30):
    normalized_values = (original_values * (new_max - new_min)) + new_min
    return normalized_values
#@staticmethod
def viz(scores, phi):
    # Create the plots
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    # Histogram
    ax[0].hist(phi, bins=20, color='skyblue', edgecolor='black')
    ax[0].set_title('Histogram of Phases')
    ax[0].set_xlabel('Phase')
    ax[0].set_ylabel('Frequency')

    # Polar plot
    ax[1] = plt.subplot(1, 2, 2, polar=True)
    c = ax[1].scatter(phi, scores, c=scores, cmap='viridis', marker='o')
    ax[1].set_title('Polar Plot of Scores vs Phases')
    plt.colorbar(c, ax=ax[1], orientation='vertical', label='Score')

    plt.tight_layout()
    plt.show()
#@staticmethod
def viz_1 (scores,name) : 
    # Your vector of scores between 0 and 1
    # Replace this with your actual scores
    #scores = np.random.rand(100)
    
    # Number of bins for the bar chart
    num_bins = len(scores)
    
    # Create the plots
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    
    # Bar chart
    ax[0].bar(range(num_bins), scores)
    ax[0].set_title(name)
    ax[0].set_xlabel('Index')
    ax[0].set_ylabel('Score')
    ax[0].set_ylim([0, 1])
    
    # Convert the scores to a 2D array for the heatmap
    # Here we're just replicating the scores for demonstration purposes
    # In a real dataset, you would map your scores to the theta and radius dimensions
    scores_matrix = np.tile(scores, (num_bins, 1))
    
    # Polar heatmap
    theta = np.linspace(0, 2*np.pi, num_bins)
    r = np.linspace(0, 1, num_bins)
    Theta, R = np.meshgrid(theta, r)
    
    ax[1] = plt.subplot(1, 2, 2, polar=True)
    c = ax[1].contourf(Theta, R, scores_matrix, 50, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(c, ax=ax[1], orientation='vertical')
    
    plt.tight_layout()
    plt.show()
#@staticmethod


def dtw_distance(s1, s2):
    """
    Compute the Dynamic Time Warping (DTW) distance between two sequences.
    """
    n, m = len(s1), len(s2)
    dtw_matrix = np.zeros((n + 1, m + 1))

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])

    return dtw_matrix[n, m]

def normalize_dtw(v1, v2):
    """
    Compute the normalized DTW distance between two vectors v1 and v2.
    """
    dtw_dist = dtw_distance(v1, v2)
    relative_difference = max(abs(max(v1) - min(v1)), abs(max(v2) - min(v2)))
    normalized_distance = dtw_dist / relative_difference
    return normalized_distance


def scores_motion_beat(beat,motion,name) : 
    Phi , r_1, median_direction, angular_variation, angular_deviation,mean_direction,mean_vector_length=phase_relative (beat,motion)
    #print('Phase : ',median_direction, angular_variation, angular_deviation,mean_direction,mean_vector_length)
    score = []
    cosine_values = np.cos((Phi))
    scores = np.where(cosine_values < 0, 1 + cosine_values, cosine_values)

    # Calculate the mean of the scores
    mean_score = np.mean(scores)
    #print(str(motion))
    #evaluation_sig(beat,motion,name)
    #a,b = cemgil(beat,
    #       motion,
    #       cemgil_sigma=0.04)
    gauss_score = gausian_score (beat,motion,name)
    #print('lenght motion beat score : ', len(motion),len(beat),gauss_score)
    sg_score = sigmoid_score (beat,motion,10) 
    #print ('sigmoid score m2',sg_score)
    #gauss_score = 0
    
    scores = mir_eval.beat.evaluate(beat,motion)
    distance_dtw, path_dtw = fastdtw(beat,np.array(motion), dist=euclidean)
    score_dtw =  1 / (1 + distance_dtw)
    score_gauso,max_acc = cemgil(beat , motion,cemgil_sigma=0.04)
    #print("*************",score_gauso)
    #score_dtw = (1/distance_dtw)
    #score_dtw = 1 / (1+distance_dtw)
    #print('DTW : ',score_dtw)
    for i in range(0,len(path_dtw),1):
        x,y = path_dtw[i]
        x_v = beat[x]
        y_v = motion[y]
        #print('*********',x_v - y_v)
        score_i = calculate_score(x_v, y_v,20)
        score.append(score_i)
    """
    for beat in beat:
            # Find the error for the closest beat to the reference beat
            beat_diff = np.min(np.abs(beat - motion))
            #print('##############',beat_diff)
    """
    #print(len(Phi),len(score))
    scoree = score[:len(score)-1]
    #viz (scoree,Phi)
    #viz_1(score,name)
    socre_sigm = np.mean(score)
    #print('socre_sigm M1',socre_sigm)
    score_mir = []
    for metric, value in scores.items():
        #print(f'{metric}: {value}')
        score_mir.append(value)

    
    # Calculer la distance DTW normalisée
    distance_normalisée = normalize_dtw(beat,motion)

    print("Distance DTW normalisée entre les vecteurs :", 1-distance_normalisée)
    #print(Phi)
    #print(score)
    #mean_direction = np.arctan2(np.mean(np.sin(Phi)), np.mean(np.cos(Phi )))

    # Calculate the mean vector length
    #mean_vector_length = np.sqrt(np.mean(np.sin(Phi )) ** 2 + np.mean(np.cos(Phi )) ** 2)
    # Create a circular plot
    
    custom = np.mean(np.abs(Phi))
    from scipy.stats import circmean
    mean_direction = np.angle(np.mean(np.exp(1j * np.radians(Phi))))
    mean_vector_length = np.abs(np.mean(np.exp(1j * np.radians(Phi))))
    mean_angular_direction = circmean(np.radians(Phi))
    normalized_to_beat_length = normalize_to_length(mean_vector_length,int(len(Phi)))
    normalized_values = normalize_values(mean_vector_length,new_min=0, new_max=len(beat))
    #r_1 = np.linspace(0, 1,len(Phi))
    import plotly.graph_objects as go
    #fig = go.Figure(data=go.Scatterpolar(r=r_1,theta=Phi,mode='markers',))
    
    
    #fig.show()
    # Create a polar scatter plot
    
    fig = go.Figure()

    # Add the original data
    fig.add_trace(go.Scatterpolar(r=r_1, theta=Phi, mode='markers', name='Phases',marker=dict(color='black', line=dict(color='red', width=4))))
    
    # Add the mean direction as a line
    
    md = np.degrees(mean_direction)
    fig.add_trace(go.Scatterpolar(
        r=[0,normalized_values],
        theta=[md, md],
        mode='lines',
        line=dict(color='black', width=4),
        name='Mean Direction'
    ))

    title_text = (
    f"{name}<br>"
    f"Mean Direction: {np.degrees(mean_direction):.2f}°<br>"
    f"Mean Vector Length: {mean_vector_length:.2f}<br>"
    f"DTW score: {score_dtw:.2f}<br>"
    f"Gauss score: {gauss_score:.2f}<br>"
    f"Score Sigmoid c = 10: {socre_sigm:.2f}<br>"
    f"F_Score +-70ms: {score_mir[0]:.2f}<br>"
    
    
)

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, max(r_1)] 
            ,gridcolor='green'
                # Définissez la plage de l'axe radial
        )
    ),
    title=dict(
        text=title_text,
        x=0,  # Alignez le titre à gauche
        y=0.8,  # Alignez le titre en haut
        #font=dict(size=20, color='black', family="Courier New", weight="bold")  # Adjust size, color, and weight

        font=dict(size=20)
        #margin=dict(t=30)  # Ajustez la taille de la police si nécessaire
    ),
    #paper_bgcolor='lightgrey'
)      
    
    #fig.show()
    #print((Phi))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    

    # Ajout des données originales
    ax.scatter(np.radians(Phi), r_1, label='Phases', color='black')

    # Ajout du disque polar avec colormap
    #cax = ax.scatter(np.radians(Phi), r_1, c=Phi, cmap='viridis', alpha=0.75)

    # Ajout de la direction moyenne en tant que ligne
    ax.plot([np.radians(md), np.radians(md)], [0, normalized_values], color='g', linewidth=2, label='Mean Direction')

    # Ajout de la barre de couleur
    #fig.colorbar(cax, ax=ax, label='Phase Angle (degrees)')
    # Réglages du graphique
    ax.set_rmax(max(r_1))
    # Ajout du titre global
    
    fig.suptitle(
    
    f"Mean Direction: {md:.2f}°,   "
    f"|R|: {mean_vector_length:.2f},   "
    f"DTW: {score_dtw:.2f}q\n"
    f"Gaussian: {gauss_score:.2f},   "
    f"Sigmoid: {sg_score[1]:.2f}\n"
    f"F_Score: {score_mir[0]:.2f},   "
    f"CMLc: {score_mir[5]:.2f},   "
    f"CMLt: {score_mir[6]:.2f}\n"
    f"AMLc : {score_mir[7]:.2f},   "
    f"AMLt: {score_mir[8]:.2f}\n",

    size=15,  # Ajustez la taille de la police ici
    x=0.5,  # Alignez le titre à gauche
    y=1  # Alignez le titre en haut
)

    
    continuity(beat,
            motion,
            continuity_phase_threshold=0.5,
            continuity_period_threshold=0.5)
    
    print("Mean Direction: ",md   )
    print("|R|: ",mean_vector_length)
    
    print("DTW: ",score_dtw,distance_dtw)

    print("Gaussian:", gauss_score)
    print("Sigmoid : ",sg_score)
    print("F_Score: ",score_mir[0])
    print("CMLc: ",score_mir[5])
    print("CMLt: ",score_mir[6])
    print("AMLc : ",score_mir[7])
    print("AMLt: ",score_mir[8])
    
    plt.show()
    
    with open('scores_sc.csv', 'a', newline='') as csvfile:
                    csv.writer(csvfile).writerow([ name,distance_dtw,score_dtw,socre_sigm,score_mir[0],score_mir[1],score_mir[2],score_mir[3],score_mir[4],score_mir[5],score_mir[6] ,mean_vector_length ,np.degrees(mean_direction),mean_score,custom,gauss_score,socre_sigm])
    return scores
#@staticmethod
def phase_difference(b1, b2):
    angle_diff = []
    count = 0

    for f1, f2 in zip(b1, b2):
        for i in range(0, len(b1) - 1, 1):
            if f2 < b1[i + 1] and f2 > b1[i]:
                phi_app_1 = 360 * (f1 - b1[i]) / (b1[i + 1] - b1[i])
                phi_app_2 = 360 * (f2 - b1[i]) / (b1[i + 1] - b1[i])

                # Calculate the phase difference
                phi_diff = phi_app_2 - phi_app_1
                angle_diff.append(phi_diff)
                count += 1
    
    # Calculate median direction
    median_direction = np.median(angle_diff)

    # Calculate angular variation
    angular_variation = np.max(angle_diff) - np.min(angle_diff)

    # Calculate angular deviation
    angular_deviation = np.std(angle_diff)
    # Calculate the mean direction
    mean_direction = np.arctan2(np.mean(np.sin(angle_diff )), np.mean(np.cos(angle_diff)))

    # Calculate the mean vector length
    mean_vector_length = np.sqrt(np.mean(np.sin(angle_diff)) ** 2 + np.mean(np.cos(angle_diff)) ** 2)
    # Create a circular plot
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='polar')
    ax.plot(angle_diff, np.ones(len(angle_diff)), 'ro')  # Plot data points
    ax.quiver(0, 0, np.cos(mean_direction), np.sin(mean_direction), angles='xy', scale_units='xy', scale=1, color='blue', width=0.02)
    ax.set_yticklabels([])  # Hide radial tick labels
    plt.title(f"Mean Direction: {np.degrees(mean_direction):.5f}°, Mean Vector Length: {mean_vector_length:.5f}")
    return angle_diff, count,median_direction, angular_variation, angular_deviation,mean_direction,mean_vector_length
#@staticmethod
def gaussian_function(x, mean, std):
    return np.exp(-(x - mean) ** 2 / (2 * std ** 2)) / (std * np.sqrt(2 * np.pi))
#@staticmethod
def gaussian_function(x, mean, std):
    return np.exp(-(x - mean) ** 2 / (2 * std ** 2)) / (std * np.sqrt(2 * np.pi))
#@staticmethod
def intersection_function(x, beat_time, mean, std):
    return gaussian_function(x, mean, std) - beat_time
#@staticmethod


def sigmoid_score (beat,motion,alpha) : 
    beat_diff = []
    for beato in beat:
        beat_diff .append( (np.min(np.abs(beato- motion))))
    #calculate_score_gauss(beat_diff,diff)
    #print(beat_diff)
    #print('############"""""',len(beat_diff),len(beat))
    expo = []
    accuracy = []
    for i in range (0,len(beat_diff),1): 
        #expo.append(calculate_score_gauss(beat_diff[i],diffo[i] / 4)) 
        sigmoid_term = (1 / (1 + np.exp(-(alpha*(abs(beat_diff[i]))))))
        accuracy.append( 2 * (1 - sigmoid_term))
    #print('lenght score sigm2 : ',len(accuracy),accuracy[0],accuracy[int(len(accuracy)-1)])
    leng = max(len(beat),len(motion))   
    return np.mean(accuracy),np.sum(accuracy) / leng,accuracy
def evaluation_sig(beat,motion,name) : 
    distance_dtw, path_dtw = fastdtw(beat,np.array(motion), dist=euclidean)
    alphas = np.arange(0,50,5)
    score_1 = []
    score = [] 
    alpha_vec = []
    for i in range(0,len(alphas),1):
        alpha = alphas[i]
        for i in range(0,len(path_dtw),1):
            x,y = path_dtw[i]
            x_v = beat[x]
            y_v = motion[y]
            #print(alpha)
            score_i = calculate_score(x_v, y_v,alpha)
            score.append(score_i)
        #print(alpha)
        socre_sigm = np.mean(score)
        score = []
        score_1.append(socre_sigm)
        alpha_vec.append(alpha)
    #print(score_1,alpha_vec)
    plt.plot(alpha_vec, score_1, marker='o')
    plt.title(name)
    plt.xlabel('Alpha')
    plt.ylabel('Score')
    plt.grid(True)
    plt.show()
#@staticmethod
def gausso (beat,motion) :

    Wg = np.exp(-(((beat-motion)**2)/(2.0*( (beat - motion ) **2))))
    return Wg
#@staticmethod
def gausian_score (beat,motion,name) : 
     
    x_values = np.linspace(np.min(beat) - 1, np.max(beat) + 1, 1000)
    import math
    diffo = []
    from scipy.optimize import fsolve

    for i in range (0,len(beat)-1,1) : 
        diffo.append(beat[i+1]- beat[i])
    #print(len(diffo),len(beat))
    last_ = diffo[-1]
    diffo =  np.append(diffo, last_)
    diffo = diffo **2
    #w_values = calculate_score_gauss()
    beat_diff = []
    for beato in beat:
            # Find the error for the closest beat to the reference beat
        beat_diff.append(np.min(np.abs(beato - motion)))
        #calculate_score_gauss(beat_diff,diff)
        #print(beat_diff)
    #print('############"""""',len(beat_diff),len(diffo))
    expo = []
    accuracy = 0
    for i in range (0,len(beat_diff),1): 
        expo.append(calculate_score_gauss(beat_diff[i],diffo[i] / 4)) 
        accuracy += np.exp(-(beat_diff[i]**2)/(2.0*(diffo[i] / 4)**2))
    # Normalize the accuracy
    #print('score',expo)
    #print('lenght score  : ',len(expo),expo[0],expo[int(len(expo)-1)])
    leng = max(len(beat),len(motion))
    score =  np.sum(accuracy) / leng
    #accuracy /= .5*(beat.shape[0] + motion.shape[0])
    #print('############ cmgil accuracy ',accuracy)
    # Add it to our list of accuracy scores
    #accuracies.append(accuracy)
    #print( accuracie, np.max(accuracies))
        #print(diffo[i] / 4)
    #print('Expooooo len beat',np.sum(expo)/len(beat))
    #print('Expooooo mean',np.mean(expo))
    
    gaussian_score = np.sum(np.max(expo)) / (len(beat) + len(motion)) / 2.0
    # Calculate Gaussian score
    #gaussian_score = np.sum(np.max(w_values)) / (len(beat) + len(motion)) / 2.0
    
    #return gaussian_score
    plt.figure(figsize=(8, 4))
    for i in range(len(beat)):
        mean = beat[i]
        #std = 1 if i == len(beat) - 1 else beat[i + 1] - beat[i]  # Set std to 1 for the last point
        #std = (beat[i + 1] - beat[i]) / 2 
        gaussian = gaussian_function(x_values, mean, diffo[i] / 4)
        #print(diffo[i])
        #gaussian = np.exp(-(x_values - mean)**2 / (2 * (diffo[i]/4)**2)) / ((diffo[i]/4) * np.sqrt(2 * np.pi))
        plt.plot(x_values, gaussian,color='black' ,label=f'Gaussian {i + 1}')
    plt.vlines(beat, ymin=0, ymax=max(gaussian), color='g', label='beat',linewidth=1)
    plt.vlines(motion, ymin=0, ymax=max(gaussian), color='r', label='motion', linestyle='dashed',linewidth=2)
        #plt.fill_between(x_values, gaussian, where=(x_values > beat) & (x_values < motion), color='yellow', alpha=0.3, label='Area between beats and motion')

        #plt.scatter(beat, np.zeros_like(beat), color='red', marker='o', label='Beat Points')
    
        #plt.legend()
    """
    beat_inter = []
    for i in range(len(beat)):
        beat_time_value = beat[i]
        #print(beat_time_value)
        intersection_x = fsolve(intersection_function, beat[i], args=(beat_time_value, beat[i], diffo[i] / 4))[0]
        intersection_y = gaussian_function(intersection_x, beat[i], diffo[i] / 4)
        beat_inter.append(intersection_y)
        plt.scatter(intersection_x, intersection_y, color='r', marker='x', label=f'Intersection {i + 1}')

        #plt.scatter(intersection_x, intersection_y, color='green', marker='x', label=f'Intersection {i + 1}')
    
    motin_inter = []
    
    for i in range(len(motion)):
        
        motion_time_value = motion[i]
        #print(motion_time_value)
        intersection_x_x = fsolve(intersection_function, beat[i], args=(motion_time_value, beat[i], diffo[i] / 4))[0]
        intersection_yx = gaussian_function(intersection_x_x, motion[i], diffo[i] / 4)
        motin_inter.append(intersection_yx)
        plt.scatter(intersection_x_x, intersection_yx, color='g', marker='o', label=f'Intersection {i + 1}')

        #plt.scatter(intersection_x_x, intersection_yx, color='black', marker='o', label=f'Intersection {i + 1}')
    #plt.set_title(title)
    """
    plt.title(name)
    plt.xlabel('X Values')
    plt.ylabel('Gaussian Values')
    #plt.show()
    
    return score
import warnings


# The maximum allowable beat time
MAX_TIME = 30000.
#@staticmethod
def validate(reference_beats, estimated_beats):
    """Checks that the input annotations to a metric look like valid beat time
    arrays, and throws helpful errors if not.

    Parameters
    ----------
    reference_beats : np.ndarray
        reference beat times, in seconds
    estimated_beats : np.ndarray
        estimated beat times, in seconds
    """
    # If reference or estimated beats are empty,
    # warn because metric will be 0
    if reference_beats.size == 0:
        warnings.warn("Reference beats are empty.")
    if estimated_beats.size == 0:
        warnings.warn("Estimated beats are empty.")
    #for beats in [reference_beats, estimated_beats]:
        #util.validate_events(beats, MAX_TIME)
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
#@staticmethod
def cemgil(reference_beats,
        estimated_beats,
        cemgil_sigma=0.04):
    """Cemgil's score, computes a gaussian error of each estimated beat.
    Compares against the original beat times and all metrical variations.

    Examples
    --------
    >>> reference_beats = mir_eval.io.load_events('reference.txt')
    >>> reference_beats = mir_eval.beat.trim_beats(reference_beats)
    >>> estimated_beats = mir_eval.io.load_events('estimated.txt')
    >>> estimated_beats = mir_eval.beat.trim_beats(estimated_beats)
    >>> cemgil_score, cemgil_max = mir_eval.beat.cemgil(reference_beats,
                                                        estimated_beats)

    Parameters
    ----------
    reference_beats : np.ndarray
        reference beat times, in seconds
    estimated_beats : np.ndarray
        query beat times, in seconds
    cemgil_sigma : float
        Sigma parameter of gaussian error windows
        (Default value = 0.04)

    Returns
    -------
    cemgil_score : float
        Cemgil's score for the original reference beats
    cemgil_max : float
        The best Cemgil score for all metrical variations
    """
    from scipy.stats import norm
    validate(reference_beats, estimated_beats)
    diffo = np.diff(reference_beats)
    
    # When estimated beats are empty, no beats are correct; metric is 0
    if estimated_beats.size == 0 or reference_beats.size == 0:
        return 0., 0.
    # We'll compute Cemgil's accuracy for each variation
    accuracies = []
    for reference_beats in _get_reference_beat_variations(reference_beats):
        accuracy = 0
        # Cycle through beats
        for beat in reference_beats:
            # Find the error for the closest beat to the reference beat
            inter_beat= np.diff(reference_beats)

            beat_diff = np.min(np.abs(beat - estimated_beats))
            #print("Diff ########",len(beat_diff))
            # Add gaussian error into the accuracy
            accuracy += np.exp(-(beat_diff**2)/(2.0*( (diffo/4)**4)))
            #accuracy += np.exp(-(beat_diff**2)/(2.0*(cemgil_sigma**2))) 
            
                        
        # Normalize the accuracy
        accuracy /= .5*(estimated_beats.shape[0] + reference_beats.shape[0])
        # Add it to our list of accuracy scores
        #print(accuracy)
        accuracies.append(accuracy)
    #print(len(accuracies))
    # Return raw accuracy with non-varied annotations
    """
    plt.figure(figsize=(10, 6))
    
    # Plot beats
    plt.plot(reference_beats, np.zeros_like(reference_beats), 'r|', markersize=20, label='Reference Beats', markeredgewidth=2)
    plt.plot(estimated_beats, np.ones_like(estimated_beats), 'g|', markersize=20, label='Estimated Beats', markeredgewidth=2)

    # Plot Gaussian curves for each beat
    for beat in reference_beats:
        beat_diff = np.min(np.abs(beat - estimated_beats))
        gaussian_curve = norm.pdf(estimated_beats, loc=beat, scale=cemgil_sigma)
        plt.plot(estimated_beats, gaussian_curve, 'b--', alpha=0.3)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Intensity')
    plt.title('Cemgil\'s Score with Beats and Gaussian Curves')
    plt.legend()
    #plt.show()
    # and maximal accuracy across all variations
    """
    return accuracies[0], np.max(accuracies)

#@staticmethod
def continuity(reference_beats,
            estimated_beats,
            continuity_phase_threshold=0.1,
            continuity_period_threshold=0.3):
    """Get metrics based on how much of the estimated beat sequence is
    continually correct.

    Examples
    --------
    >>> reference_beats = mir_eval.io.load_events('reference.txt')
    >>> reference_beats = mir_eval.beat.trim_beats(reference_beats)
    >>> estimated_beats = mir_eval.io.load_events('estimated.txt')
    >>> estimated_beats = mir_eval.beat.trim_beats(estimated_beats)
    >>> CMLc, CMLt, AMLc, AMLt = mir_eval.beat.continuity(reference_beats,
                                                        estimated_beats)

    Parameters
    ----------
    reference_beats : np.ndarray
        reference beat times, in seconds
    estimated_beats : np.ndarray
        query beat times, in seconds
    continuity_phase_threshold : float
        Allowable ratio of how far is the estimated beat
        can be from the reference beat
        (Default value = 0.175)
    continuity_period_threshold : float
        Allowable distance between the inter-beat-interval
        and the inter-annotation-interval
        (Default value = 0.175)

    Returns
    -------
    CMLc : float
        Correct metric level, continuous accuracy
    CMLt : float
        Correct metric level, total accuracy (continuity not required)
    AMLc : float
        Any metric level, continuous accuracy
    AMLt : float
        Any metric level, total accuracy (continuity not required)
    """
    validate(reference_beats, estimated_beats)
    # Warn when only one beat is provided for either estimated or reference,
    # report a warning
    if reference_beats.size == 1:
        warnings.warn("Only one reference beat was provided, so beat intervals"
                      " cannot be computed.")
    if estimated_beats.size == 1:
        warnings.warn("Only one estimated beat was provided, so beat intervals"
                      " cannot be computed.")
    # When estimated or reference beats have <= 1 beats, can't compute the
    # metric, so return 0
    if estimated_beats.size <= 1 or reference_beats.size <= 1:
        return 0., 0., 0., 0.
    # Accuracies for each variation
    continuous_accuracies = []
    total_accuracies = []
    # Get accuracy for each variation
    for reference_beats in _get_reference_beat_variations(reference_beats):
        # Annotations that have been used
        n_annotations = np.max([reference_beats.shape[0],
                               estimated_beats.shape[0]])
        used_annotations = np.zeros(n_annotations)
        # Whether or not we are continuous at any given point
        beat_successes = np.zeros(n_annotations)
        for m in range(estimated_beats.shape[0]):
            # Is this beat correct?
            beat_success = 0
            # Get differences for this beat
            beat_differences = np.abs(estimated_beats[m] - reference_beats)
            # Get nearest annotation index
            nearest = np.argmin(beat_differences)
            min_difference = beat_differences[nearest]
            # Have we already used this annotation?
            if used_annotations[nearest] == 0:
                # Is this the first beat or first annotation?
                # If so, look forward.
                if m == 0 or nearest == 0:
                    # How far is the estimated beat from the reference beat,
                    # relative to the inter-annotation-interval?
                    if nearest + 1 < reference_beats.shape[0]:
                        reference_interval = (reference_beats[nearest + 1] -
                                              reference_beats[nearest])
                    else:
                        # Special case when nearest + 1 is too large - use the
                        # previous interval instead
                        reference_interval = (reference_beats[nearest] -
                                              reference_beats[nearest - 1])
                    # Handle this special case when beats are not unique
                    if reference_interval == 0:
                        if min_difference == 0:
                            phase = 1
                        else:
                            phase = np.inf
                    else:
                        phase = np.abs(min_difference/reference_interval)
                    # How close is the inter-beat-interval
                    # to the inter-annotation-interval?
                    if m + 1 < estimated_beats.shape[0]:
                        estimated_interval = (estimated_beats[m + 1] -
                                              estimated_beats[m])
                    else:
                        # Special case when m + 1 is too large - use the
                        # previous interval
                        estimated_interval = (estimated_beats[m] -
                                              estimated_beats[m - 1])
                    # Handle this special case when beats are not unique
                    if reference_interval == 0:
                        if estimated_interval == 0:
                            period = 0
                        else:
                            period = np.inf
                    else:
                        period = \
                            np.abs(1 - estimated_interval/reference_interval)
                    if phase < continuity_phase_threshold and \
                       period < continuity_period_threshold:
                        # Set this annotation as used
                        used_annotations[nearest] = 1
                        # This beat is matched
                        beat_success = 1
                # This beat/annotation is not the first
                else:
                    # How far is the estimated beat from the reference beat,
                    # relative to the inter-annotation-interval?
                    reference_interval = (reference_beats[nearest] -
                                          reference_beats[nearest - 1])
                    phase = np.abs(min_difference/reference_interval)
                    # How close is the inter-beat-interval
                    # to the inter-annotation-interval?
                    estimated_interval = (estimated_beats[m] -
                                          estimated_beats[m - 1])
                    reference_interval = (reference_beats[nearest] -
                                          reference_beats[nearest - 1])
                    period = np.abs(1 - estimated_interval/reference_interval)
                    if phase < continuity_phase_threshold and \
                       period < continuity_period_threshold:
                        # Set this annotation as used
                        used_annotations[nearest] = 1
                        # This beat is matched
                        beat_success = 1
            # Set whether this beat is matched or not
            beat_successes[m] = beat_success
        # Add 0s at the begnning and end
        # so that we at least find the beginning/end of the estimated beats
        beat_successes = np.append(np.append(0, beat_successes), 0)
        # Where is the beat not a match?
        beat_failures = np.nonzero(beat_successes == 0)[0]
        # Take out those zeros we added
        beat_successes = beat_successes[1:-1]
        # Get the continuous accuracy as the longest track of successful beats
        longest_track = np.max(np.diff(beat_failures)) - 1
        continuous_accuracy = longest_track/(1.0*beat_successes.shape[0])
        continuous_accuracies.append(continuous_accuracy)
        # Get the total accuracy - all sequences
        total_accuracy = np.sum(beat_successes)/(1.0*beat_successes.shape[0])
        total_accuracies.append(total_accuracy)
    print('CMLc, CMLt, AMLc, AMLt : ', (continuous_accuracies[0],
            total_accuracies[0],
            np.max(continuous_accuracies),
            np.max(total_accuracies)))
    
    # Grab accuracy scores
    return (continuous_accuracies[0],
            total_accuracies[0],
            np.max(continuous_accuracies),
            np.max(total_accuracies))
    
   


