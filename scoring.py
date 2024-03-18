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
import warnings

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
#@staticmethod
def normalizeAngle( angle) : 

        angle %= 360
        if (angle > 180.0) : 
            return angle - 360
        if (angle <= -180):
            return angle + 360
        return angle
#@staticmethod
def calculate_score_gauss(x,alpha):
    #x = p-g
    expo = np.exp(-x**2 / (2 * alpha**2))
    #score_i = 2 * (1 - sigmoid_term)
    return expo 
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
   
    
    return angle_, r_1, median_direction, angular_variation, angular_deviation,mean_direction,mean_vector_length
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
    CSLc : float
        Correct metric level, continuous accuracy
    CSLt : float
        Correct metric level, total accuracy (continuity not required)
    
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
            total_accuracies[0])
#@staticmethod
def _fast_hit_windows(ref, est, window):
    '''Fast calculation of windowed hits for time events.

    Given two lists of event times ``ref`` and ``est``, and a
    tolerance window, computes a list of pairings
    ``(i, j)`` where ``|ref[i] - est[j]| <= window``.

    This is equivalent to, but more efficient than the following:

    >>> hit_ref, hit_est = np.where(np.abs(np.subtract.outer(ref, est))
    ...                             <= window)

    Parameters
    ----------
    ref : np.ndarray, shape=(n,)
        Array of reference values
    est : np.ndarray, shape=(m,)
        Array of estimated values
    window : float >= 0
        Size of the tolerance window

    Returns
    -------
    hit_ref : np.ndarray
    hit_est : np.ndarray
        indices such that ``|hit_ref[i] - hit_est[i]| <= window``
    '''

    ref = np.asarray(ref)
    est = np.asarray(est)
    ref_idx = np.argsort(ref)
    ref_sorted = ref[ref_idx]

    left_idx = np.searchsorted(ref_sorted, est - window, side='left')
    right_idx = np.searchsorted(ref_sorted, est + window, side='right')

    hit_ref, hit_est = [], []

    for j, (start, end) in enumerate(zip(left_idx, right_idx)):
        hit_ref.extend(ref_idx[start:end])
        hit_est.extend([j] * (end - start))

    return hit_ref, hit_est
#@staticmethod
def _bipartite_match(graph):
    """Find maximum cardinality matching of a bipartite graph (U,V,E).
    The input format is a dictionary mapping members of U to a list
    of their neighbors in V.

    The output is a dict M mapping members of V to their matches in U.

    Parameters
    ----------
    graph : dictionary : left-vertex -> list of right vertices
        The input bipartite graph.  Each edge need only be specified once.

    Returns
    -------
    matching : dictionary : right-vertex -> left vertex
        A maximal bipartite matching.

    """
    # Adapted from:
    #
    # Hopcroft-Karp bipartite max-cardinality matching and max independent set
    # David Eppstein, UC Irvine, 27 Apr 2002

    # initialize greedy matching (redundant, but faster than full search)
    matching = {}
    for u in graph:
        for v in graph[u]:
            if v not in matching:
                matching[v] = u
                break

    while True:
        # structure residual graph into layers
        # pred[u] gives the neighbor in the previous layer for u in U
        # preds[v] gives a list of neighbors in the previous layer for v in V
        # unmatched gives a list of unmatched vertices in final layer of V,
        # and is also used as a flag value for pred[u] when u is in the first
        # layer
        preds = {}
        unmatched = []
        pred = dict([(u, unmatched) for u in graph])
        for v in matching:
            del pred[matching[v]]
        layer = list(pred)

        # repeatedly extend layering structure by another pair of layers
        while layer and not unmatched:
            new_layer = {}
            for u in layer:
                for v in graph[u]:
                    if v not in preds:
                        new_layer.setdefault(v, []).append(u)
            layer = []
            for v in new_layer:
                preds[v] = new_layer[v]
                if v in matching:
                    layer.append(matching[v])
                    pred[matching[v]] = v
                else:
                    unmatched.append(v)

        # did we finish layering without finding any alternating paths?
        if not unmatched:
            unlayered = {}
            for u in graph:
                for v in graph[u]:
                    if v not in preds:
                        unlayered[v] = None
            return matching

        def recurse(v):
            """Recursively search backward through layers to find alternating
            paths.  recursion returns true if found path, false otherwise
            """
            if v in preds:
                L = preds[v]
                del preds[v]
                for u in L:
                    if u in pred:
                        pu = pred[u]
                        del pred[u]
                        if pu is unmatched or recurse(pu):
                            matching[v] = u
                            return True
            return False

        for v in unmatched:
            recurse(v)
#@staticmethod
def match_events(ref, est, window, distance=None):
    """Compute a maximum matching between reference and estimated event times,
    subject to a window constraint.

    Given two lists of event times ``ref`` and ``est``, we seek the largest set
    of correspondences ``(ref[i], est[j])`` such that
    ``distance(ref[i], est[j]) <= window``, and each
    ``ref[i]`` and ``est[j]`` is matched at most once.

    This is useful for computing precision/recall metrics in beat tracking,
    onset detection, and segmentation.

    Parameters
    ----------
    ref : np.ndarray, shape=(n,)
        Array of reference values
    est : np.ndarray, shape=(m,)
        Array of estimated values
    window : float > 0
        Size of the window.
    distance : function
        function that computes the outer distance of ref and est.
        By default uses ``|ref[i] - est[j]|``

    Returns
    -------
    matching : list of tuples
        A list of matched reference and event numbers.
        ``matching[i] == (i, j)`` where ``ref[i]`` matches ``est[j]``.

    """
    if distance is not None:
        # Compute the indices of feasible pairings
        hits = np.where(distance(ref, est) <= window)
    else:
        hits = _fast_hit_windows(ref, est, window)

    # Construct the graph input
    G = {}
    for ref_i, est_i in zip(*hits):
        if est_i not in G:
            G[est_i] = []
        G[est_i].append(ref_i)

    # Compute the maximum matching
    matching = sorted(_bipartite_match(G).items())

    return matching
#@staticmethod
def f_measure (precision, recall, beta=1.0):
    """Compute the f-measure from precision and recall scores.

    Parameters
    ----------
    precision : float in (0, 1]
        Precision
    recall : float in (0, 1]
        Recall
    beta : float > 0
        Weighting factor for f-measure
        (Default value = 1.0)

    Returns
    -------
    f_measure : float
        The weighted f-measure

    """

    if precision == 0 and recall == 0:
        return 0.0

    return (1 + beta**2)*precision*recall/((beta**2)*precision + recall)
#@staticmethod
def binary(reference_beats,
              estimated_beats,
              f_measure_threshold=0.07):
    """Compute the F-measure of correct vs incorrectly predicted beats.
    "Correctness" is determined over a small window.

    Examples
    --------
    >>> reference_beats = mir_eval.io.load_events('reference.txt')
    >>> reference_beats = mir_eval.beat.trim_beats(reference_beats)
    >>> estimated_beats = mir_eval.io.load_events('estimated.txt')
    >>> estimated_beats = mir_eval.beat.trim_beats(estimated_beats)
    >>> f_measure = mir_eval.beat.f_measure(reference_beats,
                                            estimated_beats)

    Parameters
    ----------
    reference_beats : np.ndarray
        reference beat times, in seconds
    estimated_beats : np.ndarray
        estimated beat times, in seconds
    f_measure_threshold : float
        Window size, in seconds
        (Default value = 0.07)

    Returns
    -------
    f_score : float
        The computed F-measure score

    """
    validate(reference_beats, estimated_beats)
    # When estimated beats are empty, no beats are correct; metric is 0
    if estimated_beats.size == 0 or reference_beats.size == 0:
        return 0.
    # Compute the best-case matching between reference and estimated locations
    matching = match_events(reference_beats,
                                 estimated_beats,
                                 f_measure_threshold)

    precision = float(len(matching))/len(estimated_beats)
    recall = float(len(matching))/len(reference_beats)
    return f_measure(precision, recall)
#@staticmethod
def gaussian_function(x, mean, std):
    return np.exp(-(x - mean) ** 2 / (2 * std ** 2)) / (std * np.sqrt(2 * np.pi))
#@staticmethod
def gaussian_score (beat,motion,name) : 
     
    x_values = np.linspace(np.min(beat) - 1, np.max(beat) + 1, 1000)
    import math
    diffo = []
    from scipy.optimize import fsolve

    for i in range (0,len(beat)-1,1) : 
        diffo.append(beat[i+1]- beat[i])
    last_ = diffo[-1]
    diffo =  np.append(diffo, last_)
    diffo = diffo **2

    beat_diff = []
    for beato in beat:
            # Find the error for the closest beat to the reference beat
        beat_diff.append(np.min(np.abs(beato - motion)))
    expo = []
    accuracy = 0
    for i in range (0,len(beat_diff),1): 
        expo.append(calculate_score_gauss(beat_diff[i],diffo[i] / 4)) 
        accuracy += np.exp(-(beat_diff[i]**2)/(2.0*(diffo[i] / 4)**2))

    leng = max(len(beat),len(motion))
    score =  np.sum(accuracy) / leng
    #return gaussian_score
    """
    plt.figure(figsize=(8, 4))
    for i in range(len(beat)):
        mean = beat[i]
        gaussian = gaussian_function(x_values, mean, diffo[i] / 4)
        plt.plot(x_values, gaussian,color='black' ,label=f'Gaussian {i + 1}')
    plt.vlines(beat, ymin=0, ymax=max(gaussian), color='g', label='beat',linewidth=1)
    plt.vlines(motion, ymin=0, ymax=max(gaussian), color='r', label='motion', linestyle='dashed',linewidth=2)
    plt.title(name)
    plt.xlabel('X Values')
    plt.ylabel('Gaussian Values')
    #plt.show()
    """
    return score, expo
#@staticmethod
def normalize_values(original_values, new_min=0, new_max=30):
    normalized_values = (original_values * (new_max - new_min)) + new_min
    return normalized_values
#@staticmethod
def sigmoid_score (beat,motion,alpha) : 
    beat_diff = []
    for beato in beat:
        beat_diff .append( (np.min(np.abs(beato- motion))))
    accuracy = []
    for i in range (0,len(beat_diff),1): 
        sigmoid_term = (1 / (1 + np.exp(-(alpha*(abs(beat_diff[i]))))))
        accuracy.append( 2 * (1 - sigmoid_term))
    leng = max(len(beat),len(motion))   
    return np.sum(accuracy) / leng ,accuracy
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
#@staticmethod
def normalize_dtw(v1, v2):
    """
    Compute the normalized DTW distance between two vectors v1 and v2.
    """
    dtw_dist = dtw_distance(v1, v2)
    relative_difference = max(abs(max(v1) - min(v1)), abs(max(v2) - min(v2)))
    normalized_distance = dtw_dist / relative_difference
    return normalized_distance
#@staticmethod
def normalize_to_length(value, length):
    min_value = 0
    max_value = length
    normalized_value = (value - min_value) / (max_value - min_value)
    return normalized_value
#@staticmethod
def scores_motion_beat(beat, motion, name):
    """
    Calculate various scores and plot motion and beat data.

    Parameters:
        beat (array_like): Array containing beat data.
        motion (array_like): Array containing motion data.
        name (str): Name of the plot.

    Returns:
        None
    """
    # Calculate phase-related values
    Phi, r_1, median_direction, angular_variation, angular_deviation, mean_direction, mean_vector_length = phase_relative(beat, motion)
    
    # Calculate Gaussian score
    gauss_score ,lst_gauss = gaussian_score(beat, motion, name)
    
    # Calculate sigmoid score C= 20 
    sg_score,lst_sig = sigmoid_score(beat, motion, 20)
    
    # Calculate continuity scores
    CSLc, CSLt = continuity(beat, motion, continuity_phase_threshold=0.15, continuity_period_threshold=0.15)
    
    # Calculate binary score
    f_score = binary(beat, motion)
    # Calculer la distance DTW normalised and score 
    distance_normalised = normalize_dtw(beat,motion)
    score_dtw = 1-distance_normalised
    # Normalize and calculate mean direction
    custom = np.mean(np.abs(Phi))
    mean_direction = np.angle(np.mean(np.exp(1j * np.radians(Phi))))
    mean_vector_length = np.abs(np.mean(np.exp(1j * np.radians(Phi))))
    normalized_values = normalize_values(mean_vector_length, new_min=0, new_max=len(beat))
    md = np.degrees(mean_direction)
    
    # Print calculated values
    print("Mean Direction: ", md)
    print("|R|: ", mean_vector_length)
    print("DTW: ", score_dtw)
    print("Score Gaussian:", gauss_score)
    print("Score Sigmoid : ", sg_score)
    print("CMLc: ", CSLc)
    print("CMLt: ", CSLt)
    print("Binary : ", f_score)
    
    # Plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    ax.scatter(np.radians(Phi), r_1, label='Phases', color='black')
    ax.plot([np.radians(md), np.radians(md)], [0, normalized_values], color='g', linewidth=2, label='Mean Direction')
    ax.set_rmax(max(r_1))
    
    # Add legend and show plot
    ax.legend()
    plt.show()
    return lst_sig,lst_gauss
