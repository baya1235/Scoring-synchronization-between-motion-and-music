
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import cv2
from math import*
import time
import lxml.etree as etree
import os
import librosa.display
import scoring as sync

class mpHands:
    import mediapipe as mp
    def __init__(self,maxHands=2,tol1=.8,tol2=.8):
        self.hands=self.mp.solutions.hands.Hands(False,maxHands,tol1,tol2)
    def Marks(self,frame):
        myHands=[]
        handsType=[]
        #frameRGB = cv2.resize(frame, (1280, 720))
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        results=self.hands.process(frameRGB)
        if results.multi_hand_landmarks != None:
            #print (results.multi_handedness)
            for hand in results.multi_handedness :
                handType = hand.classification[0].label
                handsType.append(handType)
            for handLandMarks in results.multi_hand_landmarks:
                myHand=[]
                for landMark in handLandMarks.landmark:
                    myHand.append((int(landMark.x*1280),int(landMark.y*720)))
                    #print(results)
                    #myHand.append(coords)
                    
                    
                myHands.append(myHand)
        return myHands , handsType

def progress_bar(current_frame, total_frames, bar_length=20):
    progress = current_frame / total_frames
    num_hashes = int(round(progress * bar_length))
    num_spaces = bar_length - num_hashes

    print(f"\rProcessing: [{'#' * num_hashes + ' ' * num_spaces}] {int(progress * 100)}%", end="", flush=True)
def euclidean_distance(x,y):
        
        dist = sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))
        return dist
def load_annotations(file_):
    ''' Read a musicdescription xml file.

    Returns a 4-column matrix whose columns are time (in sec), 
    
    is_beat (1 if the marker is a beat or a downbeat 0 in case of tatum), 
    is_tatum(1 if the marker is a tatum, a beat or a downbeat), and 
    is_measure(1 if the marker is a downbeat).


    Args:
        file_ (str): path of the xml file.

    Returns:
        np.array: a 4-column (time, is_beat, is_tatum, is_measure) matrix
        with each row representing a marker.
    '''
    tree = etree.parse(file_)
    data = []
    beat_tt=[]
    down_beat =[]
    for elem in tree.iter():
        if elem.tag[-7:] == 'segment':
            a = elem.getchildren()
            if 'beat' in a[0].keys():
                b = int(a[0].get('beat'))
                tatum = int(a[0].get('tatum'))
                t = float(elem.get('time'))
                #beat_tt.append(t)
                m = int(a[0].get('measure'))
                #down_beat.append(m)
                data.append([t, b, tatum, m])
                if b == 1 and   3 < t  :
                   ###print('beat : ',t)
                   #rounded_x = round(t, 1)
                   beat_tt.append(t)
                """
                if b == 0 : 
                   ###print('Tatum',t)                   
                """ 
                if m==1 : 
                   ###print('downbeat:',t)
                   
                   down_beat.append(t)


    return  beat_tt , down_beat
def find_peaks(signal):
    # Calculate the baseline as the average of the signal
    baseline = sum(signal) / len(signal)
    # Initialize an empty list to store the minimum peak indices
    min_peak_indices = []
    # Iterate over the signal
    for index, value in enumerate(signal):
        # If the value is less than the baseline, add the index to the list
        if value < baseline:
            min_peak_indices.append(index)
    # Return the list of minimum peak indices
    return min_peak_indices
def find_peak_ranges_1(signal):
    # Calculate the baseline as the average of the signal
    baseline = sum(signal) / len(signal)
    # Initialize a list to store peak ranges
    peak_ranges = []
    # Initialize variables to track the start and end of a peak range
    start_index = None

    # Iterate over the signal
    for index, value in enumerate(signal):
        # Check if the value is greater than the baseline
        if value < baseline:
            # If it's the start of a new peak range, update the start_index
            if start_index is None:
                start_index = index
        else:
            # If it's the end of a peak range, add the range to the list
            if start_index is not None:
                peak_ranges.append((start_index, index - 1))
                start_index = None

    # Check if there's an ongoing peak range at the end of the signal
    if start_index is not None:
        peak_ranges.append((start_index, len(signal) - 1))

    # Return the list of peak ranges
    return peak_ranges
def find_peak_ranges(signal):
    # Calculate the baseline as the average of the signal
    baseline = sum(signal) / len(signal)
    # Initialize a list to store peak ranges and their values
    peak_data = []
    # Initialize variables to track the start and end of a peak range
    start_index = None

    # Iterate over the signal
    for index, value in enumerate(signal):
        # Check if the value is greater than the baseline
        if value < baseline:
            # If it's the start of a new peak range, update the start_index
            if start_index is None:
                start_index = index
        else:
            # If it's the end of a peak range, add the range and its values to the list
            if start_index is not None:
                peak_data.append((start_index, index - 1, signal[start_index:index]))
                start_index = None

    # Check if there's an ongoing peak range at the end of the signal
    if start_index is not None:
        peak_data.append((start_index, len(signal) - 1, signal[start_index:]))

    # Return the list of peak ranges and their values
    return peak_data
def tapping_tempo_measure(list_left,list_right,list_time,audio_beat,test_info,peak,audio,namo,Var_text_name,tempo) : 
        import matplotlib.pyplot as plt
        #matplotlib.use('TkAgg')
        import csv
        

        from scipy.spatial.distance import euclidean
        import lxml.etree as etree
        from fastdtw import fastdtw
        import mir_eval
        import matplotlib.pyplot as plt
        import librosa
        xo, sr = librosa.load(audio)
        
        #list_right =  normalize (list_right) 
        beat,d_beat = load_annotations(audio_beat)
        delta_t = np.diff(beat)
            
        period = np.mean(delta_t)

        #frequency = 1/period
        tempo_audio = 60 / period
        #print(period)
        #print(int(period))
        tag_test = namo+'_' +Var_text_name
        fg = (60 / tempo_audio) / (1/30)
        vector_time30fps_arr = np.array(list_time)
        b1egin_left = np.array(list_left)
        b1egin_right = list_left[200:]
        sig_inv = list_left * (-1)
        delta_t = np.diff(beat)
            
        period = np.mean(delta_t)

        #frequency = 1/period
        tempo_audio = 60 / period
        #print(period)
        #print(int(period))

        fg = (60 / tempo_audio) / (1/30)
        #print(int(fg))
        signal_mean = np.mean(list_left)
        signal_std = np.std(list_left)

        # Set 'mph' as a multiple of the standard deviation above the mean
        mph_multiplier = 2  # Adjust as needed
        mph = signal_mean+ mph_multiplier * signal_std 
        
        baseline = sum(list_left) / len(list_left)
 
      
        peak_indices_1 = find_peaks(list_left)
        rangessss = find_peak_ranges(list_left) 
        rangess = find_peak_ranges_1(list_left) 
        mid = np.mean(list_left)
        import matplotlib.pyplot as plt
 
        filtered_values = []
        peak_fn = []

        for i in range(0, len(rangessss), 1):
            a = rangessss[i]
            val = a[2]
            start = a[0]
            end = a[1]
            vecto = np.arange(start, end + 1)
            #print(val)
            

            for j in range(0, len(val), 1):
                if abs(val[j]) < abs(mid*(3/4)):
                    #print(abs(val[j]),abs(mid/2))
                    filtered_values.append(j)
            #print('filterd value : ',filtered_values)
            if len(filtered_values) > 0 :
                peak_min = vecto[filtered_values[0]]
                #print('*******', peak_min)
                peak_fn.append(peak_min)

            else:
                # Handle the case when the array is empty
                print(" ")
            filtered_values = []
        peak_time_distance_ = vector_time30fps_arr[peak_fn]
        motion_event = []
        for i in range(1, len(peak_time_distance_)):
            if peak_time_distance_[i] > 3  :
               motion_event.append(peak_time_distance_[i])
        
        import matplotlib.pyplot as plt
        # Plot the signal and highlight the detected peaks
        
        plt.figure(figsize=(18, 4))

        
        plt.subplot(2,1,1)
        plt.plot(list_left, label="Finger spacing")
        plt.scatter(peak_indices_1, [list_left[i] for i in peak_indices_1], color='black', label="Motion beat candidate ")

        plt.legend()
        plt.grid(True)
        
        for start, end in rangess:
            plt.axvspan(start, end, color='red', alpha=0.3, label='Peak Range')

        plt.subplot(2,1,2)
        plt.plot(list_time,list_left, label="Finger spacing")
        plt.vlines(peak_time_distance_, min(list_left),max(list_left), label='Motion beat', color='g', linewidth=2)
        plt.legend()
        plt.grid(True)
        


        info = test_info +'_'+ namo
        print(info)
        score_mir = []
        conti_vec = []
        b1 = np.array(beat)
        b2 = np.array(motion_event)
        lst_sig,lst_gauss = sync.scores_motion_beat(b1,b2,test_info)
        plt.figure(figsize=(18, 4))
        #plt.plot(list_time,list_left, linewidth=0.8 ,label="Signal motion",color ='b')
        plt.plot(beat, lst_gauss,'o-', label="Score gauss",color ='black',linewidth=1 )
        plt.plot(beat, lst_sig,'*-', label="Score sig 10",color ='orange',linewidth=1 )
        #plt.plot(beat, lst20,'+-', label="Score sig  20",linewidth=1.5 )
        #for i in range(len(beat)):
        plt.vlines(beat, 0,1, colors='red',label='Beat', linestyles='dashed',linewidth=0.8)
        plt.vlines(b2, 0,1, colors='g',label='Visual beat', linewidth=0.8)
        librosa.display.waveplot(xo, sr=sr ,alpha=0.2)
        plt.xlabel("Time")
        plt.ylabel("Signal Value")
        #plt.title("Signal with Detected Peaks Detecting Wide Peaks")
        #plt.legend()
        plt.ylim(0, 1)
        #plt.grid(True)
        nm = test_info + '_plot.png'
        nmm = test_info + '_wav.png'
        base_path = './'

        # Create the full path
        full_path = os.path.join(base_path, nm)
        plt.savefig(full_path)
        plt.figure(figsize=(18, 4))
        librosa.display.waveplot(xo, sr=sr ,alpha=0.4)
        plt.vlines(b1, -1,1, colors='r',label='Visual beat', linewidth=1.5)
        full_path_1 = os.path.join(base_path, nmm)
        #full_path = os.path.join(base_path, nmm)
        plt.savefig(full_path_1)
        delta_t = np.diff(b2)
        file_path_txt = test_info+'.txt'
        with open(file_path_txt, 'w') as file:
            # Write each element of the array to the file
            for item in peak_fn:
                file.write(str(item) + '\n')
         
        # Calculate the average time difference
        period = np.mean(delta_t)
        tempo_vis = 60/period
        score_vis_aud = abs(tempo - tempo_vis) / tempo

        delta_t_m1 = np.diff(b2)
            
        period_m1 = np.mean(delta_t_m1)

        #frequency = 1/period
        tempo_motion1 = 60 / period_m1
        taille_beat = int(len(b1))
        taille_vis = int(len(b2))
        add_missed =  taille_vis - taille_beat 

        plt.show()

def extract_test_info(path):
    # Extract the part of the path after the last "/"
    test_info = path.split("/")[-2]

    return test_info

def normalize(value, min_val, max_val):
    if max_val == min_val:
        return 0.5
    normalized = (value - min_val) / (max_val - min_val)
    return normalized

def analyse_sync_snapping (path_video,audio_beat,test_info,audio,namo,Var_text_name,tempo) : 
    #def Play_snap(self,name_b,name_test_b,hand_sd_b,texto,file,bpm,weight_b,height_b):       
        

        # Initialisation des valeurs min et max
        min_val = float("inf")
        max_val = float("-inf")
        min_val_g = float("inf")
        max_val_g = float("-inf")
       

        findHands=mpHands(2)
        prev_frame_time= 0
        dist_right = 0
        dist_left = 0 
        angle_dg = 0
        dist_right = 0 
        list_dist = []
        list_ang = []
        angle_right = 0
        frame_nbr = 0
        list_dist_r = []
        list_time =[]
        list_ang_r = []
        enter_condition = 0  
        out_condition = 0
        enter_condition_1 = 0
        enter_condition_start = 0 
        count_tap_start=  0 
        enter_condition_start_l = 0  
        out_condition_1 = 0
        condition = 0
        condition_1 = 0
        condition_start = 0
        condition_start_l = 0
        p_right_2 = (0,0)
        peak=[]
        #path_video ='_snaping_video__testt.avi'
        #path_video = '_snaping_video_.avi'
        cap = cv2.VideoCapture(path_video)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        current_frame = 0 
        time_step = 0
        p_right_2 = (0,0)
        p_right  = (0,0)
        normalized_value = 0 
        #xdata, ydata = [], []
        #ln, = ax.plot([], [])

        while cap.isOpened():
            ret, image = cap.read()
            image = cv2.flip(image, 1)
            if not ret:
                #print("Video finished.")
                break
            #image_1 = cv2.resize(frame, (1280, 720))
            frame = cv2.resize(image, (1280, 720))
            hamza = 1 
            #affiche = cv2.rectangle(image,(580,520) , (750,620) ,(0, 255, 0),4)
            #condition = cv2.putText(image,"Start",(620,570), cv2.FONT_HERSHEY_SIMPLEX,1 , (0,0,250), 4)
            
            hand_to_detect = 'Right'    
            handData , handsType = findHands.Marks(frame)
            for hand , handType in zip (handData , handsType):
                if handType == hand_to_detect:   
                    if handType == 'Right' :
                        handColor =(255,255,0)
                        p_right = hand[8]
                        p_right_2 = hand[4]
                        p_right_z = hand[0]
                        dist_right = euclidean_distance(p_right,p_right_2)
                        
                    if handType == 'Left' : 
                        handColor =(0,0,255)
                        p_left = hand[8]
                        p_leftpouce_2 = hand[1]
                        p_left_2 = hand[4]
                        dist_left =euclidean_distance(p_left,p_left_2)
                                              
                    for ind in [0,4,5,8,11,14,17,20]:
                        cv2.circle(frame,hand[ind],5,handColor,3)
                    
                    break
            cv2.line(frame, p_right_2, p_right, (255,0,0),3)     
            frame_nbr =frame_nbr + 1        
            
            if frame_nbr >= 200 : 
                time_step = time_step + (1/30)
                list_time.append(time_step)
                list_dist_r.append(dist_right)
                list_ang_r.append(dist_left)
                    # Mettre à jour les valeurs minimale et maximale
                if dist_right < min_val:
                    min_val = dist_right
                if dist_right > max_val:
                    max_val = dist_right
                if angle_right < min_val_g:
                    min_val_g = angle_right
                if angle_right > max_val_g:
                    max_val_g = angle_right    

                # Normaliser la valeur entre 0 et 1
                normalized_value = normalize(dist_right, min_val, max_val)
                normalized_value_g = normalize(angle_right, min_val_g, max_val_g)
                list_dist.append(normalized_value)
                list_ang.append(dist_right)
                #cv2.putText(frame,"distance : "+ str(dist_right),(700, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                #cv2.putText(frame,"distance normamlize : "+ str(normalized_value),(700, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                rd = int(dist_right)
                #cv2.circle(frame,p_right_2,rd, (0,160,0), -1)
                condition_start =  dist_right < 6
                if condition_start == True   and enter_condition_start ==0 : 
                   count_tap_start = count_tap_start + 1; enter_condition_start +=1
                       
                if condition_start == False and enter_condition_start == 1 :
                   count_tap_start +=1 
                   enter_condition_start = 0 
                   #print(out_condition)                   
                if count_tap_start==2:
                   count_tap_start = 0
                   #cv2.circle(frame,(660,300), 50, (0,160,0), -1)
                   peak.append(frame_nbr)
                  
               
            # Faire quelque chose avec la valeur normalisée (par exemple, l'afficher)
            #print(normalized_value_g)
            #progress = int(frame_nbr / num_frames * 20)
            #print(f"\r[{'#' * progress}{' ' * (20 - progress)}] {frame_nbr}/{num_frames}", end="")
            
    
            new_frame_time=time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.putText(frame,"fps : "+ str(fps),(1100, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            cv2.putText(frame,"frames : " + str(frame_nbr),(1100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            cv2.putText(frame,"Time : "+str(time_step),(1100, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            current_frame += 1
            progress_bar(current_frame, total_frames) 

            #cv2.imshow('Fingers Music Synch', frame)
            if cv2.waitKey(10) & 0xFF ==ord('q'):
                
                break 
        cap.release()
        cv2.destroyAllWindows()
        tapping_tempo_measure(list_dist,list_ang,list_time,audio_beat,test_info,peak,audio,namo,Var_text_name,tempo)


def main():
    participants_folder = "./video"
    
    for participant_folder in os.listdir(participants_folder):
        #print(participants_folder)
        participant_folder_path = os.path.join(participants_folder, participant_folder)
        print(participant_folder_path)
        if os.path.isdir(participant_folder_path):
            tapping_folder_path = os.path.join(participant_folder_path, 'tapping')
            
            if os.path.exists(tapping_folder_path) and os.path.isdir(tapping_folder_path):
                namo = str(participant_folder)
                for root, dirs, files in os.walk(tapping_folder_path):
                    for file in files:
                        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                            video_path = os.path.join(root, file)
                            test_info = extract_test_info(video_path)
                            
                            print('###########Debut################')
                            
                            audio_beat = "./audio/disco.00029.wav.xml"
                            audio = "./audio/disco.00029.wav"
                            Var_text_name = 'disco_classe_binary_meduim'
                            tempo = 110
                            print(f"Processing video :",test_info)
                            print(f"Processing audio:",audio_beat)
                            analyse_sync_snapping (video_path,audio_beat,test_info,audio,namo,Var_text_name,tempo)
                            
                            print('#############Fin##############')
                
                
if __name__ == "__main__":
    main()