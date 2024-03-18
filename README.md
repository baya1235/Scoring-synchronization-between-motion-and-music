# Scoring-synchronization-between-motion-and-music
This repository contains the implementation of the methods described in the paper titled "Scoring Synchronization Between Music and Motion: Local vs Global Approaches," submitted at EUSIPCO 2024. The code provided here enables the reproduction of the testing results reported in the paper.


# Introduction 
In the field of music and motion synchronization, understanding the alignment between musical events and corresponding motion cues is crucial. This repository provides a set of methods and algorithms aimed at scoring the synchronization between music and motion, exploring both local and global approaches.


# Results on Synthetic Data
1.Run the synthetic data  script:

`python evaluation_synth_synchronization.py` 

Synthetic data generation allows for controlled experiments and evaluation of algorithm performance under various conditions. 

This script specifically generates synthetic data and calculates scores based on it,


# Results on Real Data
1.Run the real data evaluation script:
`python evaluation_video_synchronization.py`

The script facilitates experiments with real-world data, evaluating algorithm performance using audio and video recordings of participants exhibiting varying levels and qualities of synchronization. First preprocessing for audio and video data before conducting synchronization score evaluations.
 
The data is stored in the 'video' and 'audio' folders, providing access to the necessary resources for conducting evaluations.
  
To test your data place the audio files in the audio folder and the video files in the video folder within the repository directory.



 





