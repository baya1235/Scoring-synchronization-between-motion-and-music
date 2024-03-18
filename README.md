# Scoring-synchronization-between-motion-and-music
This repository contains the implementation of the methods described in the paper titled "Scoring Synchronization Between Music and Motion: Local vs Global Approaches," presented at EUSIPCO 2024. The code provided here enables the reproduction of the testing results reported in the paper.


# Introduction 
In the field of music and motion synchronization, understanding the alignment between musical events and corresponding motion cues is crucial. This repository provides a set of methods and algorithms aimed at scoring the synchronization between music and motion, exploring both local and global approaches.



# Usage 
Clone the repository: 

`git clone https://github.com/your_username/Scoring-synchronization-between-motion-and-music.git` 

Navigate to the cloned directory:

`cd Scoring-synchronization-between-motion-and-music` 
# Testing on Synthetic Data
The synthetic_data_test.py script in the repository facilitates testing synchronization methods on synthetic data. 

Synthetic data allows for controlled experiments and evaluation of algorithm performance under different conditions.

1.Run the synthetic data test script:

`python evaluation_synth_synchronization.py` 

# Testing on Real Data
The real_data_evaluation.py script in the repository allows for the evaluation of synchronization methods on real-world audio and video data. Real data evaluation provides insights into algorithm performance in practical scenarios.

Usage
To evaluate synchronization methods on real data, follow these steps:

Ensure that you have the necessary real-world audio and video data available. 

To test your data place the audio files in the audio folder and the video files in the video folder within the repository directory.

1.Run the real data evaluation script:

`python evaluation_video_synchronization.py` 





