# EMG EdgeML for Gesture recognition and real-time controls
## Collection:
[collect]("./collect.py")
The data collection program prompts the user to follow prompts to open or close the hand with varying timings depending on
the gesture measured.
  
## Preprocessing:
[collect]("./collect.py")
[train]("./traincnn.py")
Principal component analysis (PCA) for feature extraction. 
  
## Training:
[train]("./traincnn.py")
  
Labeled and preprocessed data were stored in CSV files and parsed into training and testing datasets at a ratio of 7:3, respectively.
Testing Accuracy in real time was calculated similarly to training where users follow the expected label during live inference and compare it with the model’s predicted output
  
## Plotting:
[plot]("./plot.py")
Visualize CSV output with matplotlib
