# face-mask-detection

Dataset link: https://drive.google.com/drive/folders/1-0sK9GcTTR_7-w0pxkgF-9YRdm1rmk_X?usp=sharing

Objective:
Training: Here we’ll focus on loading our face mask detection dataset from disk, training a model (using Keras/TensorFlow) on this dataset, and then serializing the face mask detector to disk
Deployment: Once the face mask detector is trained, we can then move on to loading the mask detector, performing face detection, and then classifying each face as with_mask or without_mask

Packages Required:
      tensorflow,Keras,OpenCV
      
 The convolution network consists of two pairs of Conv and MaxPool layers to extract features from the dataset. Which is then followed by a Flatten and Dropout layer to convert the data in 1D and ensure overfitting.


train_mask_detector.py: Accepts our input dataset and fine-tunes MobileNetV2 upon it to create our mask_detector.model. A training history plot.png containing accuracy/loss curves is also produced

detect_mask_image.py: Performs face mask detection in static images

detect_mask_video.py: Using your webcam, this script applies face mask detection to every frame in the stream
