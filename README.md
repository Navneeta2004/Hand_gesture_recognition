

# Hand Gesture Recognition using MediaPipe and TensorFlow

This Python script utilizes the MediaPipe library and TensorFlow to perform hand gesture recognition using a webcam feed. The code identifies hand landmarks using the MediaPipe Hands module, then feeds these landmarks into a pre-trained TensorFlow model for gesture recognition.

## Dependencies

Before running the script, make sure to install the required dependencies:

- OpenCV: `pip install opencv-python`
- NumPy: `pip install numpy`
- MediaPipe: `pip install mediapipe`
- TensorFlow: `pip install tensorflow`
- TensorFlow's Keras: `pip install keras`

## Usage

1. Ensure all dependencies are installed.
2. Run the script using a Python interpreter (`python script_name.py`).
3. The script will open a webcam feed, detect hand landmarks, and recognize gestures in real-time.

## Model and Gesture Names

The script loads a pre-trained gesture recognition model (`mp_hand_gesture`) and a file containing gesture names (`gesture.names`). Ensure these files are present in the working directory.

## Available Gestures

The script is capable of recognizing the following hand gestures:

- Peace
- Thumbs Up
- Thumbs Down
- Call Me
- Stop
- Rock
- Live Long
- Fist
- Smile

These gestures are identified based on a pre-trained model, and the recognized gesture is displayed on the video feed in real-time.

Feel free to customize the list or provide additional details about each gesture if needed. Users can refer to this section to understand the gestures that the script is designed to recognize.

## Controls

- Press 'q' to exit the script.

## Notes

- The script draws hand landmarks on the webcam feed using MediaPipe.
- Recognized gestures are displayed on the video feed.
- The script keeps track of the most common gesture over the last 10 frames.

## License

This project is licensed under the MIT License - see the [LICENSE] file for details.
```

Feel free to customize this template according to your specific needs or add more details about the project, its purpose, and any other relevant information.
