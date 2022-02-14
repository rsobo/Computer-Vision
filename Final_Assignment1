# Ryan Sobolewski
# Assignment 1
# CAP 4410

import cv2
import matplotlib
import numpy
from matplotlib import pyplot as plt

global editedFrame


# this function takes the inputted values for contrast and brightness
# then that data gets sent to editFrame to apply the edit
def changeBrightCont(brightness):
    # returns the current inputted condition value for the desired trackbar
    contrast = cv2.getTrackbarPos('Contrast', 'Edited')
    brightness = cv2.getTrackbarPos('Brightness', 'Edited')
    # generates the new frame with the effects applied
    effect = editFrame(frame, brightness, contrast)
    # Displays the new edited frame in a new window
    cv2.imshow('Edited', effect)
    # sets the edited frame to a global for use below in the main program
    global editedFrame
    editedFrame = effect


# this function takes the values for brightness and contrast and edits each frame passed
def editFrame(frame, brightness, contrast):
    # since the track bar starts at 0 and not -255/-127 respectively we have to subtract by those values
    brightness -= 255
    contrast -= 127
    # adjusting the brightness
    if (brightness != 0):
        if (brightness > 0):
            darkness = brightness
            maximum = 255
        else:
            darkness = 0
            maximum = 255 + brightness
        alpha = (maximum - darkness) / 255
        gamma = darkness
        # The function addWeighted weightedSum calculates the weighted sum of two arrays
        weightedSum = cv2.addWeighted(frame, alpha, frame, 0, gamma)
    else:
        weightedSum = frame
    # adjusting the contrast
    if (contrast != 0):
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
        # The function addWeighted weightedSum calculates the weighted sum of two arrays
        weightedSum = cv2.addWeighted(weightedSum, Alpha, weightedSum, 0, Gamma)
    return weightedSum


matplotlib.use("TkAgg")
# Creates 2 windows. 1 for unedited video and 1 for edited video
cv2.namedWindow("Unedited")
cv2.namedWindow("Edited")
# grabs the provided video
vid = cv2.VideoCapture('video.mp4')
# gets total frames (used to fix a thrown error problem in the while loop)
totalFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
# creates a video writer to putput the video
# the last parameter is hard coded from the file properties of the origional video
final = cv2.VideoWriter('final.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 50, (480, 360))
# Displays first frame for preview
ret, frame = vid.read()
cv2.imshow('Unedited', frame)
cv2.imshow('Edited', frame)
# creates a slider bar for brightness and contrast
cv2.createTrackbar('Contrast', 'Edited', 127, 2 * 127, changeBrightCont)
cv2.createTrackbar('Brightness', 'Edited', 255, 2 * 255, changeBrightCont)
# calls the function to grab slider bar values
changeBrightCont(0)
# has program wait 7 seconds before continuing and applying the setting to the final video
cv2.waitKey(5000)
cv2.destroyWindow('Unedited')


# while loop plays the video
while (vid.isOpened()):
    # grabs a single frame
    ret, frame = vid.read()
    # converts the frame to grey scale to work nicely with the equalizeHist() function
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # displays the frame unedited
    changeBrightCont(0)
    # uses the equalizeHist() function to edit the frame
    editedFrame = cv2.equalizeHist(frame)
    # creates histogram
    histyBoy = cv2.calcHist([frame], [0], None, [256], [0, 256])
    plt.hist(histyBoy)
    plt.imshow(histyBoy, cmap='gray', vmin=0, vmax=255)
    plt.ion()
    plt.show()
    # displays edited frame
    cv2.imshow('Edited', editedFrame)
    # end of video handling
    totalFrames -= 1
    if(totalFrames == 1):
        vid.release()
        break
    # writes the frame to the output videowriter
    final.write(editedFrame)
    if ((cv2.waitKey(20) and 0xFF == ord('q'))):
        break


# releases and closes the program
final.release()
cv2.destroyAllWindows()
