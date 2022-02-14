import cv2

global editedWin

# this function takes the inputted values for contrast and brightness
# then that data gets sent to editFrame to apply the edit
def changeBrightCont(brightness):
    # returns the current inputted condition value for the desired trackbar
    contrast = cv2.getTrackbarPos('Contrast', 'Frame')
    brightness = cv2.getTrackbarPos('Brightness', 'Frame')
    # generates the new frame with the effects applied
    effect = editFrame(frame, brightness, contrast)
    # Displays the new edited frame in a new window
    cv2.imshow('Final Edit', effect)
    # sets the edited frame to a global for use below in the main program
    global editedWin
    editedWin = effect

# this function takes the vlaues for brightness and contrast and edits each frame passed
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
        # The function addWeighted weightedSumculates the weighted sum of two arrays
        weightedSum = cv2.addWeighted(frame, alpha, frame, 0, gamma)
    else:
        weightedSum = frame
    # adjusting the contrast
    if (contrast != 0):
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
        # The function addWeighted weightedSumculates the weighted sum of two arrays
        weightedSum = cv2.addWeighted(weightedSum, Alpha, weightedSum, 0, Gamma)
    return weightedSum


# loads video into program
vid = cv2.VideoCapture("video.mp4")
# names the window of the first frame of the video
cv2.namedWindow('Frame')
# gets the first frame
ret, frame = vid.read()
# displays frame in window
cv2.imshow('Frame', frame)
# creates a slider bar for brightness and contrast
cv2.createTrackbar('Contrast', 'Frame', 127, 2 * 127, changeBrightCont)
cv2.createTrackbar('Brightness', 'Frame', 255, 2 * 255, changeBrightCont)
# calls the function to grab slider bar values
changeBrightCont(0)
# has program wait 7 seconds before continuing and applying the setting to the final video
cv2.waitKey(7000)

# plays video frame by frame
while (vid.isOpened()):
    ret, frame = vid.read()
    if (ret == True):
        cv2.imshow('Final Edit', editedWin)
        changeBrightCont(0)
        if (cv2.waitKey(25) & 0xFF == ord('q')):
            break
    else:
        break
# releases video and closes all windows
vid.release()
cv2.destroyAllWindows()
