# Ryan Sobolewski
# CAP 4410
# Assignment 2

import cv2
import numpy
import math


# This function creates a new window that contains the blurred box filter version of the image
# using opencv's built in box filter function
def boxFilterCV2(img, str):
    cv2.namedWindow(str)
    blurImg = cv2.boxFilter(img, -1, (5, 5))
    cv2.imshow(str, img)
    cv2.imshow(str, blurImg)


# This function implements a box filter for a 3x3 window by doing the actual array math
def boxFilterHardWay3x3(img, str):
    # Splits image into its red, green, and blue channel arrays respectively
    rImg, gImg, bImg = numpy.dsplit(img, 3)
    # Loops through every pixel value to calculate the average of every 3x3 square of pixels
    # This nested for loop goes through the red channel
    sizeX = rImg.shape[0]
    sizeY = rImg.shape[1]
    rBlur = []
    for x in range(1, sizeX - 1):
        rBlurCol = []
        for y in range(1, sizeY - 1):
            rBlurCol.append(math.floor((rImg[x - 1][y - 1] + rImg[x][y - 1] + rImg[x + 1][y - 1] +
                                        rImg[x - 1][y] + rImg[x][y] + rImg[x + 1][y] +
                                        rImg[x - 1][y + 1] + rImg[x][y + 1] + rImg[x + 1][y + 1]) / 9))
        rBlur.append(rBlurCol)
    # Loops through every pixel value to calculate the average of every 3x3 square of pixels
    # This nested for loop goes through the green channel
    sizeX = gImg.shape[0]
    sizeY = gImg.shape[1]
    gBlur = []
    for x in range(1, sizeX - 1):
        gBlurCol = []
        for y in range(1, sizeY - 1):
            gBlurCol.append(math.floor((gImg[x - 1][y - 1] + gImg[x][y - 1] + gImg[x + 1][y - 1] +
                                        gImg[x - 1][y] + gImg[x][y] + gImg[x + 1][y] +
                                        gImg[x - 1][y + 1] + gImg[x][y + 1] + gImg[x + 1][y + 1]) / 9))
        gBlur.append(gBlurCol)
    # Loops through every pixel value to calculate the average of every 3x3 square of pixels
    # This nested for loop goes through the blue channel
    sizeX = bImg.shape[0]
    sizeY = bImg.shape[1]
    bBlur = []
    for x in range(1, sizeX - 1):
        bBlurCol = []
        for y in range(1, sizeY - 1):
            bBlurCol.append(math.floor((bImg[x - 1][y - 1] + bImg[x][y - 1] + bImg[x + 1][y - 1] +
                                        bImg[x - 1][y] + bImg[x][y] + bImg[x + 1][y] +
                                        bImg[x - 1][y + 1] + bImg[x][y + 1] + bImg[x + 1][y + 1]) / 9))
        bBlur.append(bBlurCol)
    # Brings all 3 separate arrays together by concatenation
    blurFinal = numpy.concatenate((rImg, gImg, bImg), axis=2)
    cv2.namedWindow(str)
    cv2.imshow(str, blurFinal)


# This function implements a box filter for a 5x5 window by doing the actual array math
def boxFilterHardWay5x5(img, str):
    # Splits image into its red, green, and blue channel arrays respectively
    rImg, gImg, bImg = numpy.dsplit(img, 3)
    # Loops through every pixel value to calculate the average of every 5x5 square of pixels
    # This nested for loop goes through the red channel
    sizeX = rImg.shape[0]
    sizeY = rImg.shape[1]
    rBlur = []
    for x in range(2, sizeX - 2):
        rBlurCol = []
        for y in range(2, sizeY - 2):
            rBlurCol.append(math.floor(
                (rImg[x - 2][y - 2] + rImg[x - 1][y - 2] + rImg[x][y - 2] + rImg[x + 1][y - 2] + rImg[x + 2][y - 2] +
                 rImg[x - 2][y - 1] + rImg[x - 1][y - 1] + rImg[x][y - 1] + rImg[x + 1][y - 1] + rImg[x + 2][y - 1] +
                 rImg[x - 2][y] + rImg[x - 1][y] + rImg[x][y] + rImg[x + 1][y] + rImg[x + 2][y] +
                 rImg[x - 2][y + 1] + rImg[x - 1][y + 1] + rImg[x][y + 1] + rImg[x + 1][y + 1] + rImg[x + 2][y + 1] +
                 rImg[x - 2][y + 2] + rImg[x - 1][y + 2] + rImg[x][y + 2] + rImg[x + 1][y + 2] + rImg[x + 2][y + 2]) / 25))
        rBlur.append(rBlurCol)
    # Loops through every pixel value to calculate the average of every 5x5 square of pixels
    # This nested for loop goes through the green channel
    sizeX = gImg.shape[0]
    sizeY = gImg.shape[1]
    gBlur = []
    for x in range(2, sizeX - 2):
        gBlurCol = []
        for y in range(2, sizeY - 2):
            gBlurCol.append(math.floor(
                (gImg[x - 2][y - 2] + gImg[x - 1][y - 2] + gImg[x][y - 2] + gImg[x + 1][y - 2] + gImg[x + 2][y - 2] +
                 gImg[x - 2][y - 1] + gImg[x - 1][y - 1] + gImg[x][y - 1] + gImg[x + 1][y - 1] + gImg[x + 2][y - 1] +
                 gImg[x - 2][y] + gImg[x - 1][y] + gImg[x][y] + gImg[x + 1][y] + gImg[x + 2][y] +
                 gImg[x - 2][y + 1] + gImg[x - 1][y + 1] + gImg[x][y + 1] + gImg[x + 1][y + 1] + gImg[x + 2][y + 1] +
                 gImg[x - 2][y + 2] + gImg[x - 1][y + 2] + gImg[x][y + 2] + gImg[x + 1][y + 2] + gImg[x + 2][y + 2]) / 25))
        gBlur.append(gBlurCol)
    # Loops through every pixel value to calculate the average of every 5x5 square of pixels
    # This nested for loop goes through the blue channel
    sizeX = bImg.shape[0]
    sizeY = bImg.shape[1]
    bBlur = []
    for x in range(2, sizeX - 2):
        bBlurCol = []
        for y in range(2, sizeY - 2):
            bBlurCol.append(math.floor(
                (bImg[x - 2][y - 2] + bImg[x - 1][y - 2] + bImg[x][y - 2] + bImg[x + 1][y - 2] + bImg[x + 2][y - 2] +
                 bImg[x - 2][y - 1] + bImg[x - 1][y - 1] + bImg[x][y - 1] + bImg[x + 1][y - 1] + bImg[x + 2][y - 1] +
                 bImg[x - 2][y] + bImg[x - 1][y] + bImg[x][y] + bImg[x + 1][y] + bImg[x + 2][y] +
                 bImg[x - 2][y + 1] + bImg[x - 1][y + 1] + bImg[x][y + 1] + bImg[x + 1][y + 1] + bImg[x + 2][y + 1] +
                 bImg[x - 2][y + 2] + bImg[x - 1][y + 2] + bImg[x][y + 2] + bImg[x + 1][y + 2] + bImg[x + 2][y + 2]) / 25))
        bBlur.append(bBlurCol)
    # Brings all 3 separate arrays together by concatenation
    blurFinal = numpy.concatenate((rImg, gImg, bImg), axis=2)
    cv2.namedWindow(str)
    cv2.imshow(str, blurFinal)


# Sobel Edge detection X axis
def sobelX(img, str):
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelArr = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filteredSobel = cv2.filter2D(src=imgGrey, ddepth=-1, kernel=sobelArr)
    cv2.imshow(str, filteredSobel)


# Sobel Edge detection Y axis
def sobelY(img, str):
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelArr = numpy.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    filteredSobel = cv2.filter2D(src=imgGrey, ddepth=-1, kernel=sobelArr)
    cv2.imshow(str, filteredSobel)

# Sobel Edge detection on both X and Y axis
def sobelXYHardWay(img, str):
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelArrX = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelArrY = numpy.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    filteredSobel = cv2.filter2D(src=imgGrey, ddepth=-1, kernel=sobelArrX)
    filteredSobelFinal = cv2.filter2D(src=filteredSobel, ddepth=-1, kernel=sobelArrY)
    cv2.imshow(str, filteredSobelFinal)

# Sobel Edge detection in X and Y axis using opencv
def sobelXY(img, str):
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filteredSobel = cv2.Sobel(src=imgGrey, ddepth=-1, dx=1, dy=1, ksize=5)
    cv2.imshow(str, filteredSobel)

# Gaussian Filter in opencv
def gaus(img, str):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imshow(str, blur)

# Main function
def main():
    # Import files into program
    grey = cv2.imread("grey.png")
    doggo = cv2.imread("doggo.bmp")
    bike = cv2.imread("bike.bmp")
    # Display the non-edited version of each image
    cv2.namedWindow("Grey OG")
    cv2.imshow("Grey OG", grey)
    cv2.namedWindow("Dog OG")
    cv2.imshow("Dog OG", doggo)
    cv2.namedWindow("Bike OG")
    cv2.imshow("Bike OG", bike)
    # Produces box filter version
    boxFilterCV2(grey, "Grey Blur")
    boxFilterCV2(doggo, "Dog Blur")
    boxFilterCV2(bike, "Bike Blur")
    # Produces box filter from doing the math
    boxFilterHardWay3x3(grey, "Grey Blur 3x3 Math")
    boxFilterHardWay3x3(doggo, "Dog Blur 3x3 Math")
    boxFilterHardWay3x3(bike, "Bike Blur 3x3 Math")
    boxFilterHardWay5x5(grey, "Grey Blur 5x5 Math")
    boxFilterHardWay5x5(doggo, "Dog Blur 5x5 Math")
    boxFilterHardWay5x5(bike, "Bike Blur 5x5 Math")
    # Produces Sobel edge detection on X axis
    sobelX(grey, "Grey Sobel X")
    sobelX(doggo, "Dog Sobel X")
    sobelX(bike, "Bike Sobel X")
    # Produces Sobel edge detection on Y axis
    sobelY(grey, "Grey Sobel Y")
    sobelY(doggo, "Dog Sobel Y")
    sobelY(bike, "Bike Sobel Y")
    # Produces Sobel edge detection in X and Y axis
    sobelXYHardWay(grey, "Grey Sobel XY Math")
    sobelXYHardWay(doggo, "Dog Sobel XY Math")
    sobelXYHardWay(bike, "Bike Sobel XY Math")
    # Produces Sobel edge detector in X and Y axis using opencv
    sobelXY(grey, "Grey Sobel XY opencv")
    sobelXY(doggo, "Dog Sobel XY opencv")
    sobelXY(bike, "Bike Sobel XY opencv")
    gaus(grey, "Grey Gaussian")
    gaus(doggo, "Dog Gaussian")
    gaus(bike, "Bike Gaussian")
    # Kill program
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()
