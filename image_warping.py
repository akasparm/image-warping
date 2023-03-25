"""
ENPM673: PERCEPTION FOR AUTONOMOUS ROBOTS
SUBMITTED BY: AKASHKUMAR PARMAR
"""

# Importing required libraries
import numpy as np
import cv2 as cv
# Function to solve the equation to transform from Hough parameters to Cartesian parameters
def find_soln(rho, theta, temp_img):
    a = np.cos(theta)
    b = np.sin(theta)
    x1 = int(a*rho + 1000*(-b))
    y1 = int(b*rho + 1000*(a))
    x2 = int(a*rho - 1000*(-b))
    y2 = int(b*rho - 1000*(a))
    cv.line(temp_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return temp_img

def train_track():
        
    bgr_img = cv.imread("train_track.jpg") # Read the image

    # Size of the image
    height = bgr_img.shape[0] 
    width = bgr_img.shape[1]

    # track and Corresponding image points 
    track_coord = np.float32([[880, 2000], [1397, 1156], [1622, 1156], [2150, 2000]])
    image_coord = np.float32([[0, height], [0, 0], [width, 0], [width, height]])

    H = cv.getPerspectiveTransform(track_coord, image_coord) # Homogeneous Transformation matrix

    warped = cv.warpPerspective(bgr_img, H, (width, height)) # Image warping
    paralleled_img = cv.resize(warped, (480, 720)) # Resized image
    cv.imshow("Warped image", paralleled_img) # intermediate result

    gray_img = cv.cvtColor(paralleled_img, cv.COLOR_BGR2GRAY) # Gray scaling
    cv.imshow("Gray scale", gray_img) # intermediate result
    blurred_img = cv.GaussianBlur(gray_img, (19, 19), 1) # Blurring the image
    edged_img = cv.Canny(blurred_img, 10, 300, apertureSize=3) # Applying canny edge filter
    cv.imshow("Canny edges", edged_img) # intermediate result

    # Line detection
    hough_lines = cv.HoughLines(edged_img, np.pi/2, np.pi/180, 350)
    for line in hough_lines:
        rho, theta = line[0]
        lined_img = find_soln(rho, theta, paralleled_img)

    dist_list = [] # distance list to store the values of the distances for each row

    for row in range(720):
        count = 0
        for col in range(480):
            if (lined_img[row][col] == (0, 0, 255)).all():
                if count%12 == 0:
                    x1 = col
                    count += 1
                elif count%12 == 1:
                    x2 = col
                    count += 1
                elif count%12 == 2:
                    x3 = col
                    count += 1
                elif count%12 == 3:
                    x4 = col
                    count += 1
                elif count%12 == 4:
                    x5 = col
                    count += 1
                elif count%12 == 5:
                    x6 = col
                    count += 1
                elif count%12 == 6:
                    x7 = col
                    count += 1
                elif count%12 == 7:
                    x8 = col
                    count += 1
                elif count%12 == 8:
                    x9 = col
                    count += 1
                elif count%12 == 9:
                    x10 = col
                    count += 1
                elif count%12 == 10:
                    x11 = col
                    count += 1
                else:
                    x12 = col
                    count += 1

        dist_list.append((x7+x8+x9+x10+x11+x12)-(x1+x2+x3+x4+x5+x6))

    avg = np.average(dist_list)/6 # average of the distance list

    print("The average distance between the tracks is: ", avg, " pixels")

    cv.imshow("Lined Image", lined_img)
    cv.waitKey(0)
    cv.destroyAllWindows()



def main():
    train_track()


if __name__=="__main__":
    main()