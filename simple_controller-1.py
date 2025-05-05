"""camera_pid controller."""

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import math


#Getting image from camera
def get_image(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

#Image processing
def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

def blur_cv2(image):
    blr_img = cv2.GaussianBlur(image, (5,5), 0)
    return blr_img

def canny_cv2(image):
    canny_img = cv2.Canny(image, threshold1=10, threshold2=20)
    return canny_img

def roi_cv2(image):
    print(image.shape)

    ver = np.array([[(0,128),(30,90),(192,90),(256,128)]], dtype=np.int32)
    
    # 128, 256
    
    roi_img = np.zeros_like(image)
    cv2.fillPoly(roi_img, ver, 255)
    roi_img = cv2.bitwise_and(image,roi_img)
    return roi_img

def average_line(lines):
    x1s, y1s, x2s, y2s = zip(*lines)
    return (
        int(np.mean(x1s)),
        int(np.mean(y1s)),
        int(np.mean(x2s)),
        int(np.mean(y2s))
    )

"""
Function `hough_cv2`: Detects lines in an image using the probabilistic Hough transform (cv2.HoughLinesP).
The main goal is to identify sloped (non-horizontal) lines that represent lane edges for navigation,
and compute a centerline between them to determine an automatic steering angle.

Key steps of the algorithm:
1. The Hough transform is applied with tuned parameters to detect relevant lines in the image mask.
2. Horizontal lines (near-zero slope) are discarded to avoid false positives.
3. The total number of detected lines is limited: if more than 20 are found, all lines are ignored to reduce noise.
4. Lines are classified as left or right depending on their horizontal position (less than or greater than 128 px).
5. An average line is computed for each group (left and right).
6. A centerline is generated from these averages, and an error is calculated relative to the image center.
7. The steering angle is adjusted proportionally to this error using a defined gain (Kp).
8. If no valid lines are detected or there are too many lines, the steering remains straight.

The result is an image with the lane lines overlaid and the steering angle adjusted accordingly.
"""
def hough_cv2(image,img_mask):
    img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    rho = 3
    theta = np.pi/180
    threshold = 10
    min_line_len = 20
    max_line_gap = 30
    lines = cv2.HoughLinesP(img_mask, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    img_lines = np.zeros((img_mask.shape[0], img_mask.shape[1], 3), dtype=np.uint8)

    left_lines = []
    right_lines = []
    if lines is not None and len(lines)<20:
        
        print(f"len lines: {len(lines)}")
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 - x1 == 0:
                    continue  
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.2:
                    continue  
                if x1 < 128 and x2 < 128:
                    left_lines.append((x1, y1, x2, y2))
                    cv2.line(img_lines, (x1, y1), (x2, y2), (255, 0, 0), 4)
                elif x1 > 128 and x2 > 128:
                    right_lines.append((x1, y1, x2, y2))
                    cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 4)

        if left_lines and right_lines:
            avg_left = average_line(left_lines)
            avg_right = average_line(right_lines)

            mid_x1 = int((avg_left[0] + avg_right[0]) / 2)
            mid_y1 = int((avg_left[1] + avg_right[1]) / 2)
            mid_x2 = int((avg_left[2] + avg_right[2]) / 2)
            mid_y2 = int((avg_left[3] + avg_right[3]) / 2)

            cv2.line(img_lines, (mid_x1, mid_y1), (mid_x2, mid_y2), (0, 0, 255), 6)

            img_center_x = image.shape[1] // 2
            lane_center_x = (mid_x1 + 10)
            error = img_center_x - lane_center_x
            Kp = 0.005
            auto_steering = Kp * error
            set_steering_angle(-auto_steering)
            print(f"[INFO] error: {error}, steering: {auto_steering:.3f}")
    else:
        set_steering_angle(0)       
        print("no line detected go straight")

    alpha = 1
    beta = 1
    gamma = 1
    img_lane_lines = cv2.addWeighted(img_rgb, alpha, img_lines, beta, gamma)
    
    return img_lane_lines

#Display image 
def display_image(display, image):
    # Image to display
    # image_rgb = np.dstack((image, image,image,))

    # Si la imagen es de un solo canal (escala de grises)
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Unsupported image format for display.")
    
    # Display image
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

#initial angle and speed 
manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 15

# set target speed
def set_speed(kmh):
    global speed            #robot.step(50)
#update steering angle
def set_steering_angle(wheel_angle):
    global angle, steering_angle
    # Check limits of steering
    if (wheel_angle - steering_angle) > 0.1:
        wheel_angle = steering_angle + 0.1
    if (wheel_angle - steering_angle) < -0.1:
        wheel_angle = steering_angle - 0.1
    steering_angle = wheel_angle
  
    # limit range of the steering angle
    if wheel_angle > 0.5:
        wheel_angle = 0.5
    elif wheel_angle < -0.5:
        wheel_angle = -0.5
    # update steering angle
    angle = wheel_angle

#validate increment of steering angle
def change_steer_angle(inc):
    global manual_steering
    # Apply increment
    new_manual_steering = manual_steering + inc
    # Validate interval 
    if new_manual_steering <= 25.0 and new_manual_steering >= -25.0: 
        manual_steering = new_manual_steering
        set_steering_angle(manual_steering * 0.02)
    # Debugging
    if manual_steering == 0:
        print("going straight")
    else:
        turn = "left" if steering_angle < 0 else "right"
        print("turning {} rad {}".format(str(steering_angle),turn))

# main
def main():

    first = True

    # Create the Robot instance.
    robot = Car()
    driver = Driver()

    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # Create camera instance
    camera = robot.getDevice("camera")
    camera.enable(timestep)  # timestep

    # processing display
    display_img_1 = Display("display_image_1")
    display_img_2 = Display("display_image_2")

    #create keyboard instance
    keyboard=Keyboard()
    keyboard.enable(timestep)

    while robot.step() != -1:
        # Get image from camera
        image = get_image(camera)

        # Process and display image 
        grey_image = greyscale_cv2(image)
        blr_image = blur_cv2(grey_image)
        canny_image = canny_cv2(blr_image)
        roi_image = roi_cv2(canny_image)
        hough_image = hough_cv2(image, roi_image)

        display_image(display_img_1, roi_image)
        display_image(display_img_2, hough_image)




        # Read keyboard
        key=keyboard.getKey()
        if key == keyboard.UP: #up
            set_speed(speed + 5.0)
            print("up")
        elif key == keyboard.DOWN: #down
            set_speed(speed - 5.0)
            print("down")
        elif key == keyboard.RIGHT: #right
            change_steer_angle(+1)
            print("right")
        elif key == keyboard.LEFT: #left
            change_steer_angle(-1)
            print("left")
        elif key == ord('A'):
            #filename with timestamp and saved in current directory
            current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
            file_name = current_datetime + ".png"
            print("Image taken")
            camera.saveImage(os.getcwd() + "/" + file_name, 1)
            
        #update angle and speed
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)


if __name__ == "__main__":
    main()
