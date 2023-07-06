import cv2
import pygetwindow as gw
import pyautogui
import random
import numpy as np
import keyboard
import time
import glob

# Define the threshold for template matching
threshold = 0.7
movemet_threshold = 16
conditioning_time = 22  # 22 seconds

# Global Variables
top_left_g = None
bottom_right_g = None
previous_frame = None
found = False
start_time = time.time()


# Get all PNG image file names in the directory
template_filenames = glob.glob("./samples/*.png")

# Load the template images
template_images = []
for filename in template_filenames:
    template = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if template is not None:
        template_images.append(template)


# Get the program window by its title
window_title = "World of Warcraft"
program_window = gw.getWindowsWithTitle(window_title)[0]

window_x, window_y, window_width, window_height = (
    program_window.left,
    program_window.top,
    program_window.width,
    program_window.height,
)


def reset():
    global previous_frame, found, start_time
    delay_time = random.uniform(0.15, 0.2)
    time.sleep(2 * delay_time)
    # fish action on button 1
    keyboard.press_and_release("1")
    previous_frame = None
    found = False
    start_time = time.time()


def bobber_finder(gray_frame):
    global found, top_left_g, bottom_right_g
    if not found:
        for template in template_images:
            # Perform template matching
            result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val >= threshold:
                print("bobber found")
                top_left = max_loc
                bottom_right = (
                    top_left[0] + template.shape[1],
                    top_left[1] + template.shape[0],
                )
                found = True
                top_left_g, bottom_right_g = expand_crop_area(
                    top_left, bottom_right, 0.1
                )

    return top_left_g, bottom_right_g


def detect_movement(current_frame):
    global previous_frame, movemet_threshold
    if previous_frame is None:
        previous_frame = current_frame
        return False

    frame_diff = cv2.absdiff(current_frame, previous_frame)
    _, frame_diff_thresholded = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    movement_pixels = cv2.countNonZero(frame_diff_thresholded)
    movement_percentage = (movement_pixels / (frame_diff_thresholded.size + 1)) * 100
    print(movement_percentage)

    return movement_percentage >= movemet_threshold


def expand_crop_area(top_left, bottom_right, expansion_percentage):
    # Calculate the dimensions of the current crop area
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]

    # Calculate the expansion amounts for width and height
    expand_width = int(width * expansion_percentage)
    expand_height = int(height * expansion_percentage)

    # Expand the crop area by the calculated amounts
    expanded_top_left = (top_left[0] - expand_width, top_left[1] - expand_height)
    expanded_bottom_right = (
        bottom_right[0] + expand_width,
        bottom_right[1] + expand_height,
    )

    return expanded_top_left, expanded_bottom_right


def bobber_moves(gray_frame, top_left, bottom_right):
    global found
    if found:
        # Define the region of interest within the expanded rectangle
        roi_frame = gray_frame[
            top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]
        ]
        # time.sleep(0.01)
        if detect_movement(roi_frame):
            # cv2.imwrite("rectangle_capture" + "" + ".png", roi_frame) save capture
            return True

    return False


while True:
    # Capture the program window image
    img = pyautogui.screenshot(region=(window_x, window_y, window_width, window_height))

    # Convert the image to numpy array representation
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    top_left, bottom_right = bobber_finder(gray_frame)

    if bobber_moves(gray_frame, top_left, bottom_right):
        print(f">>> Fish Found")
        keyboard.press_and_release("f")
        reset()

    # Visual debugger
    # cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    # Display the captured frame
    # cv2.imshow("Program Capture", frame)

    if time.time() - start_time >= conditioning_time:
        print("reset")
        reset()

    # Check for the 'q' key to quit the program
    if cv2.waitKey(1) == ord("q"):
        break

# Close the OpenCV windows
cv2.destroyAllWindows()
