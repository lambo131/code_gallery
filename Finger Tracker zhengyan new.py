import cv2
import numpy as np
import os
import sys
import math

# Set the maximum window size to fit your screen
max_width = 720
max_height = 720
FRAMES_PER_PROCESSED_FRAME = 4

CIRCULARITY_THRESHOLD = 0.2
MIN_CONTOUR_AREA = 150
TRACKER_CIRCLE_RADIUS = 80  # Radius of the bounding circle that is a moving avg of cX and cY
CATCH_UP_SPEED = 50
LOST_TRACK_FRAME_COUNT = 12
ALPHA = 0.05
# BETA = 0.9

try:
    video_num = int(sys.argv[1])
    coordinate_num = video_num
    record_session_id = sys.argv[2]
    print(f"\n\n>>> this is a subprocess...")
     
except:
    # select video to be processed
    video_num = 4
    # select output dataset name dataset 7 .eg
    coordinate_num = video_num
    record_session_id = "Mar 25"



video_dir = f"./data/videos"
video_path = f"{video_dir}/{video_num}.mov"
print(f"\n>>> Analyzing file: {video_path}\n")

# Color detection parameters
START_FRAME = 1300
ENABLE_TOOL = False  
TEST_MODE = True

# LED color mask filter
lower_blue = np.array([0, 0, 253])
upper_blue = np.array([100, 20, 255])

# Pre-allocate buffers and kernels
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
gaussian_kernel = (9, 9)

# Initialize variables for frame processing
average_coordinates_list = []
frame_count = 0
no_acuro_frame_count = 0
use_prev_aucro = False

# moving average of the coordinates of the contour
lost_track_frames = LOST_TRACK_FRAME_COUNT + 1
lost_track = True # defined as when nothing is in the tracker circle for lost_track_frame frames
avg_cX = 0
avg_cY = 0
last_cX = -1
last_cY = -1
pred_x = 0
pred_y = 0
vX = 0
vY = 0
aX = 0
aY = 0

# Initialize ArUco detector
dict_type = cv2.aruco.DICT_ARUCO_ORIGINAL
dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(dictionary, aruco_params)
# arUco constants
marker_size_cm = 3.0
half_size = marker_size_cm / 2.0
exclude_size = 4.0
exclude_half = exclude_size / 2.0
# Pre-allocate arrays
exclude_local = np.array([
    [-exclude_half, -exclude_half],
    [-exclude_half, exclude_half],
    [exclude_half, exclude_half],
    [exclude_half, -exclude_half]
], dtype=np.float32).reshape(-1, 1, 2)
dst_points = np.array([
    [-half_size, half_size],
    [half_size, half_size],
    [half_size, -half_size],
    [-half_size, -half_size]
], dtype=np.float32)

# Enable OpenCL acceleration
cv2.ocl.setUseOpenCL(True)

# Initialize video capture with hardware acceleration if available
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    cap.release()
    # Try again with different backend
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

# initialize some frame settings
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
scale = min(max_height / width, max_height / height)
reduced_width = int(width * scale)
reduced_height = int(height * scale)

# Create trackbars if enabled
if ENABLE_TOOL:
    cv2.namedWindow('Trackbars')
    cv2.createTrackbar('Hue Min', 'Trackbars', 0, 180, lambda x: None)
    cv2.createTrackbar('Hue Max', 'Trackbars', 180, 180, lambda x: None)
    cv2.createTrackbar('Sat Min', 'Trackbars', 0, 255, lambda x: None)
    cv2.createTrackbar('Sat Max', 'Trackbars', 255, 255, lambda x: None)
    cv2.createTrackbar('Val Min', 'Trackbars', 0, 255, lambda x: None)
    cv2.createTrackbar('Val Max', 'Trackbars', 255, 255, lambda x: None)
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Result', reduced_width, reduced_height)

if TEST_MODE:
    cv2.namedWindow('Processed', cv2.WINDOW_NORMAL)
    cv2.namedWindow('filtered_mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Processed', reduced_width, reduced_height)
    cv2.resizeWindow('filtered_mask', reduced_width, reduced_height)
    cv2.resizeWindow('Contours', reduced_width, reduced_height)


def update_mask(hsv):
    # Get current trackbar positions
    lower_hsv = np.array([cv2.getTrackbarPos('Hue Min', 'Trackbars'),
                          cv2.getTrackbarPos('Sat Min', 'Trackbars'),
                          cv2.getTrackbarPos('Val Min', 'Trackbars')])
    upper_hsv = np.array([cv2.getTrackbarPos('Hue Max', 'Trackbars'),
                          cv2.getTrackbarPos('Sat Max', 'Trackbars'),
                          cv2.getTrackbarPos('Val Max', 'Trackbars')])
    # Create a mask using the HSV range
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = np.uint8(mask)
    
    # Resize the mask if needed to match the frame size
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    
    # Apply the mask to the original frame
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Display the mask and result
    cv2.imshow('Result', result)

def process_frame(frame):
    # Resize frame to processing resolution
    frame_small = cv2.resize(frame, (reduced_width, reduced_height))
    
    # Convert to HSV and apply Gaussian blur
    hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, gaussian_kernel, 0)
    
    # Create mask for blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
    
    # Morphological operations
    filtered_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)
    
    return frame_small, hsv, filtered_mask

if START_FRAME > 0:
    print(f">>> skipping to frame: {START_FRAME}...\n") 
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
# Main processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % FRAMES_PER_PROCESSED_FRAME != 1:
        continue
    
    # get hsv and filtered mask
    frame_small, hsv, filtered_mask = process_frame(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, gaussian_kernel, 0)
    
    # Detect ArUco markers
    corners, ids, rejected = aruco_detector.detectMarkers(gray)
    
    # Process markers and create exclude polygons
    exclude_polygons = []
    if ids is not None:
        for i in range(len(ids)):
            c = corners[i][0]
            src_points = np.array(c, dtype=np.float32)
            transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            inv_transform_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
            
            # Transform exclude area to image coordinates
            exclude_image = cv2.perspectiveTransform(exclude_local, inv_transform_matrix)
            exclude_polygon = exclude_image.reshape(-1, 2).astype(np.int32)
            exclude_polygons.append(exclude_polygon)
        prev_corners = corners
    # ----------------------------------------------------------------------------------------------------- LED filter
    # Find contours and filter out small contours
    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA * scale]
    # Sort the contours based on area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # Extract the contours of the three largest areas or all contours if less than three
    if len(contours) >= 3:
        largest_contours = contours[:3]
    else:
        largest_contours = contours

    #print(cv2.contourArea(largest_contours[0]))
    
    # Process valid contours ------------------------------------------------------------------------
    cX_arr = [] # a buffer arr to store and process the contours at the end after analysing all contours
    cY_arr = []
    distance_arr = []
    distance_to_pred_arr = [] # the distance to the predicted next (this) point based on the previous velocities
    in_circle = False # are there any objects still in the circle?  Set to false initially
    total_x, total_y, count = 0, 0, 0
    valid_contours = []
    
    for contour in largest_contours:
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)

        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < CIRCULARITY_THRESHOLD:
            continue

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"]) / scale
            cY = int(M["m01"] / M["m00"]) / scale
            # check if contour is in the moving average circle
            circle_center = (int(avg_cX), int(avg_cY))
            pred_circle_center = (int(pred_x), int(pred_y))
            # Point coordinates to test
            point_to_test = (cX, cY)  # Replace x and y with the coordinates of the point you want to test
            # Calculate the distance between the point and the circle center
            distance = math.sqrt((point_to_test[0] - circle_center[0])**2 + (point_to_test[1] - circle_center[1])**2)
            distance_to_pred_point = math.sqrt((point_to_test[0] - pred_circle_center[0])**2 + (point_to_test[1] - pred_circle_center[1])**2)
            cX_arr.append(cX)
            cY_arr.append(cY)
            distance_arr.append(distance)
            distance_to_pred_arr.append(distance_to_pred_point)

    for i, dist in enumerate(distance_arr):
        if dist <= TRACKER_CIRCLE_RADIUS:
            lost_track_frames = 0
            in_circle = True
            break

    if not in_circle:
        print("    nothing in tracker")
        lost_track_frames+=FRAMES_PER_PROCESSED_FRAME
    if lost_track_frames >= LOST_TRACK_FRAME_COUNT:
        lost_track = True
    else:
        lost_track = False
            
    for i, dist in enumerate(distance_arr):
        cX_temp = cX_arr[i]
        cY_temp = cY_arr[i]
        # Check if in exclude area
        exclude = False
        for poly in exclude_polygons:
            if cv2.pointPolygonTest(poly, (cX_temp, cY_temp), False) >= 0:
                exclude = True
                break
        if exclude:
            continue
        
        elif distance_to_pred_arr[i] > TRACKER_CIRCLE_RADIUS*0.8: # use distance to predicted point for diciding if a point is outside of the tracking circle
            print(f"    contour outside circle... d={dist}")
            if not lost_track: 
                print(f"    still tracking, ignoring outsiders... d={dist}")
                continue # only consider contours outside of circle if the circle has lost track

            # if a contour is outside of the tracking circle, then use the real distance between the current tracker point and the contour point
            # a virtual contour point with a fixed euclidian distance outside of the circle for updating the filter circle's pos 
            cX_temp = avg_cX+(cX_temp-avg_cX)/dist*CATCH_UP_SPEED 
            cY_temp = avg_cY+(cY_temp-avg_cY)/dist*CATCH_UP_SPEED
            #avg_cX = BETA*avg_cX + (1-BETA)*cX_virtual
            #avg_cY = BETA*avg_cY + (1-BETA)*cY_virtual
            
        total_x += cX_temp
        total_y += cY_temp
        count += 1
        valid_contours.append((avg_cX, avg_cY))

    # updating the tracking circle
    if count > 0:
        cX_update = total_x // count
        cY_update = total_y // count
        avg_cX = int(ALPHA*avg_cX + (1-ALPHA)*cX_update)
        avg_cY = int(ALPHA*avg_cY + (1-ALPHA)*cY_update)

    
    # calculating velocity and predicted next point ------------------------------------------------------
    if last_cX != -1 and last_cY != -1:
        vX = avg_cX - last_cX
        vY = avg_cY - last_cY
    last_cX = avg_cX
    last_cY = avg_cY
    pred_x = avg_cX + vX
    pred_y = avg_cY + vY
    image_with_line = cv2.line(frame, (avg_cX, avg_cY), (pred_x, pred_y), (0, 0, 255), 4)

   
    # draw the bounding circle
    cv2.circle(frame, (int(avg_cX), int(avg_cY)), TRACKER_CIRCLE_RADIUS, (0, 255, 0), 3)
    cv2.circle(frame, (int(pred_x), int(pred_y)), TRACKER_CIRCLE_RADIUS, (0, 255, 255), 2)

    # Calculate and transform coordinates ---------------------------------------------------------------
    avg_x = avg_cX
    avg_y = avg_cY
    

    if (ids is not None and len(ids) > 0) or use_prev_aucro:
        if use_prev_aucro:
            corners = prev_corners
        c = corners[0][0]
        src_points = np.array(c, dtype=np.float32)
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Transform coordinates
        avg_homo = np.array([[avg_x, avg_y, 1]], dtype=np.float32).T
        local_coords = transform_matrix @ avg_homo
        local_coords /= local_coords[2]
        local_x, local_y = local_coords[0][0], local_coords[1][0]
        
        if not lost_track:
            # Draw finger coordinates
            cv2.circle(frame, (avg_x, avg_y), 10, (0, 255, 0), -1)
            cv2.putText(frame, f"({local_x:.2f}, {local_y:.2f} cm)",
                        (avg_x + 10, avg_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            # Store results
            average_coordinates_list.append((frame_count, local_x, local_y))
        else:
            cv2.circle(frame, (avg_x, avg_y), 10, (0, 0, 255), -1)
            cv2.putText(frame, f"({local_x:.2f}, {local_y:.2f} cm)",
                        (avg_x + 10, avg_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            print("    lost track...")
            average_coordinates_list.append((frame_count, 'null', 'null'))    

        if ids is not None and len(ids) > 0:
            use_prev_aucro = True
            no_acuro_frame_count = 0
        else:
            no_acuro_frame_count>=1
            if no_acuro_frame_count > 120:
                use_prev_aucro = False
    else:
        print("    no ids...")
        average_coordinates_list.append((frame_count, 'null', 'null'))
    

    # Draw exclude areas
    for poly in exclude_polygons:
        cv2.polylines(frame, [poly], True, (0, 0, 255), 2)

    if ENABLE_TOOL:
        update_mask(hsv)
    
    if TEST_MODE:
        # Create contour imageq
        contour_img = np.zeros_like(frame_small)
        cv2.drawContours(contour_img, largest_contours, -1, (0, 255, 0), 3)
        # Show all windows
        cv2.imshow("Processed", frame)
        # cv2.imshow('hsv', hsv)
        cv2.imshow('Contours', contour_img)
        # cv2.imshow('mask', mask)
        cv2.imshow('filtered_mask', filtered_mask)
        
    # inputs handling -----------------------------------------------------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'): # stop
        print(f">>> User stop...")
        while True:
            key = cv2.waitKey(1) & 0xFF
            # Resume the loop if the 'r' key is pressed
            if key == ord('r'): # resume
                print(f">>> User resume...")
                break
    elif  key == ord('q'):
        break

    elif frame_count % 10 == 1:
        print(f"progress: {frame_count/total_frames * 100:.1f} %, frame: {frame_count}")


cap.release()
cv2.destroyAllWindows()

# Save results
coordinates_folder = f"data/{record_session_id}/coordinates"
txt_path = f"{coordinates_folder}/{video_num}.txt"
if not os.path.exists(txt_path):
    os.makedirs(coordinates_folder, exist_ok=True)
    with open(txt_path, 'w') as file:
        file.write("Frame\tX (cm)\tY (cm)\n")
        for frame, x, y in average_coordinates_list:
            file.write(f"{frame}\t{x}\t{y}\n")
    print(f">>> coordinates.txt saved to：{txt_path}\n")
else:
    print(f">>> The file '{txt_path}' already exists. Skipping writing.\n")

