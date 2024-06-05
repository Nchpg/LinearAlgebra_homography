import cv2
import numpy as np

path = "receipt.jpg"

def homography(source_points,destination_points):
    A=[]
    for i in range(source_points.shape[0]):
        x,y=source_points[i,0],source_points[i,1]
        xw,yw=destination_points[i,0],destination_points[i,1]
        A.append([x,y,1,0,0,0,-xw*x,-xw*y,-xw])
        A.append([0,0,0,x,y,1,-yw*x,-yw*y,-yw])
    A=np.array(A)
    eigenvalues, eigenvectors = np.linalg.eig(A.T@A)
    min_eig_idx=np.argmin(eigenvalues)
    smallest_eigen_vector=eigenvectors[:,min_eig_idx]
    H=np.reshape(smallest_eigen_vector,(3,3))
    H=H/H[2,2]
    return H

# Define the callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONUP:
        # Add the clicked point to the list of selected points
        points.append([x, y])
        if len(points) > 1:
            cv2.line(img, points[-2], points[-1], (0, 255,0), 1)
        if len(points) == 4:
            cv2.line(img, points[0], points[-1], (0, 255, 0), 1)

# Load the image
height,width=800,600
img = cv2.imread(path)

if img.shape[1]/800 > img.shape[0]/600:
    width = img.shape[0]/(img.shape[1]/800)
else:
    height = img.shape[1]/(img.shape[0]/600)

img = cv2.resize(img, (int(height), int(width)))
# Create a window to display the image
cv2.namedWindow('Image')

# Register the mouse callback function
cv2.setMouseCallback('Image', mouse_callback)

# Initialize the list of selected points
points = []

b = 0
while True:
    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    if key == 13 and len(points) == 4:
        b = 1
        break
    # If the 'q' key is pressed, quit the program
    if key == ord("q") or len(points)>=8:
        break
    # Draw the detected corners on the original image
    for corner in points:
        x,y = corner
        cv2.circle(img, (x,y), 5, (0, 255 if len(points) >= 4 else 0,0), -1)
    cv2.imshow('Image', img)

for corner in points:
    x,y = corner
    cv2.circle(img, (x,y), 5, (0,0,255), -1)
cv2.imshow('Image', img)
if b == 0:
    cv2.waitKey(0)
# Close the window
cv2.destroyAllWindows()

src_image = cv2.imread(path)
width,height,_=img.shape

src_image = cv2.resize(src_image, (height, width))

# Define the source points and destination points
src_pts = np.array(points[:4], dtype=np.float32)

if len(points) == 4:
    dst_pts = np.array([[0, 0], [src_image.shape[1], 0], [src_image.shape[1], src_image.shape[0]], [0, src_image.shape[0]]], dtype=np.float32)
else:
    dst_pts = np.array(points[4:], dtype=np.float32)

# Find the homography matrix
H, _ = cv2.findHomography(src_pts, dst_pts)

# Warp the source image using the homography matrix
warped_image = cv2.warpPerspective(src_image, H, (img.shape[1], img.shape[0]))


cv2.imshow('Image', warped_image)
cv2.waitKey(0)

cv2.destroyAllWindows()