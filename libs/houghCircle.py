import cv2
import numpy as np
# from libs import helper
from collections import defaultdict


def houghCircles(img_path:str, r_min:int = 20, r_max:int = 100, delta_r:int = 1, num_thetas:int = 100, bin_threshold:float = 0.4, min_edge_threshold:int = 100, max_edge_threshold:int = 200, pixel_threshold:int = 20,  post_process:bool = True):
    
    """Function that detects circles in haugh domain.

    Args:
        img_path: image path.
        r_min: minimum detected circles radius.
        r_max maximum detected circles radius.
        delta_r: the step to arrange the radius values.
        nun_thetas: the step to arrange the thetas values.
        bin_threshold: the threshold that decide whether takes a bin or not.
        min_edge_threshold: canny edge detection min threshold constant.
        max_edge_threshold: canny edge detection max threshold constant.
        pixel_threshold: filltering the duplicate drwan circles pixel threshold

    Returns:
        output_img: (np.ndarray): the output image where circles are drwan at the edges of circles shapes.
        
    """
    
    input_img = cv2.imread(img_path)

    #Edge detection on the input image
    edge_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    #ret, edge_image = cv2.threshold(edge_image, 120, 255, cv2.THRESH_BINARY_INV)
    edge_image = cv2.Canny(edge_image, min_edge_threshold, max_edge_threshold)

    if edge_image is None:
        print ("Error in input image!")
        return

    #image size
    img_height, img_width = edge_image.shape[:2]
    
    # R and Theta ranges
    dtheta = int(360 / num_thetas)
    
    ## Thetas is bins created from 0 to 360 degree with increment of the dtheta
    thetas = np.arange(0, 360, step=dtheta)
    
    ## Radius ranges from r_min to r_max 
    rs = np.arange(r_min, r_max, step=delta_r)
    
    circle_candidates = condidateCircles(thetas, rs, num_thetas)

    accumulator = calculateAccumlator(img_height, img_width, edge_image, circle_candidates)
    
    # Output image with detected lines drawn
    output_img = input_img.copy()
    # Output list of detected circles. A single circle would be a tuple of (x,y,r,threshold) 
    out_circles = []
    
    # Sort the accumulator based on the votes for the candidate circles 
    for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
        x, y, r = candidate_circle
        current_vote_percentage = votes / num_thetas
        if current_vote_percentage >= bin_threshold: 
            # Shortlist the circle for final result
            out_circles.append((x, y, r, current_vote_percentage))
    
    # Post process the results, can add more post processing later.
    if post_process :
        out_circles = postProcces(out_circles, pixel_threshold)
        
    # Draw shortlisted circles on the output image
    for x, y, r, v in out_circles:
        output_img = cv2.circle(output_img, (x,y), r, (0,255,0), 2)
    
    return output_img


def calculateAccumlator(img_height, img_width, edge_image, circle_candidates):
    # Hough Accumulator, we are using defaultdic instead of standard dict as this will initialize for key which is not 
    # aready present in the dictionary instead of throwing exception.
    accumulator = defaultdict(int)
    for y in range(img_height):
        for x in range(img_width):
            if edge_image[y][x] != 0: #white pixel
                # Found an edge pixel so now find and vote for circle from the candidate circles passing through this pixel.
                for r, rcos_t, rsin_t in circle_candidates:
                    x_center = x - rcos_t
                    y_center = y - rsin_t
                    accumulator[(x_center, y_center, r)] += 1 #vote for current candidate
    return accumulator

def condidateCircles(thetas, rs, num_thetas):
    # Evaluate and keep ready the candidate circles dx and dy for different delta radius
    # based on the the parametric equation of circle.
    # x = x_center + r * cos(t) and y = y_center + r * sin(t),  
    # where (x_center,y_center) is Center of candidate circle with radius r. t in range of [0,2PI)

    # Calculate Cos(theta) and Sin(theta) it will be required later
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))
    circle_candidates = []

    for r in rs:
        for t in range(num_thetas):
            #instead of using pre-calculated cos and sin theta values you can calculate here itself by following
            #circle_candidates.append((r, int(r*cos(2*pi*t/num_thetas)), int(r*sin(2*pi*t/num_thetas))))
            #but its better to pre-calculate and use it here.
            circle_candidates.append((r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))
    return circle_candidates

def postProcces(out_circles, pixel_threshold):
    postprocess_circles = []
    for x, y, r, v in out_circles:
      # Exclude circles that are too close of each other
      # all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc, v in postprocess_circles)
      # Remove nearby duplicate circles based on pixel_threshold
      if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for xc, yc, rc, v in postprocess_circles):
        postprocess_circles.append((x, y, r, v))
    return postprocess_circles


# circle_img = houghCircles('../assets/circles.png', r_min = 20, r_max = 100, bin_threshold = 0.3, pixel_threshold = 20)

# cv2.imshow('Detected Circles', circle_img)
# cv2.waitKey(0)