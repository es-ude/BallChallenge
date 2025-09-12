import logging

import cv2
import numpy as np

def blur(image: np.ndarray, blursize: tuple[int,int] = (20,20), show_image:bool = False) -> np.ndarray:
    image = cv2.blur(image, blursize)
    if show_image:
        cv2.imshow("blur", image)
        cv2.waitKey(0)
    return image

def morph(image:np.ndarray, kernelsize: tuple[int,int] = (7, 7), show_image:bool = False)->np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernelsize)
    cleaned_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    if show_image:
        cv2.imshow("Cleaned Image", cleaned_image)
        cv2.waitKey(0)
    return cleaned_image

def edge_detection_adaptive_threshold(image:np.ndarray, block_size:int = 31, c:int = 10, show_image:bool = False)->np.ndarray:
    edges = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
    if show_image:
        cv2.imshow("Edges Threshold", edges)
        cv2.waitKey(0)
    return edges

def edge_detection_canny(image:np.ndarray, low_threshold:int = 30, high_threshold:int = 250, show_image:bool = False)->np.ndarray:
    edges = cv2.Canny(image, low_threshold , high_threshold)
    if show_image:
        cv2.imshow("Edges Canny", edges)
        cv2.waitKey(0)
    return edges

def check_contours(contours, area_min = 1000, area_max = 4000 ,image_to_show: np.ndarray =None , show_image: bool = False)->tuple[list[tuple[int, int]], list[float]]:
    def draw_label(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, color=(255, 255, 255),
                   bg_color=(0, 0, 0), thickness=1):
        """Draws text with a background for better visibility."""
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = position
        cv2.rectangle(img, (x, y - text_size[1] - 4), (x + text_size[0] + 4, y + 4), bg_color, -1)
        cv2.putText(img, text, (x + 2, y - 2), font, font_scale, color, thickness)

    position = []
    areas = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)

        logging.debug(f"Contour #{i}: Area: {area}")
        if area < area_min or area > area_max:
            continue  # Skip small noise

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        logging.info(f"Contour #{i}: Area = {area:.2f}, Center = ({cx}, {cy})")
        if (cx, cy) not in position:
            position.append((cx, cy))
            areas.append(area)
        # Draw contour and center point
        if show_image:
            cv2.drawContours(image_to_show, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(image_to_show, (cx, cy), 5, (255, 0, 0), -1)

            # Offset for labels
            offset_x = 10
            offset_y = -10

            # Draw labels with background to avoid overlap
            draw_label(image_to_show, f"#{i}", (cx + offset_x, cy + offset_y), bg_color=(0, 0, 255))
            #  draw_label(result, f"A:{int(area)}", (cx + offset_x, cy + offset_y + 20), bg_color=(0, 255, 255), color=(0, 0, 0))
            cv2.imshow("Object Detection", image_to_show)
            cv2.waitKey(0)
    return position, areas

def find_lines(edges: np.ndarray, rho=1,angle_resolution=np.pi/180, threshold=1000, min_line_length=1000, max_line_gap=5, result_image: np.ndarray=None, show_image: bool = False):
    lines_list =[]
    lines = cv2.HoughLinesP(
        edges, # Input edge image
        rho, # Distance resolution in pixels
        angle_resolution, # Angle resolution in radians
        threshold=threshold, # Min number of votes for valid line
        minLineLength=min_line_length, # Min allowed length of line
        maxLineGap=max_line_gap  # Max allowed gap between line for joining them
    )
    if result_image is not None and show_image:
        # Iterate over points
        for points in lines:
            # Extracted points nested in the list
            x1,y1,x2,y2=points[0]
            # Draw the lines joing the points
            # On the original image
            cv2.line(result_image,(x1,y1),(x2,y2),(0,0,255),3)
            # Maintain a simples lookup list for points
            lines_list.append([(x1,y1),(x2,y2)])

        # Save the result image
        cv2.imshow('detectedLines',result_image)
        cv2.waitKey(0)


def grayscale_object_detection(image:np.ndarray, morph_kernel=(7,7), adaptive_threshold_blocksize=31, adaptive_threshold_c=10, area_min=1000, area_max=4000 ,show_image: bool = False) -> tuple[int, int]:
    result = image.copy()
    morphed_image = morph(image, kernelsize=morph_kernel, show_image=show_image)
    edges = edge_detection_adaptive_threshold(morphed_image, block_size=adaptive_threshold_blocksize, c=adaptive_threshold_c, show_image=show_image)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logging.info("No contours detected.")
        return (-1,-1)
    position, areas = check_contours(contours, area_min=area_min, area_max=area_max,image_to_show=result)
    if len(areas) > 0:
        max_area = 0
        max_area_index = 0
        for i, area in enumerate(areas):
            if area > max_area:
                max_area = area
                max_area_index = i
        return position[max_area_index]
    else:
        logging.info(f"No contours is in size {area_min}-{area_max}.")
        return (-1,-1)

def grayscale_line_detection(image: np.ndarray, morph_kernel=(7,7), adaptive_threshold_blocksize=31, adaptive_threshold_c=10, show_image: bool = False) -> tuple[int, int]:
    result = image.copy()
    morphed_image = morph(image, kernelsize=morph_kernel, show_image=show_image)
    edges = edge_detection_adaptive_threshold(morphed_image, block_size=adaptive_threshold_blocksize, c=adaptive_threshold_c, show_image=show_image)
    find_lines(edges, rho=1, angle_resolution=np.pi/180, threshold=1000, min_line_length=1000, max_line_gap=5, show_image=show_image)

def calculate_grayscale_diff(image1: np.ndarray, image2: np.ndarray, show_image:bool =False) -> np.ndarray:
    image1_g = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_g = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image1_g = np.array(image1_g, dtype=float)
    image2_g = np.array(image2_g, dtype=float)
    dif = image1_g - image2_g
    if show_image:
        cv2.imshow("Div_before_Scaling", dif)
        cv2.waitKey(0)
    return dif

def calculate_red_color_diff(image1: np.ndarray, image2: np.ndarray, show_image:bool =False) -> np.ndarray:
    print(image1.shape)
    image1_r = image1[:,:, 0]
    image2_r = image2[:,:, 0]
    image1_r = np.array(image1_r, dtype=float)
    image2_r = np.array(image2_r, dtype=float)
    dif = image1_r - image2_r
    if show_image:
        cv2.imshow("Div_before_Scaling", dif)
        cv2.waitKey(0)
    return dif

def min_max_normalise_grayscale_diff_pic(image, show_image = False) -> np.ndarray:
    image = (image-image.min())/(image.max()-image.min())*255
    image = np.array(image, dtype=np.uint8)
    if show_image:
        cv2.imshow("Div", image)
        cv2.waitKey(0)
    return image



