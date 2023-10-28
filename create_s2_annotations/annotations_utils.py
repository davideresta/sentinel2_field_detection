from PIL import Image                                      # (pip install Pillow)
import numpy as np                                         # (pip install numpy)
from skimage.measure import find_contours                  # (pip install scikit-image)
from skimage.morphology import dilation
from shapely.geometry import Polygon, MultiPolygon         # (pip install Shapely)
import os
import json
import datetime
import cv2 as cv
import sys

# extract a single contour from the sub_mask
# if the sub_mask contains multple contours, dilation is used until there is only one
# one sub_mask should correspond to one object
def find_single_contour(sub_mask):

    # fill the holes in the binary sub_mask, using cv2
    contours_to_fill, _ = cv.findContours(sub_mask,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours_to_fill:
        sub_mask = cv.drawContours(sub_mask,[cnt],0,255,-1)



    ### fields with more than one contour are ignored
    #contours = find_contours(np.pad(sub_mask, ((1,1),(1,1))), 0.5, positive_orientation="low")
    #if len(contours) > 1:
    #    print("Skipping multicontour field")
    #    return None
    #contour = contours[0]

    ### fields with more than one contour are not ignored
    ### extract contours, dilating until the countour is only one, using skimage
    contours = find_contours(np.pad(sub_mask, ((1,1),(1,1))), 0.5, positive_orientation="low")
    while len(contours) > 1:
        sub_mask = dilation(sub_mask)
        contours = find_contours(np.pad(sub_mask, ((1,1),(1,1))), 0.5, positive_orientation="low")
    assert len(contours) == 1
    contour = contours[0]



    # Flip from (row, col) representation to (x, y)
    # and subtract the padding pixel
    for i in range(len(contour)):
        row, col = contour[i]
        contour[i] = (col-1, row-1)
    
    return contour

def create_sub_masks(mask, width, height):
    
    # Initialize a dictionary of sub-masks indexed by 'colors'
    sub_masks = {}
    colors = np.unique(mask)
    for color in colors:
        sub_mask = np.where(mask==color, 255, 0)
        sub_mask = np.array(sub_mask, dtype="uint8")
        sub_masks[str(color)] = sub_mask

    return sub_masks

def get_polygon_in_sub_mask(sub_mask):
    contour = find_single_contour(sub_mask)
    if contour is None:
        return None

    # Make a polygon and simplify it
    poly = Polygon(contour)
    poly = poly.simplify(1.0, preserve_topology=False)
        
    if(poly.is_empty):
        return None

    if type(poly)==MultiPolygon:
        return None
    
    return poly

def create_categories_annotation(category_dict):
    category_list = []

    for key, value in category_dict.items():
        category = {
            "supercategory": key,
            "id": value,
            "name": key
        }
        category_list.append(category)

    return category_list

def create_image_annotation(file_name, width, height, image_id):
    images = {
        "file_name": file_name,
        "height": height,
        "width": width,
        "id": image_id
    }

    return images

def create_annotation(polygon, image_id, category_id, annotation_id):
    min_x, min_y, max_x, max_y = polygon.bounds
    min_x = int(min_x)
    min_y = int(min_y)
    max_x = int(max_x)
    max_y = int(max_y)
    
    
    width = max_x - min_x
    height = max_y - min_y
    bbox = (min_x, min_y, width, height)
    area = polygon.area

    segmentation = [np.array(polygon.exterior.coords).ravel().tolist()]
    #we expect that segmentation boundaries have at least four points
    assert len(segmentation[0]) >= 8

    annotation = {
        "segmentation": segmentation,
        "area": area,
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": bbox,
        "category_id": category_id,
        "id": annotation_id
    }

    return annotation

def get_coco_json_format():
    # Standard COCO format 
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }

    return coco_format
