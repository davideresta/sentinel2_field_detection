import glob
from annotations_utils import *

import numpy as np
import rasterio
import netCDF4 as nc
from os.path import sep

# Label ids of the dataset
category_ids = {
    "field": 1,
}

# Define which colors match which categories in the images
# All instances are fields
category_colors = { str(i): 1 for i in range(1, 5000)}

# Get "images" and "annotations" info
def get_annotations(mask_folder):
    # This id will be automatically increased as we go
    annotation_id = 0
    image_id = 0

    categories = create_categories_annotation(category_ids)
    images = []
    annotations = []
    
    for mask_file in glob.glob(mask_folder + "*.tif"):

        print("Creating annotations from file: " + mask_file)
        
        # We make a reference to the original file in the COCO JSON file
        original_file_name = os.path.basename(mask_file).split(".")[0] + ".nc"
        original_file_name = original_file_name.replace("label", "")

        # read tiff
        src = rasterio.open(mask_file)
        array = np.array(src.read(4), dtype='float32')
        array = np.where(array==-10000, 0, array)
        assert np.min(array) >= 0
        mask = np.array(array, dtype='uint32')

        w, h = mask.shape
        assert w == 256
        assert h == 256

        # images info
        image = create_image_annotation(original_file_name, w, h, image_id)
        images.append(image)

        sub_masks = create_sub_masks(mask, w, h)
        
        for color, sub_mask in sub_masks.items():

            # "0" color is background
            if color=="0":
                continue

            category_id = category_colors[color]

            polygon = get_polygon_in_sub_mask(sub_mask)
            if polygon is None:
                continue
            annotation = create_annotation(polygon, image_id, category_id, annotation_id)
            annotations.append(annotation)
            annotation_id += 1
                
        image_id += 1
        
    return categories, images, annotations, annotation_id

if __name__ == "__main__":

    coco_format = get_coco_json_format()
    
    for keyword in ["train", "val"]:
        mask_folder = ("..{sep}data{sep}sentinel2{sep}{keyword}_masks{sep}").format(keyword=keyword, sep=sep)
        
        coco_format["categories"], coco_format["images"], coco_format["annotations"], annotation_cnt = \
            get_annotations(mask_folder)

        with open("..{sep}data{sep}sentinel2{sep}annotations{sep}instances_{keyword}.json".format(keyword=keyword, sep=sep), "w") as outfile:
            json.dump(coco_format, outfile)
        
        print("Created %d annotations for images in folder: %s" % (annotation_cnt, mask_folder))
