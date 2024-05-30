from PIL import Image, ImageDraw, ImageFont
import numpy as np


## Define Utility Functions
### Define a function to annotate an image with segmentation masks
def draw_masks_pil(image, masks, labels, colors, alpha=0.3, threshold=0.5):
    """
    Annotates an image with segmentation masks, labels, and optional alpha blending.

    This function draws segmentation masks on the provided image using the given mask arrays, 
    colors, labels, and alpha values for transparency.

    Parameters:
    image (PIL.Image): The input image on which annotations will be drawn.
    masks (numpy.ndarray): A 3D numpy array of shape (n_masks, height, width) representing segmentation masks.
    labels (list of str): A list of labels corresponding to each segmentation mask.
    colors (list of tuples): A list of RGB tuples for each segmentation mask and its corresponding label.
    alpha (float, optional): The alpha value for mask transparency. Defaults to 0.3.
    threshold (float, optional): The threshold value to convert mask to binary. Defaults to 0.5.

    Returns:
    annotated_image (PIL.Image): The image annotated with segmentation masks and labels.
    """
    
    # Create a copy of the image
    annotated_image = image.copy()
    annotated_image.convert('RGBA')

    # Create an ImageDraw object for drawing on the image
    draw = ImageDraw.Draw(annotated_image)

    # Loop through the bounding boxes and labels in the 'annotation' DataFrame
    for i in range(len(labels)):
        
        # Get the segmentation mask
        mask = masks[i][0, :, :]
        mask_color = [*colors[i], alpha*255]

        # Create an empty 3D array with shape (height, width, 3)
        rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        
        # Use broadcasting to populate it with the mask color where the mask is 1
        rgb_mask[mask > threshold] = mask_color
        
        # Convert the numpy array to a PIL Image
        mask_img = Image.fromarray(rgb_mask)
        
        # Draw segmentation mask on sample image
        annotated_image.paste(mask_img, (0,0), mask=mask_img)
        
    return annotated_image


## Define a function to annotate an image with bounding boxes

def draw_bboxes_pil(image, boxes, labels, colors, font, width:int=2, font_size:int=18, probs=None):
    """
    Annotates an image with bounding boxes, labels, and optional probability scores.

    This function draws bounding boxes on the provided image using the given box coordinates, 
    colors, and labels. If probabilities are provided, they will be added to the labels.

    Parameters:
    image (PIL.Image): The input image on which annotations will be drawn.
    boxes (list of tuples): A list of bounding box coordinates where each tuple is (x, y, w, h).
    labels (list of str): A list of labels corresponding to each bounding box.
    colors (list of str): A list of colors for each bounding box and its corresponding label.
    font (str): Path to the font file to be used for displaying the labels.
    width (int, optional): Width of the bounding box lines. Defaults to 2.
    font_size (int, optional): Size of the font for the labels. Defaults to 25.
    probs (list of float, optional): A list of probability scores corresponding to each label. Defaults to None.

    Returns:
    annotated_image (PIL.Image): The image annotated with bounding boxes, labels, and optional probability scores.
    """
    
    # Define a reference diagonal
    REFERENCE_DIAGONAL = 1000
    
    # Scale the font size using the hypotenuse of the image
    font_size = int(font_size * (np.hypot(*image.size) / REFERENCE_DIAGONAL))
    
    # Add probability scores to labels
    if probs is not None:
        labels = [f"{label}: {prob*100:.2f}%" for label, prob in zip(labels, probs)]
    
    # Create a copy of the image
    annotated_image = image.copy()

    # Create an ImageDraw object for drawing on the image
    draw = ImageDraw.Draw(annotated_image)

    # Loop through the bounding boxes and labels in the 'annotation' DataFrame
    for i in range(len(labels)):
        # Get the bounding box coordinates
        x, y, x2, y2 = boxes[i]

        # Create a tuple of coordinates for the bounding box
        shape = (x, y, x2, y2)

        # Draw the bounding box on the image
        draw.rectangle(shape, outline=colors[i], width=width)
        
        # Load the font file
        fnt = ImageFont.truetype(font, font_size)
        
        # Draw the label box on the image
        label_w, label_h = draw.textbbox(xy=(0,0), text=labels[i], font=fnt)[2:]
        draw.rectangle((x, y-label_h, x+label_w, y), outline=colors[i], fill=colors[i], width=width)

        # Draw the label on the image
        draw.multiline_text((x, y-label_h), labels[i], font=fnt, fill='black' if np.mean(colors[i]) > 127.5 else 'white')
        
    return annotated_image
