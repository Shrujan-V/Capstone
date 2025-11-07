from PIL import Image, ImageDraw
from pathlib import Path

# Define the bounding boxes for the tracts
tract_boxes = {
    "tract_1": [160, 110, 253, 213],
    "tract_2": [158, 234, 236, 425],
    "tract_3": [145, 429, 199, 592],
    "tract_4": [170, 415, 331, 487],
    "tract_5": [320, 424, 371, 569],
    "tract_6": [298, 249, 371, 421],
    "tract_7": [294, 108, 373, 213],
    "tract_8": [200, 198, 337, 266],
}

# Function to draw bounding boxes on an image
def draw_bounding_boxes(image_path, output_path, boxes):
    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Draw each bounding box
    for tract, box in boxes.items():
        draw.rectangle(box, outline="red", width=3)  # Red bounding box with width 3
        draw.text((box[0], box[1] - 10), tract, fill="red")  # Label above the box

    # Save the image with bounding boxes
    image.save(output_path)
    print(f"Bounding boxes drawn and saved to {output_path}")

idx = 1

flair_before = Path(r"D:\Capstone\dataset\main\flair-before")
flair_after = Path(r"D:\Capstone\dataset\main\flair-after")

flair_after = list(flair_after.glob("*.png"))
flair_before = list(flair_before.glob("*.png"))

for before, after in zip(flair_before, flair_after):
    flair_before_path = str(before)
    flair_after_path = str(after)

    output_before_path = flair_before_path.replace("flair-before", "flair-before-tracts")
    output_after_path = flair_after_path.replace("flair-after", "flair-after-tracts")

    draw_bounding_boxes(flair_before_path, output_before_path, tract_boxes)
    draw_bounding_boxes(flair_after_path, output_after_path, tract_boxes)

    idx += 1

    if idx > 40:
        break

    print(f"Processed {idx} images.")