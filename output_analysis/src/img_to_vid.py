import cv2
import os
import re

def natural_sort_key(s):
    """Generate a key for natural sorting."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def create_video_from_images(image_folder, output_video, fps=20):
    """
    Create a video from a folder of images.

    Parameters:
        image_folder (str): Path to the folder containing the images.
        output_video (str): Path to save the output video file.
        fps (int): Frames per second for the output video.
    """
    # Get a list of image files
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    # Sort files naturally
    image_files = sorted(image_files, key=natural_sort_key)

    if not image_files:
        print("No images found in the folder.")
        return

    # Get the size of the first image
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 videos
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Write each image to the video
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Video saved to {output_video}")

# Usage example
if __name__ == "__main__":
    image_folder = "./renders"  # Replace with the path to your folder
    output_video = "kitti_gt.mp4"      # Replace with desired output file name
    create_video_from_images(image_folder, output_video)