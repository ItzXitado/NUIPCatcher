import cv2
import re
import signal
import sys
import threading
from google.cloud import vision
from google.cloud.vision_v1 import types
from google.oauth2 import service_account
import queue
import time
import pyperclip
import winsound
import os
from datetime import datetime
import logging
import keyboard

# Initialize the Vision API client
##credentials = service_account.Credentials.from_service_account_file('credentials.json')
##client = vision.ImageAnnotatorClient(credentials=credentials)

# Initialize logging
logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')

def initialize_vision_client():
    try:
        logging.debug("Initializing Vision API client...")
        print("Initializing Vision API client...")
        credentials = service_account.Credentials.from_service_account_file('credentials.json')
        client = vision.ImageAnnotatorClient(credentials=credentials)
        logging.debug("Vision API client initialized successfully.")
        print("Vision API client initialized successfully.")
        return client
    except Exception as e:
        logging.error("Failed to initialize Vision API client: %s", str(e))
        print(f"Failed to initialize Vision API client: {str(e)}")
        return None

# Initialize the Vision API client
client = initialize_vision_client()

# Define the regex pattern to detect multiple formats
pattern = re.compile(r'(\d{1,5}/\d{2}\.\d{1}\s*\w{5})|(\d{3}/\d{2}\.\d{1}\w{5})')

# Signal handler to gracefully exit
def signal_handler(sig, frame):
    print('Exiting')
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

# Function to remove zeros before the slash
def remove_zeros_before_slash(match):
    if '/' in match:
        parts = match.split('/')
        parts[0] = parts[0].lstrip('0')  # Remove leading zeros from the first part
        return '/'.join(parts)
    return match

# Function to search for specific text pattern in the extracted content
def find_specific_text(extracted_text, search_pattern):
    match = re.search(search_pattern, extracted_text)
    if match:
        return remove_zeros_before_slash(match.group())  # Remove zeros before '/'
    else:
        return None  # Return None if no match is found

# Function to detect text in an image and match regex
def detect_text_and_match_regex(image, output_queue):
    # Convert the image to a format compatible with Google Vision API
    success, encoded_image = cv2.imencode('.jpg', image)
    content = encoded_image.tobytes()
    image = types.Image(content=content)

    # Perform text detection
    response = client.text_detection(image=image)
    texts = response.text_annotations

    # Check for regex pattern matches in detected text
    for text in texts:
        if pattern.search(text.description):
            output_queue.put((True, text.description))
            return
    output_queue.put((False, None))

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

# Reduce frame resolution to speed up processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press Ctrl+C to exit")

# Queue to hold detection results
output_queue = queue.Queue(maxsize=2)  # Reduce queue size for faster processing

# Separate thread for processing frames
def process_frames():
    while True:
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None:
                break
            detect_text_and_match_regex(frame, output_queue)
        except queue.Empty:
            continue

frame_queue = queue.Queue(maxsize=2)  # Reduce queue size for faster processing
processing_thread = threading.Thread(target=process_frames)
processing_thread.start()

# Buffer to store detected matches
detected_texts = []
freeze_time = 3  # Number of seconds to freeze after detection
frozen_until = 0  # Timestamp until which processing is frozen

# Debouncing variables
debounce_count = 0
debounce_threshold = 3  # Number of consecutive detections required to confirm

# Initialize last_detection variable
last_detection = ""
match_processed = False  # Flag to track if the match was already processed

# Create directory for today
today = datetime.now().strftime("%Y-%m-%d")
if not os.path.exists(today):
    os.makedirs(today)

# Create text file for matches log
log_file_path = os.path.join(today, "matches.txt")
with open(log_file_path, 'a') as log_file:
    log_file.write("Match Log for " + today + "\n")

# Create directory for images
images_dir = os.path.join(today, "images")
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

frame_count = 0
frame_skip = 2  # Process every nth frame

# Variable to control scanning state
scanning = False

def toggle_scanning():
    global scanning
    scanning = not scanning
    if scanning:
        print("Started scanning")
    else:
        print("Finished scanning")

# Register the hotkey
keyboard.add_hotkey('ctrl+alt+.', toggle_scanning)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Skip processing if frozen
    if time.time() < frozen_until:
        cv2.putText(frame, f"Match Found: {last_detection}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Check if scanning is enabled
    if scanning:
        # Skip frames to reduce processing load
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Put frame in queue for processing
        if not frame_queue.full():
            frame_queue.put(frame)

        # Get detection result from queue
        try:
            match_found, matched_text = output_queue.get_nowait()

            # Display the result on the frame
            if match_found:
                matches = find_specific_text(matched_text, pattern)
                if matches and (last_detection != matches or not match_processed):
                    if last_detection != matches:
                        # If a new match is found, reset the debounce count
                        debounce_count = 0
                        last_detection = matches
                        match_processed = False  # Reset match processed flag

                    # Increment debounce count if the same match is found
                    debounce_count += 1

                    if debounce_count >= debounce_threshold:
                        print(matches)
                        pyperclip.copy(matches)
                        winsound.Beep(1000, 500)  # Beep sound (frequency 1000 Hz, duration 500 ms)
                        cv2.putText(frame, f"Match Found: {matches}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        detected_texts.append(matches)

                        # Log the match
                        with open(log_file_path, 'a') as log_file:
                            log_file.write(matches + "\n")

                        # Save the frame with match text as filename
                        match_filename = matches.replace("/", ".") + ".jpg"
                        match_filepath = os.path.join(images_dir, match_filename)
                        cv2.imwrite(match_filepath, frame)

                        # Filter out most common match if buffer size exceeds limit
                        if len(detected_texts) > 10:
                            detected_texts = detected_texts[-10:]  # Keep only last 10 matches
                        frozen_until = time.time() + freeze_time  # Freeze processing for a few seconds
                        debounce_count = 0  # Reset debounce count after confirming detection
                        match_processed = True  # Set match processed flag
                        scanning = False  # Stop scanning after match is found
                        print("Finished scanning")
        except queue.Empty:
            pass

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Break the loop on 'q' key press (optional, as we handle SIGINT)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()
frame_queue.put(None)  # Signal the processing thread to stop
