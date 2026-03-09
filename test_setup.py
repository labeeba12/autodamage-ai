import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from fastapi import FastAPI

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("All imports successful!")
