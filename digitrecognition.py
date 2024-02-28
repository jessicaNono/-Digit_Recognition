import tkinter as tk
from tkinter import Toplevel, Label, Frame
import cv2
from tkinter import simpledialog
import torch
from torchvision import transforms
from tkinter import Label
from PIL  import ImageOps ,  ImageTk, Image
import PIL.Image, PIL.ImageDraw
from model import SimpleNN
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import datetime

learning_rate = 0.03
momentum = 0.9
log_interval = 10
class DigitRecognizerApp:
    def __init__(self, master):

        self.master = master
        master.title("Digit Recognizer")
        self.image_preview_label = Label(master)
        self.image_preview_label.pack()
        self.model = SimpleNN()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                              momentum=momentum)

        self.model.load_state_dict(torch.load('results/mnist_model.pth'))

        self.canvas_width = 300
        self.canvas_height = 300
        self.bg_color = "white"
        self.paint_color = "teal"
        self.radius = 8

        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg=self.bg_color)
        self.canvas.pack()

        self.clear_button = tk.Button(master, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.predict_button = tk.Button(master, text="Predict", command=self.predict_digit_multiple)
        self.predict_button.pack()

        self.predicted_number_label = Label(master, text="Predicted Number: ", font=("Helvetica", 16))
        self.predicted_addition_number_label = Label(master, text="SUM of numbers:", font=("Helvetica", 16))
        self.predicted_number_label.pack()
        self.predicted_addition_number_label.pack()
        self.segments_window = None  # This will store the reference to the segments window
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = PIL.Image.new("L", (300, 300), self.bg_color)
        self.draw = PIL.ImageDraw.Draw(self.image)

        # Initialize self.predicted with a default value
        self.predicted = None
        self.last_x, self.last_y = None, None  # Initialize last point to None
        self.last_x, self.last_y = None, None  # Initialize last point to None

        self.canvas.bind("<B1-Motion>", self.paint)  # Bind painting to mouse movement with button pressed
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_point)  # Reset last point on button release
        self.canvas.bind("<Button-1>", self.start_paint)  # Set starting point for new stroke

    def start_paint(self, event):
        """Set the starting point for a new line."""
        self.last_x, self.last_y = event.x, event.y

    def paint(self, event):
        """Draw a line from the last point to the current point."""
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            # Draw a line from the last point to the current point
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=self.radius * 2, fill=self.paint_color,
                                    capstyle=tk.ROUND, smooth=tk.TRUE)
            # Also draw on the PIL image to keep the drawing in sync
            self.draw.line([self.last_x, self.last_y, x, y], fill=self.paint_color, width=self.radius * 2)

        # Update the last point
        self.last_x, self.last_y = x, y

    def reset_last_point(self, event):
        """Reset the last point to None to start a new line."""
        self.last_x, self.last_y = None, None
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = PIL.Image.new("L", (300, 300), self.bg_color)
        self.draw = PIL.ImageDraw.Draw(self.image)

        # Clear the predicted number label
        self.predicted_number_label.config(text="Predicted Number: ")
        self.reset_last_point()  # Reset last point when the canvas is cleared
        # Close the segments window if it's open

        if self.segments_window:
            self.segments_window.destroy()
            self.segments_window = None  # Reset the reference to None

    def segmentation(self):
        # Convert the PIL Image to an OpenCV image
        img = np.array(self.image.convert('L'))
        img_inverted = cv2.bitwise_not(img)  # Invert colors to match MNIST

        # Apply threshold to get a binary image
        _, thresh = cv2.threshold(img_inverted, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours from left to right
        contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])

        segments_dir = 'segments'
        os.makedirs(segments_dir, exist_ok=True)
        segments = []
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        for i, contour in enumerate(contours):
            # Compute the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            if w * h < 100:  # Skip small contours that are likely not digits
                continue

            # Extract the digit using the bounding box
            digit_img = img_inverted[y:y + h, x:x + w]

            # Resize extracted digit to match input size of neural network
            digit_img_resized = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)
            digit_img_pil = Image.fromarray(
                cv2.bitwise_not(digit_img_resized))  # Convert back to PIL Image and invert colors

            segment_img_path = os.path.join(segments_dir, f'segment_{i}_{timestamp}.png')
            digit_img_pil.save(segment_img_path)
            print(f"Saved segment {i}: {segment_img_path}")
            segments.append(digit_img_resized)

        return segments

    def predict_digit(self):
        # Convert canvas content to an image for preview and processing
        img = self.image.resize((28, 28), Image.LANCZOS).convert('L')
        img_inverted = PIL.ImageOps.invert(img)  # Invert colors to match MNIST
        img_tensor = transforms.ToTensor()(img_inverted)
        img_tensor = transforms.Normalize((0.1307,), (0.3081,))(img_tensor)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        # Display image in GUI
        tk_image = ImageTk.PhotoImage(image=img)
        self.image_preview_label.configure(image=tk_image)
        self.image_preview_label.image = tk_image

        # Predict digit
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(img_tensor)
            _, self.predicted = torch.max(outputs, 1)
            print(f'Predicted Digit: {self.predicted.item()}')

    def predict_digit_multiple(self):

        segments = self.segmentation()
        segments_window = Toplevel(self.master)
        segments_window.title("Segmented Digits and Predictions")
        predictions = []
        for i, segment in enumerate(segments):
            # Convert numpy array segment back to PIL Image for displaying
            segment_img = Image.fromarray(segment).convert('L')
            segment_img_inverted = ImageOps.invert(segment_img)  # Invert colors to match original drawing style
            segment_tk = ImageTk.PhotoImage(segment_img_inverted)

            # Create a frame for each segment and its prediction
            segment_frame = Frame(segments_window)
            segment_frame.pack()

            # Display the segment image
            segment_label = Label(segment_frame, image=segment_tk)
            segment_label.image = segment_tk  # Keep a reference
            segment_label.grid(row=0, column=0)

            # Process each segment for prediction
            segment_img_resized = segment_img.resize((28, 28), Image.LANCZOS)
            img_tensor = transforms.ToTensor()(segment_img_resized)
            img_tensor = transforms.Normalize((0.1307,), (0.3081,))(img_tensor)
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

            # Predict digit for the segment
            with torch.no_grad():
                self.model.eval()
                output = self.model(img_tensor)
                _, predicted = torch.max(output, 1)
                predicted_digit = str(predicted.item())
                predictions.append(str(predicted.item()))

            # Display the predicted digit
                prediction_label = Label(segment_frame, text=f"Predicted: {predicted_digit}", font=("Helvetica", 16))
                prediction_label.grid(row=0, column=1)


            # Combine predictions into a single string and calculate the sum
        predicted_number = ''.join(predictions)
        # Convert each prediction to integer and calculate the sum
        predictions_sum = sum([int(digit) for digit in predictions])

        print(f'Predicted Number: {predicted_number}')
        print(f'SUM: {predictions_sum}')

        self.predicted_number_label.config(text=f"Predicted Number: {predicted_number}")
        self.predicted_addition_number_label.config(text=f"Sum of Digits: {predictions_sum}")

        segments_window.mainloop()

root = tk.Tk()
app = DigitRecognizerApp(root)
root.mainloop()
