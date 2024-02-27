import tkinter as tk
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

learning_rate = 0.01
momentum = 0.5
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

        self.canvas_width = 200
        self.canvas_height = 200
        self.bg_color = "white"
        self.paint_color = "black"
        self.radius = 8

        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg=self.bg_color)
        self.canvas.pack()

        self.clear_button = tk.Button(master, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.predict_button = tk.Button(master, text="Predict", command=self.predict_digit_multiple)
        self.predict_button.pack()

  
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = PIL.Image.new("L", (200, 200), self.bg_color)
        self.draw = PIL.ImageDraw.Draw(self.image)

        # Initialize self.predicted with a default value
        self.predicted = None

    def paint(self, event):
        x1, y1 = (event.x - self.radius), (event.y - self.radius)
        x2, y2 = (event.x + self.radius), (event.y + self.radius)
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.paint_color, outline="")
        self.draw.ellipse([x1, y1, x2, y2], fill=self.paint_color)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = PIL.Image.new("L", (200, 200), self.bg_color)
        self.draw = PIL.ImageDraw.Draw(self.image)


    def segmentation(self):
        # Convert canvas content to a grayscale NumPy array
        img = self.image.convert('L')
        img_inverted = PIL.ImageOps.invert(img)  # Invert colors to match MNIST
        img_array = np.array(img_inverted)

        # Threshold to identify non-background (digit) columns
        threshold = 2
        non_bg_columns = np.where(img_array.max(axis=0) > threshold)[0]

        if len(non_bg_columns) == 0:
            print("No digit found in the image.")
            return
        segments_dir = 'segments'
        os.makedirs(segments_dir, exist_ok=True)
        # Find separators as gaps between consecutive non-background columns
        separators = np.diff(non_bg_columns) > 1
        sep_positions = non_bg_columns[:-1][separators] + 1

        # Include start and end positions for slicing
        seg_starts = np.insert(sep_positions, 0, 0)
        seg_ends = np.append(sep_positions, non_bg_columns[-1] + 1)

        segments = []
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        for i, (start, end) in enumerate(zip(seg_starts, seg_ends)):
            segment = img_array[:, start:end]
            if segment.shape[1] == 0:  # Skip empty segments
                continue

            # Process each segment
            segment_img = Image.fromarray(segment).resize((28, 28), Image.LANCZOS)
            segment_img_path = os.path.join(segments_dir, f'segment_{i}_{timestamp}.png')

            segment_img.save(segment_img_path)
            print(f"Saved segment {i}: {segment_img_path}")
            segments.append(segment)

        return  segments

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
        segments = self.segmentation();
        predictions = []
        print("prediction")
        for segment  in segments:
            # Process each segment
            segment_img = Image.fromarray(segment).resize((28, 28), Image.LANCZOS)
            img_tensor = transforms.ToTensor()(segment_img)
            img_tensor = transforms.Normalize((0.1307,), (0.3081,))(img_tensor)
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

            # Predict digit for the segment
            with torch.no_grad():
                self.model.eval()
                output = self.model(img_tensor)
                _, predicted = torch.max(output, 1)
                predictions.append(str(predicted.item()))

        # Combine predictions into a single string
        predicted_number = ''.join(predictions)
        print(f'Predicted Number: {predicted_number}')




root = tk.Tk()
app = DigitRecognizerApp(root)
root.mainloop()
