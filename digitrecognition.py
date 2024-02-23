import tkinter as tk
from tkinter import Button, Canvas
import torch
from torchvision import transforms
import PIL.Image, PIL.ImageDraw
from model import  SimpleNN

class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Digit Recognizer")

        self.model = SimpleNN()
        self.model.load_state_dict(torch.load('mnist_model.pth'))
        self.model.eval()

        self.canvas_width = 200
        self.canvas_height = 200
        self.bg_color = "white"
        self.paint_color = "black"
        self.radius = 8

        self.canvas = Canvas(master, width=self.canvas_width, height=self.canvas_height, bg=self.bg_color)
        self.canvas.pack()

        self.clear_button = Button(master, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.predict_button = Button(master, text="Predict", command=self.predict_digit)
        self.predict_button.pack()

        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = PIL.Image.new("L", (200, 200), self.bg_color)
        self.draw = PIL.ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = (event.x - self.radius), (event.y - self.radius)
        x2, y2 = (event.x + self.radius), (event.y + self.radius)
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.paint_color, outline="")
        self.draw.ellipse([x1, y1, x2, y2], fill=self.paint_color)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = PIL.Image.new("L", (200, 200), self.bg_color)
        self.draw = PIL.ImageDraw.Draw(self.image)

    def predict_digit(self):
        img = self.image.resize((28, 28), PIL.Image.Resampling.LANCZOS)  # Updated line
        img = transforms.ToTensor()(img)
        img = img.view(1, 784)  # Flatten the image
        with torch.no_grad():
            outputs = self.model(img)
            _, predicted = torch.max(outputs, 1)
            print(f'Predicted Digit: {predicted.item()}')
root = tk.Tk()
app = DigitRecognizerApp(root)
root.mainloop()
