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

        self.predict_button = tk.Button(master, text="Predict", command=self.predict_digit)
        self.predict_button.pack()

        self.correct_button = tk.Button(master, text="Correct", command=self.correct_prediction)
        self.correct_button.pack()

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

    def correct_prediction(self):
        if self.predicted is not None:  # Ensure there was a prediction
            correct_label = simpledialog.askinteger("Input", "What is the correct digit?", parent=self.master,
                                                    minvalue=0, maxvalue=9)
            print(correct_label)
            print(self.predicted.item())
            if correct_label is not None and correct_label != self.predicted.item():
                # Convert canvas content to an image for correction
                img = self.image.resize((28, 28), Image.LANCZOS).convert('L')
                img_inverted = ImageOps.invert(img)  # Invert colors if necessary to match training data

                # Convert image to tensor for correction
                img_tensor = transforms.ToTensor()(img_inverted)
                img_tensor = transforms.Normalize((0.1307,), (0.3081,))(img_tensor)
                img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

                correct_label_tensor = torch.tensor([correct_label], dtype=torch.long)

                # Perform correction
                self.model.train()  # Switch to training mode
                self.optimizer.zero_grad()  # Clear previous gradients
                outputs = self.model(img_tensor)
                loss = F.nll_loss(outputs, correct_label_tensor)  # Ensure loss function is consistent with training
                loss.backward()  # Backpropagate the error
                self.optimizer.step()  # Adjust model parameters
                self.model.eval()  # Switch back to evaluation mode

                # Save the updated model
                torch.save(self.model.state_dict(), 'results/mnist_model.pth')
                print(f"Model updated with correction: {correct_label}")
            else:
                print("No correction needed or provided.")
        else:
            print("No prediction has been made yet.")


root = tk.Tk()
app = DigitRecognizerApp(root)
root.mainloop()
