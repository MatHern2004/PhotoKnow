import tkinter as tk

root = tk.Tk()

root.title("Simple Tkinter GUI")
root.geometry("400x300+700+300") 

my_label = tk.Label(root, text="Welcome to our image captioner!")

my_label.pack(pady=20)

root.mainloop()