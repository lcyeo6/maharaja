"""
@description:
    
    Front end interface 
@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
@References:
    http://effbot.org/tkinterbook/
    https://www.datacamp.com/community/tutorials/gui-tkinter-python#ITT
    
"""

import tkinter as tk
import svm_prediction
import lexicon_prediction

def svm():
    # Runs SVM prediction when SVM button is clicked
    text = user_input.get()
    y = svm_prediction.predict(text)
    s.set(str(y))
    return None
    
def lex():
    # Runs Lexicon prediction when Lexicon button is clicked
    text = user_input.get()
    x = lexicon_prediction.predict(text)
    l.set(str(x))
    return None
    
def clear():
    # Clear user input and result
    user_input.delete(0, "end")
    s.set("")
    l.set("")
    return None

# Creating a new Window
window = tk.Tk()
window.geometry("1000x470")
window.resizable(width=False, height=False)

# Title
window.title("Review Prediction")
top_frame = tk.Frame(window).pack()
bottom_frame = tk.Frame(window).pack(side = "bottom")

# Enter review text from user
tk.Label(top_frame, text="Enter review text", font="Verdana 20").pack()
tk.Label(top_frame, text="(Please input English only)", font="Verdana 10").pack()

s = tk.StringVar()
l= tk.StringVar()

# Type box
user_input = tk.Entry(top_frame, textvariable = tk.StringVar(), width = 100)
user_input.pack(pady = 10)

# Buttons
# SVM Button
tk.Button(bottom_frame,text="SVM", width=12, bg="yellow", font="Verdana 13", command=svm).pack(side= "right", padx= 100)

# Lexicon Button
tk.Button(bottom_frame,text="Lexicon", width=12, bg="blue", font="Verdana 13", command=lex).pack(side= "left", padx= 100)

# Clear button
tk.Button(bottom_frame, text="Clear", width=12, bg="red", font="Verdana 13", command=clear).pack(side = "bottom", pady = 100)


# Predicted SVM score
tk.Label(top_frame, text="Predicted Star Rating - SVM", font="Verdana 15").pack()
tk.Label(top_frame, textvariable = s, font = "Verdana 13").pack()

# Predicted Lexicon sentiment
tk.Label(top_frame, text="Predicted Sentiment - Lexicon", font="Verdana 15").pack()
tk.Label(top_frame, textvariable = l, font = "Verdana 13").pack()

window.mainloop()