# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 19:35:28 2018

@author: Shree
"""

"""from tkinter import *

master = Tk()
Label(master, text="First Name").grid(row=0)
Label(master, text="Last Name").grid(row=1)

e1 = Entry(master)
e2 = Entry(master)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)


mainloop( )"""

from tkinter import *
root = Tk()
root.geometry("200x100")

def retrieve_input():
    inputval = textBox.get("1.0","end-1c")
    print (inputval)

textBox = Text(root, height=2, width=10)
textBox.pack()

buttonCommit = Button(root, height=1, width=2, text="Commit", 
                      command=retrieve_input)

buttonCommit.pack()
mainloop();