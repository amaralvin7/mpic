import matplotlib.pyplot as plt
import os
import tkinter as tk
import shutil
from PIL import ImageTk, Image
import sys

canvas_width=200
canvas_height=200


#This first block of code sets up the structure of the GUI (Graphical User Interface), and defines a function to perform whenever a button is pushed 
class Particle_ID_GUI:

    #This block sets up the overall structure of the GUI
    def __init__(self, root, path):
        
        self.root = root
        root.title("Assign particle IDs")
        self.move_from = os.path.join(path, 'start')
        self.move_to = os.path.join(path, 'finish')
        self.file_list = [f for f in os.listdir(self.move_from) if '.jpg' in f]
        self.label_list = []
        self.file_counter = 0
        self.current_file = self.file_list[self.file_counter]
        self.prev_label = None
        self.prev_file = None
        
        ##Create an area where the photo will be displayed
        self.particle_photo = tk.Canvas(root, width=canvas_width, height=canvas_height)
        self.particle_photo.create_rectangle(0,100,100,0, fill='white', outline='white', tag='the_tag')
        self.particle_photo.grid(columnspan=3)

        ##Buttons
        # self.agg_button = tk.Button(root, text='aggregate', command=lambda: self.move_image('aggregate'))
        # self.agg_button.grid(row=1,column=2)

        # self.dd_button = tk.Button(root, text='dense_detritus', command=lambda: self.move_image('dense_detritus'))
        # self.dd_button.grid(row=2,column=2)

        # self.fiber_button = tk.Button(root, text='fiber', command=lambda: self.move_image('fiber'))
        # self.fiber_button.grid(row=3,column=2)

        # self.llp_button = tk.Button(root, text='large_loose_pellet', command=lambda: self.move_image('large_loose_pellet'))
        # self.llp_button.grid(row=3,column=2)
        
        # self.lfp_button = tk.Button(root, text='long_pellet', command=lambda: self.move_image('long_pellet'))
        # self.lfp_button.grid(row=2,column=2)

        # self.mp_button = tk.Button(root, text='mini_pellet', command=lambda: self.move_image('mini_pellet'))
        # self.mp_button.grid(row=5,column=2)

        # self.phyto_button = tk.Button(root, text='phytoplankton', command=lambda: self.move_image('phytoplankton'))
        # self.phyto_button.grid(row=1,column=3)

        # self.rhiz_button = tk.Button(root, text='rhizaria', command=lambda: self.move_image('rhizaria'))
        # self.rhiz_button.grid(row=2,column=3)

        # self.salp_button = tk.Button(root, text='salp_pellet', command=lambda: self.move_image('salp_pellet'))
        # self.salp_button.grid(row=3,column=3)

        # self.short_button = tk.Button(root, text='short_pellet', command=lambda: self.move_image('short_pellet'))
        # self.short_button.grid(row=4,column=3)

        # self.swim_button = tk.Button(root, text='swimmer', command=lambda: self.move_image('swimmer'))
        # self.swim_button.grid(row=4,column=2)

        # self.bub_button = tk.Button(root, text='bubble', command=lambda: self.move_image('bubble'))
        # self.bub_button.grid(row=5,column=3)

        # self.noise_button = tk.Button(root, text='noise', command=lambda: self.move_image('noise'))
        # self.noise_button.grid(row=3,column=0)

        self.b1 = tk.Button(root, text='narrow', command=lambda: self.move_image('fiber_narrow'))
        self.b1.grid(row=1,column=1)

        self.b2 = tk.Button(root, text='thick', command=lambda: self.move_image('fiber_thick'))
        self.b2.grid(row=3,column=1)

        ##Create an undo button
        self.close_button = tk.Button(root, text='UNDO', command=self.undo)
        self.close_button.grid(row=1,column=0)

        self.skip_button = tk.Button(root, text='SKIP', command=lambda: self.move_image('skip'))
        self.skip_button.grid(row=3,column=0)
        
        self.advance_photo()


 #Define a function that will be used repeatedly to advance the photo images.  This can be adjusted depending on file directory structure and naming#
    def set_next_file(self):
        
        self.file_counter += 1
        if self.file_counter == len(self.file_list):
            sys.exit()
        else:
            self.current_file = self.file_list[self.file_counter]


    def advance_photo(self):
        photo_path = os.path.join(self.move_from, self.current_file)
        print(self.current_file)
        plt.imshow(Image.open(photo_path))
        try:
            particle=Image.open(photo_path)
            if float(particle.size[0])>float(particle.size[1]):
                particle_resize_widthpercent = (canvas_width/float(particle.size[0]))
                particle_resize_height = int((float(particle.size[1])*float(particle_resize_widthpercent)))
                particle_resize = particle.resize((canvas_width,particle_resize_height))
                self.particle_photo.image = ImageTk.PhotoImage(particle_resize)
                self.particle_photo.create_image(0,0,image = self.particle_photo.image, anchor = 'nw', tag= 'the_tag')
            else:
                particle_resize_heightpercent = (canvas_height/float(particle.size[1]))
                particle_resize_width = int((float(particle.size[0])*float(particle_resize_heightpercent)))
                particle_resize = particle.resize((particle_resize_width,canvas_height))
                self.particle_photo.image = ImageTk.PhotoImage(particle_resize)
                self.particle_photo.create_image(0,0,image = self.particle_photo.image, anchor = 'nw', tag= 'the_tag')
        except:
            self.particle_photo.create_rectangle(0,250,250,0, fill='white', outline='white', tag= 'the_tag')
            self.particle_photo.create_text(100,100,text="Not available", anchor = 'nw', tag='the_tag')


    #Function if the 'Aggregate' button is pushed.  Label it in the new csv file and display the next particle
    def move_image(self, label):
        plt.close()
        self.label_list.append(label)
        shutil.copy(os.path.join(self.move_from, self.current_file), os.path.join(self.move_to, label, self.current_file))
        os.remove(os.path.join(self.move_from, self.current_file))
        self.particle_photo.delete('the_tag')
        self.prev_label = label
        self.prev_file = self.current_file
        self.set_next_file()
        self.advance_photo()


    def undo(self):
        shutil.copy(os.path.join(self.move_to, self.prev_label, self.prev_file), os.path.join(self.move_from, self.prev_file), )
        os.remove(os.path.join(self.move_to, self.prev_label, self.prev_file))
        self.particle_photo.delete('the_tag')
        self.current_file = self.prev_file
        self.label_list.pop()
        self.file_counter -= 1
        if self.file_counter == 0:
            self.prev_file = None
            self.prev_label = None
        else:
            self.prev_file = self.file_list[self.file_counter - 1]
            self.prev_label = self.label_list[self.file_counter - 1]
        self.advance_photo()


if __name__ == '__main__':
    path = '/Users/particle/imgs/relabel'
    root = tk.Tk()
    my_gui = Particle_ID_GUI(root, path)
    root.mainloop()
