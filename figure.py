import os

import math
import warnings
import tkinter as tk
from collections import OrderedDict

from tkinter import ttk, font
from PIL import Image, ImageTk
import re


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


dataset_paths = ['vimeo_test', 'vimeo_septuplet_test', 'snu_film_easy', 'snu_film_medium', 'snu_film_hard', 'snu_film_extreme',
                 'snu_film_arb_medium', 'snu_film_arb_hard', 'snu_film_arb_extreme', 'xiph_cropped-4k', 'xiph_resized-2k', 'xtest_single', 'xtest_multiple']


class AutoScrollbar(ttk.Scrollbar):
    """ A scrollbar that hides itself if it's not needed. Works only for grid geometry manager """

    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        raise tk.TclError('Cannot use place with the widget ' + self.__class__.__name__)


class CanvasImage(tk.Canvas):
    """ Display and zoom image """

    def __init__(self, placeholder, path, row, column, width, height):
        """ Initialize the ImageFrame """
        self.imscale = 1.0  # scale for the canvas image zoom, public for outer classes
        self.__delta = 1.3  # zoom magnitude
        self.__filter = Image.Resampling.BICUBIC  # could be: NEAREST, BILINEAR, BICUBIC and ANTIALIAS
        self.__previous_state = 0  # previous state of the keyboard
        self.path = path  # path to the image, should be public for outer classes
        # Create ImageFrame in placeholder widget
        self.__imframe = placeholder  # placeholder of the ImageFrame object
        # Vertical and horizontal scrollbars for canvas
        hbar = AutoScrollbar(self.__imframe, orient='horizontal')
        vbar = AutoScrollbar(self.__imframe, orient='vertical')
        hbar.grid(row=row + 1, column=column, columnspan=2, sticky='we')
        vbar.grid(row=row, column=column + 2, sticky='ns')
        # Create canvas and bind it with scrollbars. Public for outer classes
        with warnings.catch_warnings():  # suppress DecompressionBombWarning
            warnings.simplefilter('ignore')
            self.__image = Image.open(self.path)  # open image, but down't load it
        self.imwidth, self.imheight = self.__image.size  # public for outer classes
        self.canvas = tk.Canvas(self.__imframe, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set,
                                width=width, height=height)
        self.canvas.grid(row=row, column=column, columnspan=2, sticky='nswe')
        self.canvas.update()  # wait till canvas is created
        hbar.configure(command=self.__scroll_x)  # bind scrollbars to the canvas
        vbar.configure(command=self.__scroll_y)
        # Bind events to the Canvas
        self.canvas.bind('<Configure>', lambda event: self.__show_image())  # canvas is resized
        self.canvas.bind('<ButtonPress-1>', self.__move_from)  # remember canvas position
        self.canvas.bind('<ButtonPress-2>', self.__default)  # remember canvas position
        self.canvas.bind('<B1-Motion>', self.__move_to)  # move canvas to the new position
        self.canvas.bind('<MouseWheel>', self.__wheel)  # zoom for Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>', self.__wheel)  # zoom for Linux, wheel scroll down
        self.canvas.bind('<Button-4>', self.__wheel)  # zoom for Linux, wheel scroll up
        # Handle keystrokes in idle mode, because program slows down on a weak computers,
        # when too many key stroke events in the same time
        self.canvas.bind('<Key>', lambda event: self.canvas.after_idle(self.__keystroke, event))
        # Decide if this image huge or not
        self.__huge = False  # huge or not
        self.__huge_size = 14000  # define size of the huge image
        self.__band_width = 1024  # width of the tile band
        Image.MAX_IMAGE_PIXELS = 1000000000  # suppress DecompressionBombError for the big image
        if self.imwidth * self.imheight > self.__huge_size * self.__huge_size and \
                self.__image.tile[0][0] == 'raw':  # only raw images could be tiled
            self.__huge = True  # image is huge
            self.__offset = self.__image.tile[0][2]  # initial tile offset
            self.__tile = [self.__image.tile[0][0],  # it have to be 'raw'
                           [0, 0, self.imwidth, 0],  # tile extent (a rectangle)
                           self.__offset,
                           self.__image.tile[0][3]]  # list of arguments to the decoder
        self.__min_side = min(self.imwidth, self.imheight)  # get the smaller image side
        # Create image pyramid
        self.__pyramid = [self.smaller()] if self.__huge else [Image.open(self.path)]
        # Set ratio coefficient for image pyramid
        self.__ratio = max(self.imwidth, self.imheight) / self.__huge_size if self.__huge else 1.0
        self.__curr_img = 0  # current image from the pyramid
        self.__scale = self.imscale * self.__ratio  # image pyramide scale
        self.__reduction = 2  # reduction degree of image pyramid
        w, h = self.__pyramid[-1].size
        while w > 512 and h > 512:  # top pyramid image is around 512 pixels in size
            w /= self.__reduction  # divide on reduction degree
            h /= self.__reduction  # divide on reduction degree
            self.__pyramid.append(self.__pyramid[-1].resize((int(w), int(h)), self.__filter))
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle((0, 0, self.imwidth, self.imheight), width=0)
        self.__default()  # show image on the canvas
        self.canvas.focus_set()  # set focus on the canvas

    def smaller(self):
        """ Resize image proportionally and return smaller image """
        w1, h1 = float(self.imwidth), float(self.imheight)
        w2, h2 = float(self.__huge_size), float(self.__huge_size)
        aspect_ratio1 = w1 / h1
        aspect_ratio2 = w2 / h2  # it equals to 1.0
        if aspect_ratio1 == aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(w2)  # band length
        elif aspect_ratio1 > aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(w2 / aspect_ratio1)))
            k = h2 / w1  # compression ratio
            w = int(w2)  # band length
        else:  # aspect_ratio1 < aspect_ration2
            image = Image.new('RGB', (int(h2 * aspect_ratio1), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(h2 * aspect_ratio1)  # band length
        i, j, n = 0, 1, round(0.5 + self.imheight / self.__band_width)
        while i < self.imheight:
            print('\rOpening image: {j} from {n}'.format(j=j, n=n), end='')
            band = min(self.__band_width, self.imheight - i)  # width of the tile band
            self.__tile[1][3] = band  # set band width
            self.__tile[2] = self.__offset + self.imwidth * i * 3  # tile offset (3 bytes per pixel)
            self.__image.close()
            self.__image = Image.open(self.path)  # reopen / reset image
            self.__image.size = (self.imwidth, band)  # set size of the tile band
            self.__image.tile = [self.__tile]  # set tile
            cropped = self.__image.crop((0, 0, self.imwidth, band))  # crop tile band
            image.paste(cropped.resize((w, int(band * k) + 1), self.__filter), (0, int(i * k)))
            i += band
            j += 1
        print('\r' + 30 * ' ' + '\r', end='')  # hide printed string
        return image

    def redraw_figures(self):
        """ Dummy function to redraw figures in the children classes """
        pass

    def __default(self, *args, **kw):
        self.imscale = min(self.canvas.winfo_height() / self.imheight, self.canvas.winfo_width() / self.imwidth)
        k = self.imscale * self.__ratio  # temporary coefficient
        self.__curr_img = min((-1) * int(math.log(k, self.__reduction)), len(self.__pyramid) - 1)
        self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))
        x, y = 0, (self.canvas.winfo_height() - self.imscale * self.imheight) / 2
        self.canvas.scale('all', x, y, self.imscale, self.imscale)  # rescale all objects
        # Redraw some figures before showing image on the screen
        self.redraw_figures()  # method for child classes
        self.__show_image()

    def grid(self, **kw):
        """ Put CanvasImage widget on the parent widget """
        self.__imframe.grid(**kw)  # place CanvasImage widget on the grid
        self.__imframe.grid(sticky='nswe')  # make frame container sticky
        self.__imframe.rowconfigure(0, weight=1)  # make canvas expandable
        self.__imframe.columnconfigure(0, weight=1)

    def pack(self, **kw):
        """ Exception: cannot use pack with this widget """
        raise Exception('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        """ Exception: cannot use place with this widget """
        raise Exception('Cannot use place with the widget ' + self.__class__.__name__)

    # noinspection PyUnusedLocal
    def __scroll_x(self, *args, **kwargs):
        """ Scroll canvas horizontally and redraw the image """
        self.canvas.xview(*args)  # scroll horizontally
        self.__show_image()  # redraw the image

    # noinspection PyUnusedLocal
    def __scroll_y(self, *args, **kwargs):
        """ Scroll canvas vertically and redraw the image """
        self.canvas.yview(*args)  # scroll vertically
        self.__show_image()  # redraw the image

    def __show_image(self):
        """ Show image on the Canvas. Implements correct image zoom almost like in Google Maps """
        box_image = self.canvas.coords(self.container)  # get image area
        box_canvas = (self.canvas.canvasx(0),  # get visible area of the canvas
                      self.canvas.canvasy(0),
                      self.canvas.canvasx(self.canvas.winfo_width()),
                      self.canvas.canvasy(self.canvas.winfo_height()))
        box_img_int = tuple(map(int, box_image))  # convert to integer or it will not work properly
        # Get scroll region box
        box_scroll = [min(box_img_int[0], box_canvas[0]), min(box_img_int[1], box_canvas[1]),
                      max(box_img_int[2], box_canvas[2]), max(box_img_int[3], box_canvas[3])]
        # Horizontal part of the image is in the visible area
        if box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0] = box_img_int[0]
            box_scroll[2] = box_img_int[2]
        # Vertical part of the image is in the visible area
        if box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1] = box_img_int[1]
            box_scroll[3] = box_img_int[3]
        # Convert scroll region to tuple and to integer
        self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))  # set scroll region
        x1 = max(box_canvas[0] - box_image[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            if self.__huge and self.__curr_img < 0:  # show huge image
                h = int((y2 - y1) / self.imscale)  # height of the tile band
                self.__tile[1][3] = h  # set the tile band height
                self.__tile[2] = self.__offset + self.imwidth * int(y1 / self.imscale) * 3
                self.__image.close()
                self.__image = Image.open(self.path)  # reopen / reset image
                self.__image.size = (self.imwidth, h)  # set size of the tile band
                self.__image.tile = [self.__tile]
                image = self.__image.crop((int(x1 / self.imscale), 0, int(x2 / self.imscale), h))
            else:  # show normal image
                image = self.__pyramid[max(0, self.__curr_img)].crop(  # crop current img from pyramid
                    (int(x1 / self.__scale), int(y1 / self.__scale),
                     int(x2 / self.__scale), int(y2 / self.__scale)))
            #
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1)), self.__filter))
            imageid = self.canvas.create_image(max(box_canvas[0], box_img_int[0]),
                                               max(box_canvas[1], box_img_int[1]),
                                               anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

    def __move_from(self, event):
        """ Remember previous coordinates for scrolling with the mouse """
        canvases = [self.__imframe.overlayed, self.__imframe.imgt] + self.__imframe.imgt_preds
        for canvas in canvases:
            canvas.canvas.scan_mark(event.x, event.y)

    def __move_to(self, event):
        """ Drag (move) canvas to the new position """
        canvases = [self.__imframe.overlayed, self.__imframe.imgt] + self.__imframe.imgt_preds
        for canvas in canvases:
            canvas.canvas.scan_dragto(event.x, event.y, gain=1)
            canvas.__show_image()  # zoom tile and show it on the canvas

    def outside(self, x, y):
        """ Checks if the point (x,y) is outside the image area """
        bbox = self.canvas.coords(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            return False  # point (x,y) is inside the image area
        else:
            return True  # point (x,y) is outside the image area

    def __wheel(self, event):
        """ Zoom with mouse wheel """
        canvases = [self.__imframe.overlayed, self.__imframe.imgt] + self.__imframe.imgt_preds
        for canvas in canvases:
            x = canvas.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
            y = canvas.canvas.canvasy(event.y)
            if canvas.outside(x, y): return  # zoom only inside image area
            scale = 1.0
            # Respond to Linux (event.num) or Windows (event.delta) wheel event
            if event.num == 5 or event.delta == -120:  # scroll down, smaller
                if round(canvas.__min_side * canvas.imscale) < 30: return  # image is less than 30 pixels
                canvas.imscale /= canvas.__delta
                scale /= canvas.__delta
            if event.num == 4 or event.delta == 120:  # scroll up, bigger
                i = min(canvas.canvas.winfo_width(), canvas.canvas.winfo_height()) >> 1
                if i < canvas.imscale: return  # 1 pixel is bigger than the visible area
                canvas.imscale *= canvas.__delta
                scale *= canvas.__delta
            # Take appropriate image from the pyramid
            k = canvas.imscale * canvas.__ratio  # temporary coefficient
            canvas.__curr_img = min((-1) * int(math.log(k, canvas.__reduction)), len(canvas.__pyramid) - 1)
            canvas.__scale = k * math.pow(canvas.__reduction, max(0, canvas.__curr_img))
            #
            canvas.canvas.scale('all', x, y, scale, scale)  # rescale all objects
            # Redraw some figures before showing image on the screen
            canvas.redraw_figures()  # method for child classes
            canvas.__show_image()

    def __keystroke(self, event):
        """ Scrolling with the keyboard.
            Independent from the language of the keyboard, CapsLock, <Ctrl>+<key>, etc. """
        if event.state - self.__previous_state == 4:  # means that the Control key is pressed
            pass  # do nothing if Control key is pressed
        else:
            self.__previous_state = event.state  # remember the last keystroke state
            # Up, Down, Left, Right keystrokes
            if event.keycode in [68, 39, 102]:  # scroll right: keys 'D', 'Right' or 'Numpad-6'
                self.__scroll_x('scroll', 1, 'unit', event=event)
            elif event.keycode in [65, 37, 100]:  # scroll left: keys 'A', 'Left' or 'Numpad-4'
                self.__scroll_x('scroll', -1, 'unit', event=event)
            elif event.keycode in [87, 38, 104]:  # scroll up: keys 'W', 'Up' or 'Numpad-8'
                self.__scroll_y('scroll', -1, 'unit', event=event)
            elif event.keycode in [83, 40, 98]:  # scroll down: keys 'S', 'Down' or 'Numpad-2'
                self.__scroll_y('scroll', 1, 'unit', event=event)

    def crop(self, bbox):
        """ Crop rectangle from the image and return it """
        if self.__huge:  # image is huge and not totally in RAM
            band = bbox[3] - bbox[1]  # width of the tile band
            self.__tile[1][3] = band  # set the tile height
            self.__tile[2] = self.__offset + self.imwidth * bbox[1] * 3  # set offset of the band
            self.__image.close()
            self.__image = Image.open(self.path)  # reopen / reset image
            self.__image.size = (self.imwidth, band)  # set size of the tile band
            self.__image.tile = [self.__tile]
            return self.__image.crop((bbox[0], 0, bbox[2], band))
        else:  # image is totally in RAM
            return self.__pyramid[0].crop(bbox)

    def destroy(self):
        """ ImageFrame destructor """
        self.__image.close()
        map(lambda i: i.close, self.__pyramid)  # close all pyramid images
        del self.__pyramid[:]  # delete pyramid list
        del self.__pyramid  # delete pyramid variable
        self.canvas.destroy()


class FirstWindow(ttk.Frame):
    def __init__(self, mainframe):
        ttk.Frame.__init__(self, master=mainframe)


class SampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.figure_path = 'save'
        self._exp_frame = None
        self._dataset_frame = None
        self._figure_frame = None
        self.make_exp_frame()
        default_font = font.nametofont('TkDefaultFont')
        default_font.configure(size=18)

    def make_exp_frame(self):
        self.destroy_all()
        self._exp_frame = ExpFrame(self)
        self._exp_frame.pack()

    def make_dataset_frame(self):
        self.destroy_all()
        self._dataset_frame = DatasetFrame(self)
        self._dataset_frame.pack()

    def make_figure_frame(self):
        self.destroy_all()
        self._figure_frame = FigureFrame(self)
        self._figure_frame.pack()

    def destroy_all(self):
        if self._exp_frame is not None:
            self._exp_frame.destroy()
        if self._dataset_frame is not None:
            self._dataset_frame.destroy()
        if self._figure_frame is not None:
            self._figure_frame.destroy()


class ExpFrame(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.master.title('Select experiments')
        self.master.geometry('800x600')
        self.experiments = os.listdir(self.master.figure_path)
        self.experiments = natural_sort(self.experiments)
        self.CheckVarietys = [tk.IntVar() for _ in range(len(self.experiments))]
        self.checkbuttons = [tk.Checkbutton(self, text=self.experiments[i], variable=self.CheckVarietys[i]) for i in
                             range(len(self.experiments))]
        for checkbutton in self.checkbuttons:
            checkbutton.pack()

        start_figure = tk.Button(self, text='Make Figure', command=self.make_figure)
        start_figure.pack()

    def make_figure(self):
        self.master.figure_experiments = []
        for i in range(len(self.CheckVarietys)):
            if self.CheckVarietys[i].get() == 1:
                self.master.figure_experiments.append(self.experiments[i])
        self.master.make_dataset_frame()


class DatasetFrame(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.master.title('Select dataset')
        button_go_to_dataset = tk.Button(self, text='Go to select dataset', command=self.master.make_exp_frame, width=20)
        button_go_to_dataset.grid(row=0, column=0, sticky='w')
        self.listbox = tk.Listbox(self, selectmode='extended')
        self.listbox.insert(0, 'vimeo test')
        self.listbox.insert(1, 'vimeo septuplet')
        self.listbox.insert(2, 'SNU FILM easy')
        self.listbox.insert(3, 'SNU FILM medium')
        self.listbox.insert(4, 'SNU FILM hard')
        self.listbox.insert(5, 'SNU FILM extreme')
        self.listbox.insert(6, 'SNU FILM arb medium')
        self.listbox.insert(7, 'SNU FILM arb hard')
        self.listbox.insert(8, 'SNU FILM arb extreme')
        self.listbox.insert(9, 'xiph cropped 4k')
        self.listbox.insert(10, 'xiph resized 2k')
        self.listbox.insert(11, 'xtest single')
        self.listbox.insert(12, 'xtest multiple')
        self.listbox.grid(row=1, column=0, sticky='nsew')

        select_button = tk.Button(self, text='Select dataset', command=self.select_dataset)
        select_button.grid(row=2, column=0)

    def select_dataset(self):
        self.master.ind = self.listbox.curselection()[0]
        self.master.psnr_lines = []
        self.master.dataset_path = dataset_paths[self.master.ind]
        for i in range(len(self.master.figure_experiments)):
            with open(os.path.join(self.master.figure_path, self.master.figure_experiments[i], 'output/imgs_test',
                                   self.master.dataset_path,
                                   'results.txt')) as f:
                psnrs = f.readlines()
                if i == 0:
                    self.master.file_order_list = [psnrs[j].split(":")[0] for j in range(len(psnrs))]
                file_psnr = {psnrs[j].split(":")[0]: float(psnrs[j].split(":")[1].split(" ")[3]) for j in range(len(psnrs))}
                self.master.psnr_lines.append(file_psnr)
        self.master.file_order_list = sorted(self.master.file_order_list, key=lambda x: self.master.psnr_lines[1][x] - self.master.psnr_lines[2][x], reverse=True)
        self.master.make_figure_frame()


class FigureFrame(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.counter = 0
        self.master.title('Figure')
        self.num_frames = len(self.master.figure_experiments) + 2
        self.columns = 3
        self.rows = self.num_frames // 3
        self.width = 2100 // 3
        self.height = self.width * 270 // 480
        self.master.geometry(f'{self.width * 3 + 100}x{self.height * 3 + 100}+100+100')

        self.make_figure()

    def prev(self):
        self.counter -= 1
        if self.counter < 0:
            self.counter = len(self.master.file_order_list) - 1
        self.destroy_all()
        self.make_figure()

    def next(self):
        self.counter += 1
        if self.counter >= len(self.master.file_order_list):
            self.counter = 0
        self.destroy_all()
        self.make_figure()

    def go_to(self, _):
        key = self.entry_goto.get()
        self.counter = self.master.file_order_list.index(key)
        self.destroy_all()
        self.make_figure()

    def open_eog(self):
        img_path = self.master.file_order_list[self.counter]
        overlayed_path = os.path.join(self.master.figure_path, self.master.figure_experiments[0], 'output/imgs_test',
                                                        self.master.dataset_path, img_path,
                                                        'overlayedd.jpg')
        # os.system(f"eog {overlayed_path}")
        for i in range(len(self.master.figure_experiments)):
            imgt_pred_path = os.path.join(self.master.figure_path, self.master.figure_experiments[i], 'output/imgs_test',
                                          self.master.dataset_path, img_path,
                                          'imgt_pred.jpg')
            os.system(f"nohup eog {imgt_pred_path} &")
        imgt_path = os.path.join(self.master.figure_path, self.master.figure_experiments[0], 'output/imgs_test',
                                      self.master.dataset_path, img_path,
                                      'imgt.jpg')

    def make_figure(self):
        button_go_to_dataset = tk.Button(self, text='go to select dataset', command=self.master.make_exp_frame, width=20)
        button_go_to_exp = tk.Button(self, text='go to select exp', command=self.master.make_dataset_frame, width=20)
        button_go_to_dataset.grid(row=0, column=0, sticky='nsew')
        button_go_to_exp.grid(row=0, column=1, sticky='nsew')
        img_path = self.master.file_order_list[self.counter]
        self.overlayed_label = tk.Label(self, text=f'Overlayed {img_path}')
        self.overlayed_label.grid(row=1, column=0, columnspan=2)
        self.overlayed = CanvasImage(self, os.path.join(self.master.figure_path, self.master.figure_experiments[0], 'output/imgs_test',
                                                        self.master.dataset_path, img_path.replace('.', '_overlayed.')), 2, 0, self.width, self.height)
        self.imgt_pred_labels = [
            tk.Label(self, text=f'{self.master.figure_experiments[i]} {float(self.master.psnr_lines[i][img_path]):2f}')
            for i in range(len(self.master.figure_experiments))]
        for i, imgt_pred_label in enumerate(self.imgt_pred_labels):
            imgt_pred_label.grid(row=3 * ((i+1) // 3) + 1, column=3 * ((i + 1) % 3), columnspan=2)
        self.imgt_preds = [CanvasImage(self, os.path.join(self.master.figure_path, self.master.figure_experiments[i], 'output/imgs_test',
                                                          self.master.dataset_path, img_path.replace('.', '_pred.')), 3 * ((i+1) // 3) + 2,  3 * ((i +1)% 3 ), self.width, self.height) for i in
                           range(len(self.master.figure_experiments))]
        self.imgt_label = tk.Label(self, text=f'GT')
        self.imgt_label.grid(row=(3 * (i+1) // 3) + 1, column=3 * ((i + 2) % 3 ), columnspan=2)
        self.imgt = CanvasImage(self, os.path.join(self.master.figure_path, self.master.figure_experiments[0], 'output/imgs_test',
                                                   self.master.dataset_path, img_path), (3 * (i+1) // 3) + 1, 3 * ((i + 2) % 3 ), self.width, self.height)
        button_next = tk.Button(self, text='next', command=self.next, width=10)
        button_prev = tk.Button(self, text='prev', command=self.prev, width=10)
        self.entry_goto = tk.Entry(self, width=20)
        self.entry_goto.bind('<Return>', self.go_to)
        button_prev.grid(row=self.rows+6, column=0, sticky='nsew')
        button_next.grid(row=self.rows+6, column=1, sticky='nsew')
        self.label = tk.Label(self, text='go to:', width=10)
        self.label.grid(row=self.rows+6, column=3, sticky='nsew')
        self.entry_goto.grid(row=self.rows+6, column=4, sticky='nsew')
        self.button_open_eog = tk.Button(self, text='open in eog', command=self.open_eog)
        self.button_open_eog.grid(row=self.rows+7, column=0, sticky='nsew')

    def destroy_all(self):
        self.overlayed.destroy()
        self.imgt.destroy()
        for imgt_pred in self.imgt_preds:
            imgt_pred.destroy()


if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()
