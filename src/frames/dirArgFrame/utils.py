from functools import partial
from pathlib import Path
from tkinter import ttk

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import latex

# with open("configs/app_setting.toml", "rb") as f:
#             d = tomllib.load(f)

class MPLEnityFrames:
    n = 0
    m = 0
    def __init__(self, master, Image, side="left", figsize=(0.3, 0.3),sep="",**kwargs) -> None:
        self.listFrames = []
        self.imgButton = []
        self.root = master
        self.mainframe = ttk.Frame(self.root)
        # self.mainframe.grid_rowconfigure(0, weight=1)
        # self.mainframe.grid_columnconfigure(0, weight=1)
        self.mainframe.grid()
        self.figsize = figsize
        self.side = side
        imFrame = ttk.Frame(self.mainframe)
        imFrame.grid_columnconfigure(0, weight=1)
        imFrame.grid(row=0, column=0)
        if "columnconfigure" in kwargs:
            self.mainframe.grid_columnconfigure(kwargs["columnconfigure"], weight=1)
        if "sticky" in kwargs:
            self.mainframe.grid_configure(sticky=kwargs["sticky"])
        self.setImage(imFrame, Image,self.figsize, sep)
        self.enityFrame = ttk.Frame(self.mainframe)
        self.enityFrame.grid(row=0, column=1)
    def create(self):
        pass
    def setImage(self,frame,img,figsize, sep="", **kwargs):
        ha = 'center'
        va = 'center'
        for key, value in kwargs.items():
            if key == 'horizontalalignment':
                ha = kwargs[key]
            if key == 'verticalalignment':
                va = kwargs[key]
        fig = Figure(figsize=figsize)
        fig.set_facecolor("#f0f0f0")
        canvas = FigureCanvasTkAgg(fig,frame)
        limg = latex(img)
        if limg.find(r"\limits"):
            limg = limg.replace(r"\limits", "")
        strimg = r"$"+limg+"$"+sep+" "
        axs = fig.subplots(nrows=1, ncols=1)
        axs.axis(False)
        axs.set_xlim(0, 1)
        axs.set_ylim(0, 1)
        axs.text(0.5,0.5, strimg,
                            horizontalalignment=ha,
                            verticalalignment=va)
        w = canvas.get_tk_widget()
        w.grid_columnconfigure(0, weight=1)

        w.grid(row=0, column=0, sticky='nsew')
        pass

    def get(self):
        d = []
        for ent in self.listFrames:
            d.append(ent.children['!entry'].get())
        if len(d) == 1:
            return [d[0]]
        return d
class MPLEntryFrames(MPLEnityFrames):
    def __init__(self, master, Image, side="left", figsize=(0.3, 0.3), sep=None, **kwargs) -> None:
          super().__init__(master, Image, side, figsize, sep, **kwargs)
          self.create(**kwargs)
    def text(self):
         return self.enity.get()
    def create(self,**kw):
        self.enity = ttk.Entry(self.enityFrame)
        if "sticky" in kw:
            self.enity.grid_configure(sticky=kw['sticky'])
        self.enity.grid()
class MPLButtonFrames(MPLEnityFrames):
    img = []
    def __init__(self, master,id, Image, side="left", figsize=(0.3, 0.3),buttonImg=None,command=None, sep=None, **grid_opts) -> None:
        self.img.append(str(buttonImg))
        super().__init__(master, Image, side, figsize, sep, **grid_opts)
        self.create(id, buttonImg, command)
    def create(self, id, buttonImg, command):
        c = partial(command, id, self.enityFrame)
        self.entry = ttk.Button(self.enityFrame, image=self.img[-1], command=c)
        self.entry.grid()
    def defaultImg(self):
        self.entry["image"] = self.img
    pass
class MPLMPLFrames(MPLEnityFrames):
    def __init__(self, master, Image, side="left", figsize=(0.75, 0.75),figsize2=(1.65,1), sep="", Image2=None, **kwarg):
          super().__init__(master, Image, side, figsize, sep, **kwarg, horizontalalignment='center')
          self.create(Image2, figsize2)
    def create(self, image, figsize):
        self.setImage(self.enityFrame, image, figsize, sep="", horizontalalignment='center')
def savefig(*args, tex=False, **kwargs):
        p = Path(".")
        postfix = ""
        for key, val in kwargs.items():
            if key == "name":
                name = val
            if key == "figsize":
                figsize = val
            if key == "postfix":
                postfix = val
            if key == "path":
                p = p.joinpath(val)
        fign, axsn = plt.subplots(1, 1, figsize=figsize)
        axsn.axis(False)
        if tex:
            t = latex(args[1])
        else:
            t = str(args[1])
        axsn.text(0.5,0.5, f"${t}{postfix}$", 
                            horizontalalignment='center',
                            verticalalignment='center',)
        p = p.joinpath(Path(f"{name}{args[0]}.png"))
        fign.savefig(p)
        plt.close()