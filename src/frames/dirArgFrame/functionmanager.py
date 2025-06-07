from functools import partial
from pathlib import Path
import dill
import pickle
import tkinter as tk
from tkinter import PhotoImage, ttk

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import Function, Integral, Symbol, latex
from sympy.abc import t

from src.base.core import TasksManager
from src.frames.BDManager.core import BDManager
from src.frames.dirArgFrame.calculateframe import CalculateFrame
from src.frames.dirArgFrame.utils import MPLButtonFrames, MPLEntryFrames, MPLMPLFrames, savefig

def edit(frame=None, btn=None, editBtn=None):
    btn.grid()
    editBtn.destroy()
    pass


class FunctionManager(TasksManager):

    N = 1
    pathfig = Path("MyLab/tempfig/")
    styleEquation={"text.usetex":False,
"font.size":12,
"text.latex.preamble":'\\usepackage{{amsmath, amssymb}}'}
    c = None
    def destroyFrame(self):
        self.c.destroy()
        self.c = None
        pass

    def setImage(self, frame, img, sep):
        fig = Figure(figsize=self.figsize)
        fig.set_facecolor("#f0f0f0")
        canvas = FigureCanvasTkAgg(fig,frame)
        strimg = f"${latex(img)}$ "+sep+" "
        axs = fig.subplots(nrows=1, ncols=1)
        axs.axis(False)
        axs.text(0.5,0.5, strimg, 
                            horizontalalignment='center',
                            verticalalignment='center')
        canvas.get_tk_widget().grid(row=0, column=0)
        pass
    def parseFun(self, n=None, btn=None, frame=None):
        name = "expr"
        self.c.pretty()
        self.expr = self.c.symexpr
        savefig(n-1, self.expr, tex=True, name=name, figsize=(3, 0.25), path=self.pathfig)
        # self.db["FUNDB"].update_by_id(str(self.expr),latex(self.expr),id=n)
        path = "fs"
        with open(f"MyLab/data/"+path+f"/f{n}.dill", "wb") as f:
            dill.dump(self.expr, f)
        with open(f"MyLab/data/"+path+f"/f{n}.dill", "rb") as f:
            dill.load(f)
        tp = self.pathfig.joinpath(f"{name}{n-1}.png")
        image = PhotoImage(file=tp)
        self.img.append(image)
        # self.img.append(image)
        b = frame.children["!button"]
        btn.destroy()
        f = frame
        editBtn = ttk.Button(frame, image=self.img[-1])
        e = partial(edit, f, b, editBtn)
        editBtn["command"] = e
        frame.children["!button"].grid_forget()
        self.destroyFrame()
        editBtn.grid()
        pass
    def createCalculateFrame(self, n, frame=None):
        if not self.c:
            self.c = CalculateFrame(self.fn, self.dfn, self.ufn)
            p = partial(self.parseFun, n, self.c.parseBtn, frame)
            self.c.parseBtn["command"] = p
            self.canvas.update()
            return
        elif not self.c.live:
            self.c = CalculateFrame(self.fn, self.dfn, self.ufn)
            p = partial(self.parseFun, n, self.c.parseBtn, frame)
            self.c.parseBtn["command"] = p
            self.canvas.update()
            return
        if self.c:
            self.c.destroy()
            self.c.destroyFrame()
            del self.c
            return
        pass
    def _resize_frame(self, event):
            # Изменяем размеры фрейма в соответствии с Canvas
            self.canvas.itemconfig(self.frame_id, width=event.width)
    def __init__(self, master,db:BDManager, countFun,  sc, exprs=None, postfix="(t, x, u)",task="") -> None:
        self.db = db
        self.task=task
        self.root = ttk.Frame(master)
        self.container = ttk.Frame(self.root)
        self.canvas = tk.Canvas(self.container, width=600, height=600)
        
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.grid_rowconfigure(0, weight=1)
        self.scrollable_frame.grid_rowconfigure(1, weight=1)
        self.scrollable_frame.grid_rowconfigure(2, weight=1)
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=1)
        self.scrollable_frame.grid_columnconfigure(2, weight=1)
        self.scrollable_frame.bind(
    "<Configure>",
    lambda e: self.canvas.configure(
        scrollregion=self.canvas.bbox("all")
    )
)

        self.scrollable_frame.grid(sticky='nsew')

        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        self.container.grid(sticky='nsew')

        self.frame_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=sc.set)

        self.canvas.grid_rowconfigure(0, weight=1)
        self.canvas.grid_columnconfigure(0, weight=1)
        self.canvas.grid_rowconfigure(1, weight=1)
        self.canvas.grid_columnconfigure(1, weight=1)
        self.canvas.grid(sticky='nsew')
        self.canvas.bind('<Configure>', self._resize_frame)
        self.sc = sc
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid(sticky='nsew')
        self.n = countFun
        self.fnbf = []
        var = t
        self.fn =  [Function(f"x_{i}")(var) for i in range(1, self.n+1)]
        self.ufn = [Function(f"u")(var)]
        self.dfn = [self.fn[i].diff(var) for i in range(self.n)]
        t0 = Symbol(str(var)+"_0")
        t_k = Symbol(str(var)+"_k")
        for i, df in enumerate(self.dfn):
            with open(f"MyLab/data/dfs/df{i+1}.dill", "wb") as f:
                dill.dump(df, f)
        style = ttk.Style()
        style.configure('TButton',background="white")
        style.configure('TFrame', background="#f0f0f0")
        try: self.pathfig.mkdir(parents=True, exist_ok=False)
        except: pass
        self.img = []
        self.timg = []
        self.fi = []
        
        plt.rcParams.update(self.styleEquation)

        # for i in range(self.n):
        #     self.db["FUNDB"].add(i+1, "0", "0")
        print(self.db["TASKINFO"].get("taskNameId"))
        # self.task =  self.db.execute(f"SELECT value FROM TASKNAME WHERE id = {self.db["TASKINFO"].get("taskNameId")[0][0]}" )
        self.fsFrame = ttk.Frame(self.scrollable_frame)


        self.calculateFramesList = [ttk.Frame(self.fsFrame)]*(self.n)
        self.intFrames = [ttk.Frame(self.fsFrame)]
        functional = Integral(Symbol('alpha_1')+Symbol('alpha_2')*Symbol('u')**2, (t, t0, t_k))
        self.intFrame = ttk.Frame(self.fsFrame)
        self.borderargsFrame = ttk.Frame(self.scrollable_frame)
        self.sep = ttk.Separator(self.scrollable_frame, orient="vertical")
        self.x0sFrames = []
        self.borderFrame = ttk.LabelFrame(self.borderargsFrame, text="Начальные и граничные условия")
        self.border0Frame = ttk.Frame(self.borderFrame)
        self.borderKFrame = ttk.Frame(self.borderFrame)
        self.borderXKFrame = ttk.Frame(self.borderKFrame)
        self.borderPKFrame = ttk.Frame(self.borderKFrame)
        self.t0Frame = ttk.LabelFrame(self.borderargsFrame, text="Начальный момент времени")
        self.typeBk = tk.BooleanVar()
        self.changeButton = ttk.Checkbutton(self.borderFrame, variable=self.typeBk, text="Задать уравнением",
                                            command=self.changeBk, state=tk.NORMAL)

        phi = Function('phi')
        for i in range(self.n):
                self.x0sFrames.append(MPLEntryFrames(self.border0Frame, Symbol(f'x_{{{i+1}0}}'), sep="=", sticky="e"))

        match self.task:
            case self.TASKS.naiskDvi.value:
                if exprs:
                    self.createMPLButtons(exprs, self.dfn, self.calculateFramesList, "f", postfix=postfix, tex=True)
                else:
                    fs = [Symbol(f'f_{i}') for i in range(1, self.n+1)]
                    self.createMPLButtons(fs,self.dfn, self.calculateFramesList, "f", postfix=postfix, tex=True)
                MPLMPLFrames(self.intFrame, "I", Image2=functional.subs({Symbol(f"alpha_1"): 1,
                                                                 Symbol(f"alpha_2"): 0}), sep="=", sticky='e')
                for i in range(self.n):
                    self.x0sFrames.append(
                        MPLEntryFrames(self.borderXKFrame, Symbol(f'x_{{{i + 1}k}}'), sep="=", sticky='e'))

            case self.TASKS.linSys.value:
                if exprs:
                    self.createMPLButtons(exprs, self.dfn, self.calculateFramesList, "f", postfix=postfix, tex=True)
                else:
                    fs = [Symbol(f'f_{i}') for i in range(1, self.n+1)]
                    self.createMPLButtons(fs,self.dfn, self.calculateFramesList, "f", postfix=postfix, tex=True)
                MPLMPLFrames(self.intFrame, "I", Image2=functional,sep=" =",figsize2=(1.75, 1), sticky='e')
                for i in range(self.n):
                    self.x0sFrames.append(
                        MPLEntryFrames(self.borderXKFrame, Symbol(f'x_{{{i + 1}k}}'), sep="=", sticky='e'))
                self.additionalFrame = ttk.LabelFrame(self.borderargsFrame, text="Дополнительные параметры")
                for i in range(1,3):
                    self.x0sFrames.append(MPLEntryFrames(self.additionalFrame, Symbol(f"alpha_{i}"), sep="="))
                pass
            case self.TASKS.enOpt.value:
                if exprs:
                    self.createMPLButtons(exprs, self.dfn, self.calculateFramesList, "f", postfix=postfix, tex=True)
                else:
                    fs = [Symbol(f'f_{i}') for i in range(1, self.n+1)]
                    self.createMPLButtons(fs,self.dfn, self.calculateFramesList, "f", postfix=postfix, tex=True)
                MPLMPLFrames(self.intFrame, "I", Image2=functional.subs({Symbol(f"alpha_{i}"):1 for i in range(1, 3)}), sep="=", sticky='e')
                for i in range(self.n):
                    self.x0sFrames.append(
                        MPLEntryFrames(self.borderXKFrame, Symbol(f'x_{{{i + 1}k}}'), sep="=", sticky='e'))
                pass
            case self.TASKS.metodDyn.value:
                if exprs:
                    self.createMPLButtons(exprs[:-1], self.dfn, self.calculateFramesList, "f", postfix=postfix, tex=True)
                    self.createMPLButtons([exprs[-1]], [Function(f"f_{self.n + 1}")],
                                          self.calculateFramesList, name="f", tex=True, figsize2=(0.3, 0.6))
                else:
                    fs = [Symbol(f'f_{i}') for i in range(1, self.n + 1)]
                    self.createMPLButtons(fs, self.dfn, self.calculateFramesList, "f", postfix=postfix, tex=True)
                    self.createMPLButtons([Function(f"f_{self.n+1}")(t)], [Function(f"f_{self.n+1}")], self.calculateFramesList, name="f", tex=True, figsize2=(0.3, 0.6))
                MPLMPLFrames(self.intFrame, "I", Image2=functional.subs({Symbol(f"alpha_1"):0,
                                                                        Symbol(f"alpha_2"):1})+Symbol("lambda")*Function(f"f_{{{self.n+1}}}")(Symbol("T")), sep="=", sticky='e')
        self.createMPLButtons([phi], [phi(Symbol('t'))], [self.borderPKFrame], sep="=", name="phi")
        self.x0sFrames.append(MPLEntryFrames(self.t0Frame, Symbol("t_0"), sep="=", sticky='e'))

        self.gridFun()
    def updateFun(self, newFs, i):
        image = PhotoImage(file=newFs)
        self.img[i] = image
        self.fnbf[i].entry["image"] = image
        self.fnbf[i].entry["state"] = tk.DISABLED
    def gridFun(self):
        self.fsFrame.grid(sticky='nsew')
        self.intFrame.grid_columnconfigure(0, weight=1)
        self.fsFrame.grid_columnconfigure(0, weight=1)
        self.intFrame.grid(row=1, column=0,sticky='nsew')
        self.borderargsFrame.grid_rowconfigure(2, weight=1)
        self.borderargsFrame.grid_columnconfigure(2, weight=1)
        self.borderargsFrame.grid(row=0, column=2,sticky='nsew')
        self.scrollable_frame.grid_rowconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(2, weight=1)
        self.sep.grid_rowconfigure(0, weight=1)
        self.sep.grid_columnconfigure(0, weight=1)
        self.sep.grid(row=0, column=1, sticky='nse', padx=10,ipadx=10)
        self.t0Frame.grid(row=1, column=2, sticky='nsew')
        self.borderFrame.grid(row=0, column=2,sticky='nsew')
        self.border0Frame.grid(sticky='w')
        self.borderKFrame.grid()
        self.borderXKFrame.grid()
        self.borderPKFrame.grid()
        self.borderPKFrame.grid_forget()

        match self.task:
            case self.TASKS.enOpt.value:
                self.changeButton.grid()
            case self.TASKS.naiskDvi.value:
                self.changeButton.grid()
            case self.TASKS.linSys.value:
                self.additionalFrame.grid(row=2, column=2, sticky='nsew')
                self.changeButton.grid()
            case self.TASKS.metodDyn.value:
                self.changeButton.grid()
                self.changeButton.grid_forget()
        pass

    def changeBk(self):
        self.db["TASKINFO"].update('Isbkfun', self.typeBk.get())
        print(self.db["TASKINFO"].get("Isbkfun")[0][0])
        if not self.typeBk.get():
            if self.borderPKFrame.winfo_ismapped():
                self.borderPKFrame.grid_forget()
            self.borderXKFrame.grid()
            print(len(self.x0sFrames))
        else:
            if self.borderXKFrame.winfo_ismapped():
                self.borderXKFrame.grid_forget()
            self.borderPKFrame.grid()
            pass
        pass

    def createMPLButtons(self, lsymbols:list[Symbol],rsymbols:list[Symbol], frames:list[ttk.Frame], name:str, sep="=", figsize=(1, 0.25), side="left",tex=False, postfix="", figsize2=(0.6, 0.6)):
        for i, frame in enumerate(frames):
            frame.grid_rowconfigure(i, weight=1)
            frame.grid()

        for i, arg in enumerate(zip(lsymbols, rsymbols)):
            self.fi.append(arg[0])
            savefig(i, self.fi[-1], name=name, tex=tex, figsize=figsize, path=self.pathfig, postfix=postfix)
            tp = self.pathfig.joinpath(f"{name}{i}.png")
            image = PhotoImage(file=tp)
            self.img.append(image)
            self.fnbf.append(MPLButtonFrames(frames[i], self.N, arg[1],buttonImg=self.img[-1], sep=sep, figsize=figsize2, side=side, command=self.createCalculateFrame, sticky='e'))
            self.N+=1
    def applyFun(self):
        pass
    def close(self):
        for im in self.pathfig.iterdir():
            im.unlink()
        # try:
        #     self.db["FUNDB"].delete()
        # except:
        #     pass
        self.root.destroy()
        self.sc.destroy()

