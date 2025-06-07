import tkinter
from functools import partial
from pathlib import Path
import re

from tkinter import END, PhotoImage, StringVar, ttk

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import latex, parse_expr, simplify

from src.frames.dirArgFrame.utils import savefig

class CalculateFrame(tkinter.Tk):
    live=True
    pathfig = Path("MyLab/tempfig/")
    operationsList = ["+","-","*","/", r"**", "."]
    local = {"^":"**", "{":"(", "}":")", "[":"(", "]":")"}
    abc =  ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    abcFrame = None
    alphabetFrame = None
    specialFuncList = ["sin", "cos", "tan", "cot","exp","log"]
    numList = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    scobeList = ["(", ")"]
    def delete(self):
        if self.history:
            self.lastsym = self.history.pop()        
            self.entry.delete(len(self.history[0:-len(self.lastsym)]), END)
            self.expr = ''.join(self.history)
        else: 
            self.symexpr = 0
            self.lastsym = None

        self.update()
    def erase(self):
        self.entry.delete(0, END)
        self.history=[]
        self.symexpr = 0
        self.lastsym = None
    def add(self, sym=None):
        if (str(sym) == '.') and (self.history==[] or self.lastsym == "("):
            self.lastsym = '.'
            self.history.append(str(sym))
            self.expr += str(sym)
            self.update()
            return
        if (str(sym) == '-') and (self.history==[] or self.lastsym == "("):
            self.history.append(str(sym))
            self.expr += str(sym)
            self.update()
            return
        if str(sym) in self.operationsList and (self.lastsym in self.operationsList or not self.lastsym):
            return
        if sym in self.fn and (self.lastsym in self.fn or  self.lastsym in self.abc or self.lastsym in self.numList):
            self.lastsym = sym
            self.history.append("*")
            self.history.append(str(sym))
            self.expr +="*"+str(sym)
            self.update()
            return
        if self.lastsym in self.fn and (sym in self.fn or  sym in self.abc or sym in self.numList):
            self.lastsym = sym
            self.history.append("*")
            self.history.append(str(sym))
            self.expr +="*"+str(sym)
            self.update()
            return
        if sym in self.abc and (self.lastsym in self.abc or self.lastsym in self.numList):
            self.lastsym = str(sym)
            self.history.append("*")
            self.history.append(str(sym))
            self.expr+="*"+str(sym)
            self.update()
            return
        if (self.lastsym in self.numList or self.lastsym in self.abc) and sym in self.specialFuncList:
            self.lastsym = str(sym)
            self.history.append("*")
            self.history.append(str(sym))
            self.expr+="*"+str(sym) +"("
            self.update()
            return
        if sym in self.scobeList:
            self.lastsym = str(sym)
            self.history.append(str(sym))
            self.expr += str(sym)
            self.update()
            return
        if sym in self.numList:
            self.lastsym = str(sym)
            self.history.append(str(sym))
            self.expr += str(sym)
            self.update()
            return
        if sym in self.operationsList:
            self.lastsym = sym
            if sym == "^":
                self.history.append("**")
                self.history.append("(")
                self.expr+=str(sym+"{")
            else:
                self.history.append(str(sym))
                self.expr+=str(sym)
            self.update()
            return
        if sym in self.fn or sym in self.uprFun:
            self.lastsym = sym
            self.history.append(str(sym))
            self.expr+=str(sym)
            self.update()
            return
        if sym in self.specialFuncList:
            self.lastsym = sym
            self.history.append(str(sym))
            self.history.append("(")
            self.expr+=str(sym)+"("
            self.update()
        if sym in self.abc:
            self.lastsym = sym
            self.history.append(str(sym))
            self.expr+=str(sym)
            self.update()
            return
        if sym in self.uprFun:
            self.lastsym = sym
            self.history.append(str(sym))
            self.expr+=str(sym)
            self.update()
            return
    def destroyFrame(self):
        self.mainframe.destroy()
        pass
    def destroy(self):
        self.live = False
        super().destroy()
        self.history = []
        self.expr = ""
        self.lastsym = None
        pass
    def __init__(self, fn, dfn, ufn) -> None:
        self.live=True

        super().__init__()
        self.maxsize(500, 450)
        self.minsize(500, 450)
        self.geometry("500x450")
        self.fn = fn
        self.uprFun = ufn
        self.symexpr=0
        self.mainframe = ttk.Frame(self)
        self.mainframe.grid()
        self.img = []
        self.timg = []
        self.history = []
        self.expr = ""
        self.lastsym = None
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        self.texvar = StringVar(self.mainframe)
        fig = Figure((5, 1))
        self.axs = fig.subplots(1, 1)
        plt.rcParams.update({"text.usetex":False,
"font.size":12,
"text.latex.preamble":'\\usepackage{{amsmath, amssymb}}'})
        canvaFrame = ttk.Frame(self.mainframe)
        self.entry = ttk.Entry(canvaFrame, textvariable=self.texvar)
        self.entry.pack(fill="x")
        self.canvas = FigureCanvasTkAgg(fig, canvaFrame)
        self.canvas.get_tk_widget().pack()
        self.axs.axis(False)
        canvaFrame.grid(row=0, column=0)
        self.btnFrame = ttk.Frame(self.mainframe)
        self.funFrame = ttk.Frame(self.btnFrame)
        self.funFrame.grid()
        self.btnFrame.grid_rowconfigure(1, weight=1)
        self.btnFrame.grid()
        self.opandscobeFrame = ttk.Frame(self.funFrame)
        self.opandscobeFrame.grid(row=0, column=4)
        self.opFrame = ttk.Frame(self.opandscobeFrame)
        self.numFrame = ttk.Frame(self.funFrame)
        self.numFrame.grid(row=0, column=2)
        self.opFrame.grid(row=0, column=0,sticky='n')
        self.scobeFrame = ttk.Frame(self.opandscobeFrame)
        self.scobeFrame.grid(row=1, column=0,sticky='n')
        self.butFrame = ttk.Frame(self.funFrame)
        self.butFrame.grid()


        self.spFrame = ttk.Frame(self.funFrame)
        self.spFrame.grid(row=0, column=0,sticky='n')
        self.fsFrame = ttk.Frame(self.btnFrame)
        self.fFrame = ttk.Frame(self.fsFrame)
        self.btnFrame.grid_columnconfigure(0, weight=1)
        self.fsFrame.grid(row=1,column=0)
        self.fFrame.grid(row=0,column=0)
        self.uFrame = ttk.Frame(self.fsFrame)
        self.uFrame.grid(row=0, column=1)
        ttk.Separator(self.funFrame, orient='vertical').grid(row=0, column=1,sticky='nsew',padx=5)
        ttk.Separator(self.funFrame, orient='vertical').grid(row=0, column=3, sticky='nsew',padx=5)
        self.delFrame = ttk.Frame(self.btnFrame)
        self.delFrame.grid(row=2,column=0)
        btnEr = ttk.Button(self.delFrame, text="C",command=self.erase)
        btnEr.grid(row=0, column=0)
        btnDel = ttk.Button(self.delFrame, text="<", command=self.delete)
        btnDel.grid(row=0, column=1)
        for i, op in enumerate(self.operationsList):
            self.createButton(self.opFrame, i, op, "op", self.add, self.operationsList, tex=True, n = (2, 2))
        for i, sp in enumerate(self.specialFuncList):
            self.createButton(self.spFrame, i, sp, "sp", self.add, self.specialFuncList, tex=True, n=(2,2))
        for i, num in enumerate(self.numList):
            self.createButton(self.numFrame, i, num, "num", self.add, self.numList, tex=True)
        for i, num in enumerate(self.scobeList):
            self.createButton(self.scobeFrame, i, num, "scobe", self.add, self.scobeList, n = (2, 2))
        for i, f in enumerate(self.fn):
            self.createButton(self.fFrame, i, f, "fs", self.add, self.fn, n=(len(self.fn), len(self.fn)), tex=True)
        for i, f in enumerate(self.uprFun):
            self.createButton(self.uFrame, i, f, "us", self.add, self.uprFun, n=(len(self.uprFun), len(self.uprFun)), tex=True)
        self.optFrame = ttk.Frame(self.mainframe)
        self.optFrame.grid()
        self.abcFrame = ttk.Frame(self.btnFrame)
        self.abcBtnFrame = ttk.Frame(self.abcFrame)
        # self.abcBtn = ttk.Button(self.abcBtnFrame,text="abc", command=self.showAbc)
        plt.rcParams.update(plt.rcParamsDefault)
        
        self.simpleBtn = ttk.Button(self.optFrame, text="Update", command=self.pretty)
        self.simpleBtn.grid(row=0, column=0)
        self.simplifyBtn = ttk.Button(self.optFrame, text="Simplify", command=self.simplifyExpr)
        self.simplifyBtn.grid(row=0, column=1)
        self.texvar.trace_add("write",self.write)
        self.parseBtn = ttk.Button(self.mainframe, text="Ok")
        self.parseBtn.grid()
        pass
    # def showAbc(self):
    #     if self.alphabetFrame is None:
    #         self.alphabetFrame = ttk.Frame(self.abcFrame)
    #         self.alphabetFrame.grid()
    #         for i, a in enumerate(self.abc):
    #             self.createButton(self.alphabetFrame, i, a, "abc", self.add, self.abc, n=(len(self.abc), 5), tex=False)
    #         return
    #     if self.alphabetFrame is not None:
    #         self.alphabetFrame.destroy()
    #         self.alphabetFrame = None
    #         return
    #     pass
    def simplifyExpr(self):
        
        self.symexpr = parse_expr(''.join(self.history), local_dict=self.local)
        self.symexpr =simplify(self.symexpr)
        self._up()
        pass 
    def pretty(self):
        self.symexpr = parse_expr(''.join(self.history), local_dict=self.local)
        self._up()
    def _up(self):
        self.expr = latex(self.symexpr)
        s = '({0})'.format('|'.join(map(re.escape, sorted(self.operationsList, reverse=True)))).replace("^", "**")
        self.history = re.split(s, str(self.symexpr).replace("^", "**"))
        self.update()
        pass
    def update(self):
        self.texvar.set(self.expr)
        pass
    def createButton(self, master, i, op, name, func, ops, n=(3,3), tex=True):
        if not ((self.pathfig/Path(name+str(i)+".png"))).exists():
            savefig(i, op, tex=tex, name=name, path=self.pathfig, figsize=(0.35, 0.35))
        tp = self.pathfig.joinpath(f"{name}{i}.png")
        image = PhotoImage(file=tp, master=self.mainframe)
        add_special_sym = partial(func, op)
        self.img.append(image)
        btn = ttk.Button(master, command=add_special_sym)
        btn["image"] = self.img[-1]
        btn.grid(row=i//(n[1]), column=(i)%n[1])
        pass
    def write(self,*args):
            self.axs.clear()
            self.axs.axis(False)
            self.expr = self.texvar.get()
            
            match self.texvar.get():
                case "":
                    self.axs.text(0.5,0.5, '', 
                            horizontalalignment='center',
                            verticalalignment='center',
                            )
                case _:
                    self.axs.text(0.5,0.5, f"$ = {self.texvar.get()} $", 
                            horizontalalignment='center',
                            verticalalignment='center',)
                    
            try: self.canvas.draw()
            except ValueError:
                pass
            pass
    def __del__(self):
        self.history = []
        self.expr = ""
        self.lastsym = None
        pass
    # def createSimpleCalc(self):
    #     pass
    # def createCustomCalc(self):
    #     pass
    pass