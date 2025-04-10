import json
from functools import partial
from pathlib import Path
import sqlite3
from tkinter import Tk, ttk, BooleanVar

import sympy as sp

from src.base.core import TasksManager
from src.frames.BDManager.core import BDManager
from src.frames.dirPlotFrame.pontryagin import SolversAgent
from src.frames.dirArgFrame.utils import MPLEntryFrames

from matplotlib import pyplot as plt
import matplotlib as mpl

from sympy import latex
from src.base.core import LogicFrame

import dill

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import  tkinter as tk

STEPS = ["FUNS","BS"]
EXPERIMENTSSTEP = ["HAMILT","U","FUNCT", "PSI0"]


class PlotFrame(LogicFrame, TasksManager):
    """
    Фрейм для работы с поиском решения задачи

    Argumets
    --------
    sysFrame : None|FigureCanvasTkAgg
        Фрейм для отображение вводимой пользователем системы.
    plotCanvas: None|Tk
        Окно с графиком процессов.
    """
    id=3
    name = "Поиск решения"
    plotCanvas = None
    sysCanvas = None
    live=False
    MAXERRORINPUT = 5
    CURRENTERRORCOUNT = 0

    kwargs = {"balancePoint":None, "balanceFunction":None, "balanceLine":None}
    def _resize_frame(self, event):
            # Изменяем размеры фрейма в соответствии с Canvas
            self.canvas.itemconfig(self.frame_id, width=event.width)
    def __init__(self, master, db:BDManager, nextFrame=None):
        """
        Paramters
        ---------
        master:ttk.Frame

        db:BDManager

        """
        super().__init__(master, nextFrame, use_button = False)
        self.plotFrame = None
        self.psiFrame = None
        self.psi0Frame = None
        self.root = master
        self.task = ""
        self.db = db
        self.dictExperiments = {}
        self.dictAdditional = {}
        self.countExperements = 0
        self.plotList = []
        self.fs = []

        self.sysFrame = ttk.Frame(self.frame)
        self.plotFrame = ttk.Frame(self.frame)
        self.psiFrame = ttk.Frame(self.frame)
        self.addExperimentButton = ttk.Button(self.psiFrame, text="Добавить эксперимент", command=self.addExperiment)
        self.removeExperimentButton = ttk.Button(self.psiFrame, text="Убрать эксперимент", command=self.removeExperiment, state=tk.DISABLED)
        self._psiFrame = ttk.Frame(self.psiFrame)
        self.warningLabel = ttk.Label(self.frame, text="Пожалуйста, выберите начальные значения", justify="center")
        self.solvebtn = ttk.Button(self.frame, text = "Solve", command=self.solve)

        self.figSys = Figure(figsize=(5, 5))
        self.figSys.set_facecolor(u'#f0f0f0')

        self.axsSys = self.figSys.subplots(nrows=1, ncols=1)
        self.sysCanvas = FigureCanvasTkAgg(self.figSys, self.sysFrame)

    def removeExperiment(self):
        if self.countExperements>1:
            self.dictExperiments[self.countExperements][1].destroy()
            self.dictExperiments.pop(self.countExperements)
            self.countExperements-=1
            if self.countExperements == 1:
                self.removeExperimentButton["state"] = tk.DISABLED
        pass
    def addExperiment(self):
        self.psi0 = []
        self.countExperements += 1
        self.experimentFrame = ttk.LabelFrame(self.scrollable_frame, text=f"Эксперимент №{self.countExperements}")
        self.psi0Frame = ttk.LabelFrame(self.experimentFrame, text="Начальные значения сопряженной системы")
        self.l = BooleanVar()
        self.tkCurrent = BooleanVar()
        match self.task:
            case self.TASKS.metodDyn.value:
                self.psi0Frame["text"] = "Дополнительные параметры"
                self.psi0.append(
            MPLEntryFrames(self.psi0Frame, sp.Symbol('lambda'), sep='=', sticky='nsew', columnconfigure=0))
            case _:
                for i in range(1, self.n+1):
                    self.psi0Frame["text"] = "Начальные значения сопряженной системы"
                    self.psi0.append(MPLEntryFrames(self.psi0Frame, sp.Symbol(f'psi{i}'), sep='=', sticky='nsew', columnconfigure=0))
                ttk.Separator(self.psi0Frame, orient="horizontal").grid(column=0, sticky='ew')
                ttk.Radiobutton(self.psi0Frame,text="Ограниченное управление", value=True,variable=self.l).grid(sticky='w')
                ttk.Radiobutton(self.psi0Frame,text="Неограниченное управление", value=False,variable=self.l).grid(sticky='w')
                ttk.Checkbutton(self.psi0Frame, text="Закрепить конечное время", variable=self.tkCurrent).grid()

        ttk.Separator(self.psi0Frame, orient="horizontal").grid(column=0, sticky='ew')

        self.psi0.append(MPLEntryFrames(self.psi0Frame, sp.Symbol('t_k'), sep='=',sticky='nsew', columnconfigure=0))
        self.dictAdditional[self.countExperements] = [self.l, self.tkCurrent]
        self.psi0Frame.grid(row=0,column=0, sticky='n')
        self.psi0Frame.grid_columnconfigure(0, weight=1)
        self.psi0Frame.grid_columnconfigure(1, weight=1)
        self.experimentFrame.grid(sticky='nsew')
        self.dictExperiments[self.countExperements] = (self.psi0, self.experimentFrame)
        if self.countExperements>1:
            self.removeExperimentButton["state"] = tk.ACTIVE
        pass
    def grid(self, row, col, rowspan=1, colspan=1):
        super().grid(row, col)
        self.sysFrame.grid(row=0, column=0, sticky='nsew')
        self.sysFrame.grid_rowconfigure(0, weight=1)
        self.sysFrame.grid_columnconfigure(0, weight=1)

        self.plotFrame.grid(row=0, column=1, sticky='nsew')
        self.plotFrame.grid_rowconfigure(0, weight=1)
        self.plotFrame.grid_columnconfigure(0, weight=1)

        self.warningLabel.grid_columnconfigure(0, weight=1)
        self.warningLabel.grid(row=1,column=0,sticky='ew')
        self.psiFrame.grid_columnconfigure(0, weight=1)
        self.psiFrame.grid(row=2,column=0,sticky='nsew')
        self.solvebtn.grid(sticky='ew',row=3,column=0)
        self._psiFrame.grid_columnconfigure(0, weight=1)
        self._psiFrame.grid_rowconfigure(0, weight=1)
        self._psiFrame.grid(sticky='nsew')
        self.addExperimentButton.grid(sticky='nsew')
        self.removeExperimentButton.grid(sticky='nsew')
        self.sysCanvas.get_tk_widget().grid(row=0, column=0)
        self.sysFrame.grid_columnconfigure(0, weight=1)
        self.sysFrame.grid_rowconfigure(1, weight=1)
    def update(self):
        # self.fig.clear()
        self.axsSys.clear()
        self.axsSys.axis(False)
        self.axsSys.text(0.5,0.5, f"{self.to_sys_latex(self.dfs, self.fs)}",
                        horizontalalignment='center',
                        verticalalignment='center')
        self.sysCanvas.draw()
        self.sc.grid(row=0, column=1, sticky='nsew')
        self.sc.grid_columnconfigure(0, weight=1)
        self.sc.grid_rowconfigure(0, weight=1)
    def show(self):
        self.task = self.db["TASKINFO"].get("value")[0][0]
        print(self.task)
        self.n = self.db["TASKINFO"].get("N")[0][0]
        self.isBkFun = self.db["TASKINFO"].get("Isbkfun")[0][0]
        match self.task:
            case self.TASKS.metodDyn.value:
                self.fs = [0]*(self.n+1)
                self.dfs = [0]*self.n
            case _:
                self.fs = [0] * self.n
                self.dfs = [0] * self.n
        p = Path("MyLab/data/fs/").glob('**/*')
        if self.isBkFun == "True":
            for i, file in enumerate(p):
                    with open(file, "rb") as f:
                        if i < self.n:
                            self.fs[i]=dill.load(f)
                        else:
                            self.phi = dill.load(f)
        else:
            for i, file in enumerate(p):
                with open(file, "rb") as f:
                    self.fs[i]=dill.load(f)
        d = Path("MyLab/data/dfs/").glob('**/*')
        for i, file in enumerate(d):
            with open(file, "rb") as f:
                self.dfs[i]=dill.load(f)


        self.update()

        
        pass
    def setPsi0(self):
        self.task = self.db["TASKINFO"].get("value")[0][0]
        if self._psiFrame:
            self._psiFrame.destroy()
            self._psiFrame = ttk.Frame(self.psiFrame)
            self._psiFrame.grid()
        self.container = ttk.Frame(self._psiFrame)
        
        self.psiFrame.grid_rowconfigure(0, weight=1)
        self.psiFrame.grid_rowconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(self.container)
        self.canvas.grid(sticky='nsew')

        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        self.container.grid(sticky='nsew')
        
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.grid_rowconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid(row=0, column=0,sticky='nsew')

        self.frame_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        self.sc = ttk.Scrollbar(self.container, orient="vertical")
        self.sc["command"] = self.canvas.yview
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        self.canvas.configure(yscrollcommand=self.sc.set)
        self.canvas.grid_columnconfigure(0, weight=1)

        self.addExperiment()

        self.sc.grid(row=0, column=1, sticky='nse', padx=10)
        self.sc.grid_columnconfigure(1, weight=1)
        self.sc.grid_rowconfigure(0, weight=1)

        self.canvas.bind('<Configure>', self._resize_frame)

        pass
    def to_bs_latex(self, usetex=False):
        l = ""
        xs = [sp.Function(f'x_{i + 1}') for i in range(self.n)]
        match self.task:
            case  self.TASKS.metodDyn.value:
                t0 = self.db["BORDERDB"].get()[self.n][1]
                bd0 = [float(i[1]) for i in self.db["BORDERDB"].get()[:self.n]]
                if usetex:
                    l = r"\begin{cases} "
                    for i, x in enumerate(xs):
                        l += sp.latex(sp.Eq(x(t0), bd0[i])) + r"\\"
                    l += r"\end{cases}"
                    pass
                else:
                    for i in range(self.n):
                        l += f"$x_{i + 1}({t0}) = {bd0[i]}$" + "\n"
            case _:

                t0 = self.db["BORDERDB"].get()[-1][1]
                bd0 = [float(i[1]) for i in self.db["BORDERDB"].get()[:self.n]]
                if self.isBkFun:
                    if usetex:
                        l = r"\begin{cases} "
                        for i, x in enumerate(xs):
                            l += sp.latex(sp.Eq(x(t0), bd0[i])) + r"\\"
                        l+=sp.latex(sp.Eq(self.phi,0))
                        l += r"\end{cases}"

                    else:
                        for i in range(self.n):
                            l += f"$x_{i + 1}({t0}) = {bd0[i]}$" + "\n"
                        l+="$"+sp.latex(self.phi.subs(sp.Symbol('t'), sp.Symbol('t_k')))+" = 0$\n"
                else:
                    bdk = [float(i[1]) for i in self.db["BORDERDB"].get()[self.n:2 * self.n]]
                    if usetex:
                        l = r"\begin{cases} "
                        for i, x in enumerate(xs):
                            l += sp.latex(sp.Eq(x(t0), bd0[i])) + r" \quad " + sp.latex(sp.Eq(x(sp.Symbol('t_{k}')), bdk[i])) + r"\\"
                        l += r"\end{cases}"
                    else:
                        for i in range(self.n):
                            l += f"$x_{i + 1}({t0}) = {bd0[i]}$" + " " + f"$x_{i + 1}(t_k) = {bdk[i]}$" + "\n"

        return l
    def to_fs_latex(self, dfs, fs, usetex=False):
        if usetex:
                l = r"\begin{cases} "
                for eq in zip(dfs, fs):
                    l += r"".join(latex(eq[0])) +r"="+ r"".join(latex(eq[1]))+r"\\ "
                l+=r"\end{cases}\\ "
                return f"{l}\n"
        else:
            l = ""
            for eq in zip(dfs, fs):
                l += "$" + latex(eq[0]) + r"=" + latex(eq[1]) + "$" + "\n"
            return l

    def to_sys_latex(self, lhs, rhs,usetex=False):
        if self.n == 1:
            return f"{str(lhs[0])} = {latex(rhs[0])}"
        print(self.db["BORDERDB"].get())
        l = ""
        l+=self.to_fs_latex(lhs, rhs, usetex)+self.to_bs_latex(usetex)
        print(l)
        return l
    def close(self):
        pass
    def solve(self):
        self.bs = [float(i[1]) for i in self.db["BORDERDB"].get()]
        self.kwargs.update({opt:val for opt, val in zip(self.db["TASKINFO"].values, *self.db["TASKINFO"].get())})
        if self.isBkFun:
            self.kwargs['phi'] = self.phi
        def noneCanvas(i):
            plt.close(self.figList[i-1])
            self.canvasList[i - 1] = None
            self.plotList[i - 1].destroy()
            self.plotList[i - 1] = None

        for plot in  self.plotList:
            if plot:
                plot.destroy()
        self.plotList = [None]*self.countExperements
        self.figList = []
        self.canvasList = [None]*self.countExperements
        self.dataDict = {key: value for key, value in zip(STEPS, (self.to_fs_latex(self.dfs, self.fs, True),self.to_bs_latex(True)))}
        self.dataDict["EXPERIMENTS"] = {i: {} for i in range(1, self.countExperements + 1)}
        for i in range(1, self.countExperements+1):
            self.kwargs['tkCurrent'] = self.dictAdditional[i][1].get()
            self.kwargs['limit'] = self.dictAdditional[i][0].get()
            try:
                p = SolversAgent(self.task, self.dfs, self.fs, self.bs, psi0=[float(i.text()) for i in self.dictExperiments[i][0]], **self.kwargs)
            except:
                p = SolversAgent(self.task, self.dfs, self.fs, self.bs)
            try:
                solver = p.solver()
                self.fig, self.axs = solver.plot.getFig()
                self.dataDict["EXPERIMENTS"][i] = {key: value for key, value in zip(EXPERIMENTSSTEP, solver.get_data())}
                if not self.canvasList[i-1]:
                    # mpl.rcParams['text.usetex'] = True
                    self.plotFrame = tk.Tk()
                    self.plotFrame.title(f"Эксперимент №{i}")
                    self.plotFrame.resizable(False, False)
                    self.figList.append(self.fig)
                    self.canvasList[i-1] = FigureCanvasTkAgg(self.fig, self.plotFrame)
                    self.canvasList[i-1].get_tk_widget().grid(row=0, column=1)
                    self.plotList[i-1] = self.plotFrame
                    p = partial(noneCanvas, (i))
                    self.plotFrame.protocol("WM_DELETE_WINDOW", p)
                    self.CURRENTERRORCOUNT = 0
                self.fig.savefig(f"MyLab/output/fig/plot{i}.png")
            except:
                self.warningLabel["text"] = f"Эксперимент №{i} не смог сойтись, пожалуйста выберите другие значения"
                self.warningLabel.configure(background="#FFF000")
                self.warningLabel.after(5000, lambda :self.warningLabel.configure(text="Пожалуйста, выберите начальные значения", background="#F0F0F0"))
                self.CURRENTERRORCOUNT+=1
                # if self.CURRENTERRORCOUNT == self.MAXERRORINPUT:
                #     hintWindow = tk.Tk()
                #     hintWindow.resizable(False, False)
                #     hintWindow.geometry("300x90")
                #     hintWindow.title("Подсказка")
                #     # hintPoint = someClass(task)
                #     def close():
                #         hintWindow.destroy()
                #     hint = ttk.Label(hintWindow, text=f"Попробуйте возле точки hintPoint")
                #     hint.pack(anchor="center")
                #     ttk.Button(hintWindow, text="Ok", command=close).pack(side="bottom")
                #     self.CURRENTERRORCOUNT = 0
                print("Некорректные значения")
            finally:
                pass
        with open("MyLab/data/data.json", "w+") as file:
            json.dump(self.dataDict, file)
        pass
        
def main():
    root = Tk()
    db = sqlite3.connect("data/functionList.db")
    app = PlotFrame(root, db)
    app.grid(1,1)
    root.mainloop()
    pass
if __name__ == "__main__":
    main()
    pass