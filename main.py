# -*- coding: utf-8 -*-
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog
import dill
import shutil
import sympy as sp

from src.frames.dirArgFrame.utils import MPLEntryFrames
from src.base.core import States
from src.base.core import MainMenu
from src.frames.dirArgFrame.core import ArgFrame
from src.frames.dirCheckFrame.core import CheckFrame
from src.frames.dirCompileFrame.core import CompileFrame
from src.frames.dirPlotFrame.core import PlotFrame
from src.frames.BDManager.core import BDManager

LICENSE = """
Copyright <2025> <Mikhail L.>

Permission is hereby granted, free of charge, to any person obtaining 
a copy of this software and associated documentation files (the “Software”), 
to deal in the Software without restriction, 
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall 
be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, 
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
FIGUREPATH = r'./fig/'
def main():
    myapp = App()

    myapp.run()

class App(tk.Tk):
    """
    Основной класс программы.

    Parameters
    ----------
    geometry : str
        Размер окна в формате NxM.
    pathfig : Path
        Расположение данных программы.
    """
    APP_NAME = 'MyLab'
    OPTIONS = {'varepsilon':r'1e-5', 'num':r'50'}
    METHODS = {'diffmethod':r"RK45", "method":r"hybr"}
    DIFFSOLVEMETHODES = ['RK45', 'RK23', 'DOP853', 'LSODA']
    NONLINEARSOLVEMETHODS = ['hybr',

                             'lm',

                             'broyden1',

                             'broyden2',

                             'anderson',

                             'linearmixing',

                             'diagbroyden',

                             'excitingmixing',

                             'krylov',

                             'df-sane']
    kwargs = {"balancePoint": None, "balanceFunction": None, "balanceLine": None}

    geometry = "800x600"
    pathfig = Path('MyLab/data/')
    greetings = r"--------Выберите тип задачи---------"
    def createDirs(self):
        """
        Метод создает основные директории для программы при первом запуске.
        """
        gost = r""" 
"""
        Path("MyLab").mkdir(exist_ok=True, parents=False)
        Path("MyLab/saves").mkdir(exist_ok=True, parents=False)
        Path("MyLab/output").mkdir(exist_ok=True, parents=False)
        Path("MyLab/output/fig").mkdir(exist_ok=True, parents=False)
        self.pathfig.mkdir(exist_ok=True, parents=False)
        # (self.pathfig/Path("fig")).mkdir(exist_ok=True, parents=False)

        p1 = self.pathfig/Path("fs")
        p2 = self.pathfig/Path("dfs")

        f = self.pathfig/Path("functionList.db")

        p1.mkdir(exist_ok=True, parents=True)
        p2.mkdir(exist_ok=True, parents=True)
        with f.open("w+b") as file:
            pass
        with open(Path("MyLab/output")/Path('gost.sty'),"w+", encoding='utf-8') as file:
            file.write(gost)
            pass
        pass
    def showDoc(self):
        
        pass
    def showLic(self):
        window = tk.Tk(baseName="Лицензия")
        window.resizable(False, False)
        ttk.Label(window, text=LICENSE).grid()
        pass
    def __init__(self):
        super().__init__()
        self.title(self.APP_NAME)
        self.createDirs()
        self.bdDict = {
            # "FUNDB":
            # {
            #     "id":"INTEGER PRIMARY KEY",
            #     "TexView": "Text Not Null",
            #     "SymView": "Text"
            # }
            # ,
            "TASKINFO":
            {
                "id":"INT PRIMARY KEY",
                "value":"TEXT",
                "n":"INT",
                "varepsilon":"TEXT",
                "num":"TEXT",
                "findBalancePoint":"BOOLEAN",
                "diffmethod":"TEXT",
                "method":"TEXT",
                "isbkfun":"BOOL"
            }
            ,
           "BORDERDB":
            {
               "id":"INTEGER PRIMARY KEY",
               "value":"TEXT NOT NULL"
            }
            # ,
            # "FUNCTIONALDB":
            # {
            #     "id":"INTEGER PRIMARY KEY",
            #     "value":"TEXT NOT NULL"
            # }
        }

        self.maxsize(800, 800)
        self.minsize(600, 600)

        self.bd = BDManager(self.bdDict)
        self.bd["TASKINFO"].add(1,r"0",2,self.OPTIONS["varepsilon"],self.OPTIONS["num"],False, f"'{self.METHODS["diffmethod"]}'", f"'{self.METHODS["method"]}'", False)
        self.states = [States.StartState, States.ChooseArgsState, None, None, None]

        self.root = ttk.Frame()
        self.root.grid(sticky='nsew')
        self.root.rowconfigure(0, weight = 1)
        self.root.columnconfigure(0, weight = 1)

        self.resizable()

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.protocol("WM_DELETE_WINDOW", self.close)

        s = ttk.Style()
        s.configure("My.TFrame",background="#B2DFDB", highlightthickness=2)

        self.noteFrame = ttk.Frame(self.root,style="My.TFrame")

        style = ttk.Style()

        # Настройка внешнего вида вкладок
        style.configure("M.TNotebook", tabposition="n")
        style.configure("M.TNotebook.Tab", padding =[10,10] , anchor="center", width=self.root.winfo_screenwidth())

        self.main = ttk.Notebook(self.noteFrame, style="M.TNotebook")

        menu = MainMenu(self, {"Задача":
                                    {"Открыть":self.open, "Сохранить":self.save},
                               "Настройки":{"Расширенная настройка решателя":self.showAdditionalSettings},
                                "О программе":{"Лицензия":self.showLic, "Документация":self.showDoc}
                                 })

        self.config(menu=menu())

        f1 = ttk.Frame(self.root, style="My.TFrame")
        f2 = ttk.Frame(self.main, width=400, height=280)
        f3 = ttk.Frame(self.main, width=400, height=280)
        f4 = ttk.Frame(self.root)
        
        self.ComF = CompileFrame(f4, self.bd)

        self.PF = PlotFrame(f3, self.bd, self.ComF)

        self.AF = ArgFrame(f2,self.bd, self.PF)

        self.CF = CheckFrame(f1,self.bd,self.AF)

        self.CF.combobox.bind("<<ComboboxSelected>>", self.on_select)

        self.buttonFrame = ttk.Frame(self.root)

        self.nextButton = ttk.Button(self.buttonFrame, text="Далее", command=self.next, state=tk.ACTIVE)
        self.prevButton = ttk.Button(self.buttonFrame, text="Назад", command=self.prev, state=tk.DISABLED)

        self.frames = [f2, f3]
        self.ws = [self.AF, self.PF]

        self.root.grid_rowconfigure(0, weight=0)  # Первая строка (верхний фрейм) не растягивается
        self.root.grid_rowconfigure(1, weight=1)  # Вторая строка (средний фрейм) растягивается
        self.root.grid_rowconfigure(2, weight=0)  # Третья строка (нижний фрейм) не растягивается
        self.root.grid_columnconfigure(0, weight=1)  # Единственный столбец растягивается

        f1.grid(sticky='new')
        f1.grid_rowconfigure(0, weight=1)
        f1.grid_columnconfigure(0, weight=1)
        f2.grid_rowconfigure(0, weight=1)
        f2.grid_columnconfigure(0, weight=1)
        f3.grid_rowconfigure(0, weight=1)
        f3.grid_columnconfigure(0, weight=1)

        self.main.add(f2, text=self.AF.name, state='normal', compound=tk.LEFT)
        self.main.add(f3, text=self.PF.name, state='disabled', compound=tk.LEFT)

        self.CF.grid(0,0)
        self.noteFrame.grid_rowconfigure(0, weight=1)
        self.noteFrame.grid_columnconfigure(0, weight=1)
        self.noteFrame.grid(row=1,column=0,sticky='nsew')
        def stop(event):
            return 'break'
        self.main.bind("<ButtonPress-1>", stop)
        self.main.grid_rowconfigure(0, weight=1)
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid(sticky='nsew')

        self.grid()
        self.buttonFrame.grid_columnconfigure(0, weight=1)
        self.buttonFrame.grid_columnconfigure(1, weight=1)
        self.buttonFrame.grid()
        self.nextButton.grid(row=0,column=1, sticky='nsew')
        self.prevButton.grid(row=0,column=0, sticky='nsew')
        f4.grid_rowconfigure(0, weight=1)
        f4.grid_columnconfigure(0, weight=1)
        f4.grid(sticky='ew')
        self.ComF.grid(2, 0)
        pass
    def on_select(self, event):
        if self.CF.combobox.get() == self.greetings:
            self.AF.disableCnt()
            return
        try:
            # self.bd["TASKINFO"].remove(1)
            self.AF.denyCnt()
            # self.AF.countFunEntry.enity.delete(0, tk.END)
        except:
            pass
        self.AF.countFunEntry.enity.delete(0, tk.END)
        self.PF.countExperements=0
        self.bd["TASKINFO"].update("id", 1,"value", f"{self.CF.combobox.get()}")

        # self.main.tab(1, state='disabled')
        self.prev()
        pass
    def showAdditionalSettings(self):
        additionalWindow = tk.Tk()

        self.opts = []

        self.optFrame = ttk.LabelFrame(additionalWindow, text="Параметры решателя")
        for key, value in self.OPTIONS.items():
            self.opts.append(MPLEntryFrames(self.optFrame, sp.Symbol(key), sep='=', sticky='nsew', columnconfigure=0))
            self.opts[-1].enity.insert(0, value)

        self.optFrame.grid(row=0, column=0, sticky='n')
        self.optFrame.grid_columnconfigure(0, weight=1)
        self.optFrame.grid_columnconfigure(1, weight=1)
        ttk.Separator(self.optFrame, orient="horizontal").grid(pady=5, sticky='nsew')
        ttk.Label(self.optFrame, text='Метод решения дифф. уравнений').grid()
        diffcombobox = ttk.Combobox(self.optFrame, values=self.DIFFSOLVEMETHODES)
        diffcombobox.set(self.DIFFSOLVEMETHODES[0])
        diffcombobox.grid()

        ttk.Label(self.optFrame, text='Метод').grid()
        combobox = ttk.Combobox(self.optFrame, values=self.NONLINEARSOLVEMETHODS)
        combobox.set(self.NONLINEARSOLVEMETHODS[0])
        combobox.grid()

        self.additionalFrame = ttk.LabelFrame(self.optFrame, text="Дополнительные данные задачи")
        # ttk.Separator(self., orient="horizontal").grid(pady=5, sticky='nsew')
        self.bd["TASKINFO"].update("Findbalancepoint", False)

        def changeInfoBalancePoint():
            if enabled.get():
                self.bd["TASKINFO"].update("Findbalancepoint", True)
            else:
                self.bd["TASKINFO"].update("Findbalancepoint", False)
            pass

        enabled = tk.BooleanVar(additionalWindow, value=False)
        self.includeBalancePointCheckButton = ttk.Checkbutton(self.additionalFrame, text="Искать точку равновесия",
                                                              command=changeInfoBalancePoint,
                                                              variable=enabled)


        def updateSettings():
            for key, i in zip(self.OPTIONS.keys(),self.opts):
                self.OPTIONS[key] = i.enity.get()
                self.bd["TASKINFO"].update(key, self.OPTIONS[key])
            self.bd["TASKINFO"].update("diffmethod", diffcombobox.get())
            self.bd["TASKINFO"].update("method", combobox.get())
            print(self.bd["TASKINFO"])
            pass
        ttk.Button(additionalWindow, text="Ok", command=updateSettings).grid(column=0)
        self.includeBalancePointCheckButton.grid()
        self.additionalFrame.grid(column=0)
        pass
    def next(self):
        self.AF.apply()
        # except: return
        self.main.tab(1, state='normal')
        self.main.select(1)
        self.nextButton["state"] = tk.DISABLED
        self.prevButton["state"] = tk.ACTIVE
        pass
    def prev(self):
        self.main.select(0)
        self.nextButton["state"] = tk.ACTIVE
        self.prevButton["state"] = tk.DISABLED
    def grid(self):
        super().grid()
        for frame in self.ws:
            frame.grid(0,0)
    def close(self):

        try:
            self.CF.close()
        except: pass
        try:
            self.AF.close()
        except: pass
        try:
            self.PF.close()
        except: pass
        try:
            for im in self.pathfig.iterdir():
                im.unlink()
        except:
            pass
        # self.bd.__del__()
        def deleteDirs(dirName):
            p = Path('MyLab/data/'+dirName)
            files = p.glob("**/*")
            [x.unlink() for x in files if x.is_file()]
        deleteDirs("fs")
        deleteDirs("dfs")
        self.quit()
        self.destroy()
    def open(self):
        to = filedialog.askdirectory()
        if to!="":
            print(to)
            self.AF.denyCnt()
            to = Path(to+'/')
            f = to/Path("functionList.db")
            shutil.copy(f, Path("MyLab/data/functionList.db"))
            self.bd = BDManager(self.bdDict)
            f = (to/Path("fs")).glob("**/*")
            dfs = (to/Path("dfs")).glob("**/*")
            [shutil.copy(x, Path("MyLab/data/fs/")) for x in f if x.is_file()]
            [shutil.copy(x, Path("MyLab/data/dfs/")) for x in dfs if x.is_file()]
            print(self.bd["TASKINFO"].get())
            task = self.bd["TASKINFO"].get()[0][1]
            self.CF.combobox.set(task)
            n = self.bd["TASKINFO"].get("n")[0][0]
            isBkFun = self.bd["TASKINFO"].get("Isbkfun")[0][0]
            if n>0:
                self.AF.countFunEntry.enity.delete(0, tk.END)
                self.AF.countFunEntry.enity.insert(0, str(n))
                p = Path("MyLab/data/fs/").glob('**/*')
                match task:
                    case "Метод динамического программирования":
                        fs = [0 for _ in range(n+1)]
                        for i, file in enumerate(p):
                            with open(file, "rb") as f:
                                fs[i] = dill.load(f)
                            self.AF.updateFun(fs[i], i, "expr")
                    case _:
                        if isBkFun:
                            fs = [0 for _ in range(n+1)]
                        else:
                            fs = [0 for _ in range(n)]
                        for i, file in enumerate(p):
                            with open(file, "rb") as f:
                                fs[i] = dill.load(f)
                            self.AF.updateFun(fs[i], i, "expr")

                self.AF.applyCnt(exprs=fs, postfix="")
                bs = self.bd["BORDERDB"].get()
                print(bs)
                for t, v in zip(self.AF.fm.x0sFrames,bs):
                    t.enity.delete(0, tk.END)
                    t.enity.insert(0, v[1])
        self.prev()
        pass
    def save(self):
        to = filedialog.askdirectory()
        if to!="":
            to = Path(to+'/')
            (to/Path("fs")).mkdir(parents=True, exist_ok=True)
            (to/Path("dfs")).mkdir(parents=True, exist_ok=True)
            f = Path("MyLab/data/functionList.db")
            shutil.copy(f, to/Path("functionList.db"))
            f = Path("MyLab/data/fs").glob("**/*")
            dfs = Path("MyLab/data/dfs").glob("**/*")
            [shutil.copy(x, to/Path("fs/"+x.name)) for x in f if x.is_file()]
            [shutil.copy(x, to/Path("dfs/"+x.name)) for x in dfs if x.is_file()]
        pass
    def run(self):
        self.mainloop()

if __name__ == '__main__':
    main()
