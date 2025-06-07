# -*- coding: utf-8 -*-
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog
import dill
import shutil

import sympy as sp

from src.frames.TaskManager.core import AppTaskManager
from src.frames.dirArgFrame.utils import MPLEntryFrames
from src.base.core import States
from src.base.core import MainMenu
from src.frames.dirArgFrame.core import ArgFrame
from src.frames.dirCheckFrame.core import CheckFrame
from src.frames.dirCompileFrame.core import CompileFrame
from src.frames.dirPlotFrame.core import PlotFrame
from src.frames.BDManager.core import BDManager

"""
            "GROUPS":
            {
                "id":"INT PRIMARY KEY",
                "value":"INT"
            },
            "USERS":
            {
                "id":"INT PRIMARY KEY",
                "F":"TEXT NOT NULL",
                "I":"TEXT NOT NULL",
                "O":"TEXT NOT NULL",
                "groupid":"INT",
                "FOREIGN KEY":
                    {"groupid":"GROUPS"}
            },
            "TASKS":
            {
                "Id":"INT PRIMARY KEY",
                "userID":"INT",
                "taskID":"INT",
                "FOREIGN KEY":
                {"userId":"USERS",
                 "taskId":"TASKINFO"}
            
            },
            "TASKINFO":
            {
                "id":"INT PRIMARY KEY",
                "borderId":"INT",
                "solveId":"INT",
                "taskNameId":"INT",
                "n":"INT",
                "findBalancePoint":"BOOLEAN",
                "isbkfun":"BOOL",
                "FOREIGN KEY":
                {"borderId":"BORDERDB",
                 "solveId":"SOLVEINFO",
                 "taskNameId":"TASKNAME"}
            },
            "SOLVEINFO":
            {
                "id":"INT PRIMARY KEY",
                "varepsilon":"TEXT",
                "num":"TEXT",
                "diffmethod":"TEXT",
                "method":"TEXT",
            },
            "BORDERDB":
            {
               "id":"INTEGER PRIMARY KEY",
               "value":"TEXT NOT NULL"
            },
            "TASKNAME":
            {
                "id":"INTEGER PRIMARY KEY",
                "value":"TEXT NOT NULL"
            }
        }"""

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
    App()

class TaskSolver:
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
    
    greetings = r"--------Выберите тип задачи---------"

    def showDoc(self):
        
        pass
    def showLic(self):
        window = tk.Tk(baseName="Лицензия")
        window.resizable(False, False)
        ttk.Label(window, text=LICENSE).grid()
        pass
    def __init__(self, bd, task_id, user_id):
        self.task_id = task_id
        self.user_id = user_id
        self.bd = bd
        self.task_name = self.bd.execute(f"SELECT value FROM TASKNAME WHERE id = (SELECT taskNameId FROM TASKINFO WHERE id={self.task_id})")[0][0]
        self.window=tk.Toplevel()
        
        self.window.title(self.APP_NAME)
        
        
        
        self.window.maxsize(800, 800)
        self.window.minsize(600, 600)
        self.root = ttk.Frame(self.window)
        self.root.grid(sticky='nsew')
        self.root.rowconfigure(0, weight = 1)
        self.root.columnconfigure(0, weight = 1)

        self.window.resizable()

        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)

        self.window.protocol("WM_DELETE_WINDOW", self.close)

        s = ttk.Style()
        s.configure("My.TFrame",background="#B2DFDB", highlightthickness=2)

        self.noteFrame = ttk.Frame(self.root,style="My.TFrame")

        style = ttk.Style()

        # Настройка внешнего вида вкладок
        style.configure("M.TNotebook", tabposition="n")
        style.configure("M.TNotebook.Tab", padding =[10,10] , anchor="center", width=self.root.winfo_screenwidth())

        self.main = ttk.Notebook(self.noteFrame, style="M.TNotebook")

        menu = MainMenu(self.window, {
            # "Задача":
                                    # {"Открыть":self.open, "Сохранить":self.save},
                               "Настройки":{"Расширенная настройка решателя":self.showAdditionalSettings},
                                "О программе":{"Лицензия":self.showLic, "Документация":self.showDoc}
                                 })

        self.window.config(menu=menu())

        f1 = ttk.Frame(self.root, style="My.TFrame")
        f2 = ttk.Frame(self.main, width=400, height=280)
        f3 = ttk.Frame(self.main, width=400, height=280)
        f4 = ttk.Frame(self.root)
        
        self.ComF = CompileFrame(f4, self.bd,self.task_name)

        self.PF = PlotFrame(f3, self.bd,self.task_name,self.task_id, self.ComF)

        self.AF = ArgFrame(f2,self.bd,self.task_name,self.task_id, self.PF)

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

        self.CF.combobox.set(self.task_name)
        self.CF.combobox["state"] = tk.DISABLED
        self.CF.combobox.event_generate("<<ComboboxSelected>>")  # Генерируем событие
        self.window.grid()
        pass
    def on_select(self, event):
        if self.CF.combobox.get() == self.greetings:
            self.AF.disableCnt()
            return
        try:
            self.AF.denyCnt()
        except:
            pass
        self.AF.countFunEntry.enity.delete(0, tk.END)
        self.PF.countExperements=0
        self.PF.removeExperimentButton['state'] = tk.DISABLED
        # self.bd["TASKINFO"].remove(1)
            #         "TASKINFO":
            # {
            #     "id":"INT PRIMARY KEY",
            #     "borderId":"INT",
            #     "solveId":"INT",
            #     "taskNameId":"INT",
            #     "n":"INT",
            #     "findBalancePoint":"BOOL",
            #     "isbkfun":"BOOL",
            #     "FOREIGN KEY":
            #     {"borderId":"BORDERDB",
            #      "solveId":"SOLVEINFO",
            #      "taskNameId":"TASKNAME"}
            # },
            # "SOLVEINFO":
            # {
            #     "id":"INT PRIMARY KEY",
            #     "varepsilon":"TEXT",
            #     "num":"TEXT",
            #     "diffmethod":"TEXT",
            #     "method":"TEXT",
            # },

        self.prev()
        pass
    def showAdditionalSettings(self):
        additionalWindow = tk.Toplevel()

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
       
        # self.bd["TASKINFO"].update("Findbalancepoint", False)

        def changeInfoBalancePoint():
            # if enabled.get():
            #     self.bd.execute(f"UPDATE TASKINFO SET Findbalancepoint = '{True}' WHERE id = {self.task_id}")
            #     # self.bd["TASKINFO"].update("Findbalancepoint", True)
            # else:
            #     self.bd.execute(f"UPDATE TASKINFO SET Findbalancepoint = '{False}' WHERE id = {self.task_id}")
            pass

        enabled = tk.BooleanVar(additionalWindow, value=False)
        self.includeBalancePointCheckButton = ttk.Checkbutton(self.additionalFrame, text="Искать точку равновесия",
                                                              command=changeInfoBalancePoint,
                                                              variable=enabled)


        def updateSettings():
            for key, i in zip(self.OPTIONS.keys(),self.opts):
                self.OPTIONS[key] = i.enity.get()
                self.bd.execute(f"UPDATE SOLVEINFO SET {key} = '{self.OPTIONS[key]}' WHERE id = {self.task_id}")
                # self.bd["SOLVEINFO"].update(key, self.OPTIONS[key])
            self.bd.execute(f"UPDATE SOLVEINFO SET diffmethod = '{diffcombobox.get()}' WHERE id = {self.task_id}")
            self.bd.execute(f"UPDATE SOLVEINFO SET method = '{combobox.get()}' WHERE id = {self.task_id}")
            print(self.bd["SOLVEINFO"])
            pass
        ttk.Button(additionalWindow, text="Ok", command=updateSettings).grid(column=0)
        self.includeBalancePointCheckButton.grid()
        # self.additionalFrame.grid(column=0)
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
        self.window.destroy()
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

                bs = self.bd["BORDERDB"].get()
                self.AF.countFunEntry.enity.delete(0, tk.END)
                self.AF.countFunEntry.enity.insert(0, str(n))
                self.AF.applyCnt()
                p = Path("MyLab/data/fs/").glob('**/*')
                match task:
                    case "Метод динамического программирования":
                        fs = [0 for _ in range(n+1)]
                    case _:
                        if isBkFun:
                            fs = [0 for _ in range(n+1)]
                            self.AF.fm.typeBk.set(True)
                            self.AF.fm.changeBk()
                            bsEnity = self.AF.fm.x0sFrames[:n]
                            try:
                                bsEnity+=self.AF.fm.x0sFrames[2*n:]
                            except:
                                pass
                            for t, v in zip(bsEnity, bs):
                                t.enity.delete(0, tk.END)
                                t.enity.insert(0, v[1])
                        else:
                            fs = [0 for _ in range(n)]
                            for t, v in zip(self.AF.fm.x0sFrames, bs):
                                t.enity.delete(0, tk.END)
                                t.enity.insert(0, v[1])
                for i, file in enumerate(p):
                    with open(file, "rb") as f:
                        fs[i] = dill.load(f)
                    self.AF.updateFun(fs[i], i, "expr")
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

class App:
    pathfig = Path('MyLab/data/')
    def createDirs(self):
        """
        Метод создает основные директории для программы при первом запуске.
        """
        gost = r"""\NeedsTeXFormat{LaTeX2e}
% \ProvidesPackage{gost}[2014/06/01 кафедра математического и компьютерного моделирования]

\ProcessOptions

\hoffset 0pt
\voffset 0pt

\RequirePackage[
  a4paper, includehead, includefoot, mag=1000,
  headsep=0mm, headheight=0mm,
  left=25mm, right=15mm, top=20mm, bottom=20mm
]{geometry}

\RequirePackage[utf8]{inputenc}
\RequirePackage[T2A]{fontenc}
\RequirePackage[russian]{babel}

% \RequirePackage{cmap} % Улучшенный поиск русских слов в полученном pdf-файле
\RequirePackage[unicode, pdftex]{hyperref}
\RequirePackage{pdfpages}
\RequirePackage[nottoc]{tocbibind}

\RequirePackage[onehalfspacing]{setspace} %"умное" расстояние между строк - установить 1.5 интервала от нормального
\RequirePackage{cite}  %"умные" библиографические ссылки (сортировка и сжатие)
\RequirePackage{indentfirst} %делать отступ в начале параграфа
\RequirePackage{enumerate}  %создание и автоматическая нумерация списков
\RequirePackage{longtable} % Длинные таблицы
\RequirePackage{multirow,makecell,array} % Улучшенное форматирование таблиц
\RequirePackage{graphicx} \graphicspath{{fig/}}
\RequirePackage{float}
\RequirePackage{pdflscape} % Для включения альбомных страниц


\renewcommand{\rmdefault}{ftm} % Включаем Times New Roman
\renewcommand{\sfdefault}{far} % Включаем Arial
%%% Выравнивание и переносы %%%
\sloppy % Избавляемся от переполнений
\clubpenalty=10000 % Запрещаем разрыв страницы после первой строки абзаца
\widowpenalty=10000 % Запрещаем разрыв страницы после последней строки абзаца
\righthyphenmin=2 % Минимальное число символов при переносе - 2.

\RequirePackage{fancyvrb}

\RequirePackage{amssymb,amsmath,amsfonts,latexsym,mathtext} %расширенные наборы  математических символов

\RequirePackage{amsthm}
\theoremstyle{definition}
\newtheorem{theorem}{Теорема}
\newtheorem{proposition}[theorem]{Предложение}
\newtheorem{corollary}[theorem]{Следствие}
\newtheorem{lemma}[theorem]{Лемма}
\newtheorem{definition}[theorem]{Определение}
\newtheorem{example}[theorem]{Пример}
\newtheorem{remark}[theorem]{Замечание}

\RequirePackage[tableposition=top]{caption}
\DeclareCaptionLabelFormat{gostfigure}{Рисунок #2}
\DeclareCaptionLabelFormat{gosttable}{Таблица #2}
\DeclareCaptionLabelSeparator{gost}{~---~}
% \captionsetup{labelsep=gost,justification=justified,singlelinecheck=off}
\captionsetup[figure]{labelformat=gostfigure, labelsep=gost}
\captionsetup[table]{labelformat=gosttable, labelsep=gost,justification=justified,singlelinecheck=off}

\RequirePackage{fancyhdr}
\pagestyle{fancyplain}
\renewcommand{\headrulewidth}{0pt}
\fancyhf{}
\rfoot{\fancyplain{}{\thepage}}

\fancypagestyle{plain}{ 
    \fancyhf{}
    \rfoot{\thepage}}

% \makeatletter 
% \renewcommand\chapter{\if@openright\cleardoublepage\else\clearpage\fi \thispagestyle{fancyplain}%
% \global\@topnum\z@ \@afterindentfalse \secdef\@chapter\@schapter} 
% \makeatother

\addtocontents{toc}{\protect\thispagestyle{fancyplain}}

\setcounter{page}{1}

\RequirePackage{titlesec}
\titleformat{\chapter}[block] 
    {\normalsize\bfseries}
    {\thechapter}
    {1em}{}

\titleformat{\section}[block] 
    {\normalsize\bfseries}
    {\thesection}
    {1em}{}

\titleformat{\subsection}[block] 
    {\normalsize\bfseries}
    {\thesubsection}
    {1em}{}

\titleformat{\paragraph}[block] 
    {\normalsize\bfseries}
    {\thesection}
    {1em}{}


\titlespacing{\chapter}{\parindent}{-30pt}{8pt}
\titlespacing{\section}{\parindent}{*4}{*4}
\titlespacing{\subsection}{\parindent}{*4}{*4}
\titlespacing{\paragraph}{\parindent}{*4}{*4}

\addto\captionsrussian{%
  \renewcommand\contentsname{CОДЕРЖАНИЕ}
  \renewcommand\appendixname{ПРИЛОЖЕНИЕ}
  \renewcommand\bibname{\hfil{}СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ\hfil{}}
}

\RequirePackage{enumitem}
\makeatletter
    \AddEnumerateCounter{\asbuk}{\@asbuk}{м)}
\makeatother
\setlist{nolistsep}
\renewcommand{\labelitemi}{-}
%\renewcommand{\labelenumi}{\asbuk{enumi})}
%\renewcommand{\labelenumii}{\arabic{enumii})}

\RequirePackage{tocloft}
\renewcommand{\cfttoctitlefont}{\hspace{0.38\textwidth} \bfseries\MakeUppercase}
\renewcommand{\cftbeforetoctitleskip}{-1em}
\renewcommand{\cftaftertoctitle}{\mbox{}\hfill \\ \mbox{}\hfill{\footnotesize Стр.}\vspace{-2.5em}}
\renewcommand{\cftchapfont}{\normalsize\bfseries}
\renewcommand{\cftsecfont}{\hspace{31pt}}
\renewcommand{\cftsubsecfont}{\hspace{11pt}}
\renewcommand{\cftbeforechapskip}{1em}
\renewcommand{\cftparskip}{-1mm}
\renewcommand{\cftdotsep}{1}
\renewcommand{\cftchapdotsep}{\cftdotsep}
\setcounter{tocdepth}{2} % задать глубину оглавления — до subsection включительно

\newcommand{\likechapterheading}[1]{
    \clearpage
    \begin{center}
    \textbf{\MakeUppercase{#1}}
    \end{center}}

\newcommand{\abbreviations}{\likechapterheading{ОБОЗНАЧЕНИЯ И СОКРАЩЕНИЯ}\addcontentsline{toc}{chapter}{ОБОЗНАЧЕНИЯ И СОКРАЩЕНИЯ}}
\newcommand{\definitions}{\likechapterheading{ОПРЕДЕЛЕНИЯ}\addcontentsline{toc}{chapter}{ОПРЕДЕЛЕНИЯ}}
\newcommand{\abbrevdef}{\likechapterheading{ОПРЕДЕЛЕНИЯ, ОБОЗНАЧЕНИЯ И СОКРАЩЕНИЯ}\addcontentsline{toc}{chapter}{ОПРЕДЕЛЕНИЯ, ОБОЗНАЧЕНИЯ И СОКРАЩЕНИЯ}}
\newcommand{\intro}{\likechapterheading{ВВЕДЕНИЕ}\addcontentsline{toc}{chapter}{ВВЕДЕНИЕ}}
\newcommand{\conclusions}{\likechapterheading{ЗАКЛЮЧЕНИЕ}\addcontentsline{toc}{chapter}{ЗАКЛЮЧЕНИЕ}}

\makeatletter
  \renewcommand{\@biblabel}[1]{#1.} % Заменяем библиографию с квадратных скобок на точку:
\makeatother

\renewcommand{\rmdefault}{cmr} % Шрифт с засечками

\renewcommand{\sfdefault}{cmss} % Шрифт без засечек

\renewcommand{\ttdefault}{cmtt} % Моноширинный шрифт

\makeatletter
\newcommand\appendix@chapter[1]{%
  \renewcommand{\@makeschapterhead}[1]{\@makechapterhead{#1}}%
  \renewcommand{\thechapter}{\Asbuk{chapter}}%
  \refstepcounter{chapter}%
  \orig@chapter*{\appendixname~\thechapter~#1}%
  \addcontentsline{toc}{chapter}{\appendixname~\thechapter~~#1}%
}
\let\orig@chapter\chapter
\g@addto@macro\appendix{\let\chapter\appendix@chapter}
\makeatother

\newcommand{\Appendix}{
\appendix
\singlespacing
\renewcommand\thechapter{\Asbuk{chapter}}
\titleformat{\chapter}[display]
    {\filcenter}
    {\centering\MakeUppercase{\appendixname} \thechapter}
    {8pt}
    {\bfseries}{}
}
"""
        Path("MyLab").mkdir(exist_ok=True, parents=False)
        Path("MyLab/saves").mkdir(exist_ok=True, parents=False)
        Path("MyLab/output").mkdir(exist_ok=True, parents=False)
        Path("MyLab/output/fig").mkdir(exist_ok=True, parents=False)
        self.pathfig.mkdir(exist_ok=True, parents=False)
        # (self.pathfig/Path("fig")).mkdir(exist_ok=True, parents=False)

        p1 = self.pathfig/Path("fs")
        p2 = self.pathfig/Path("dfs")

        

        p1.mkdir(exist_ok=True, parents=True)
        p2.mkdir(exist_ok=True, parents=True)
        # with f.open("w+b") as file:
        #     pass
        with open(Path("MyLab/output")/Path('gost.sty'),"w+", encoding='utf-8') as file:
            file.write(gost)
            pass
        pass
    def __init__(self):
        self.createDirs()
        self.bdDict = {
            "GROUPS":
            {
                "id":"INT PRIMARY KEY",
                "value":"INT"
            },
            "USERS":
            {
                "id":"INT PRIMARY KEY",
                "F":"TEXT NOT NULL",
                "I":"TEXT NOT NULL",
                "O":"TEXT NOT NULL",
                "groupid":"INT",
                "FOREIGN KEY":
                    {"groupid":"GROUPS"}
            },
            "TASKS":
            {
                "Id":"INT PRIMARY KEY",
                "userID":"INT",
                "taskID":"INT",
                "FOREIGN KEY":
                {"userId":"USERS",
                 "taskId":"TASKINFO"}
            
            },
            "TASKINFO":
            {
                "id":"INT PRIMARY KEY",
                "borderId":"INT",
                "solveId":"INT",
                "taskNameId":"INT",
                "n":"INT",
                "findBalancePoint":"BOOLEAN",
                "isbkfun":"BOOL",
                "FOREIGN KEY":
                {"borderId":"BORDERDB",
                 "solveId":"SOLVEINFO",
                 "taskNameId":"TASKNAME"}
            },
            "SOLVEINFO":
            {
                "id":"INT PRIMARY KEY",
                "varepsilon":"TEXT",
                "num":"TEXT",
                "diffmethod":"TEXT",
                "method":"TEXT",
            },
            "BORDERDB":
            {
               "id":"INTEGER PRIMARY KEY",
               "value":"TEXT NOT NULL"
            },
            "TASKNAME":
            {
                "id":"INTEGER PRIMARY KEY",
                "value":"TEXT NOT NULL"
            }
        }


        self.bd = BDManager(self.bdDict)
        try:
            self.bd["TASKNAME"].add(1, "'Наискорейшее движение точки в сопротивляющейся среде'")
        except:
            pass
        try:
            self.bd["TASKNAME"].add(2, "'Линейные системы с квадратичным критерием качества'")
        except:
            pass
        try:
            self.bd["TASKNAME"].add(3, "'Энергетически оптимальное движение точки в сопротивляющейся среде'")
        except:
            pass
        try:
            self.bd["TASKNAME"].add(4, "'Метод динамического программирования'")
        except:
            pass
        try:
            self.bd["GROUPS"].add(1, "411")
        except:
            pass
        try:
            self.bd["GROUPS"].add(2, "247")
        except:
            pass
        f = self.pathfig/Path("functionList.db")
        def nextStep():
            print(self.TM.task_manager.current_task,self.TM.user_manager.current_user)
            TaskSolver(self.bd, self.TM.task_manager.current_task,self.TM.user_manager.current_user)
        self.TM = AppTaskManager(self.bd, nextStep)
        self.TM.mainloop()
        pass


if __name__ == '__main__':
    main()