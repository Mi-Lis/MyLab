from enum import Enum
from typing import List

from scipy.cluster.hierarchy import weighted

from src.interfaces.ibaselogic import IbaseLogic
from src.interfaces.ibutton import IButton
from src.interfaces.iframe import IFrame
from src.interfaces.istate import IState

import tkinter as tk
from tkinter import ttk
# TASKS = ["Наискорейшее движение точки в сопротивляющейся среде", "Линейные системы с квадратичным критерием качества", "Энергетически оптимальное движение точки в сопротивляющейся среде", "Метод динамического программирования"]

class TasksManager:
    class TASKS(Enum):
        naiskDvi = "Наискорейшее движение точки в сопротивляющейся среде"
        linSys = "Линейные системы с квадратичным критерием качества"
        enOpt = "Энергетически оптимальное движение точки в сопротивляющейся среде"
        metodDyn = "Метод динамического программирования"

    pass
class Task:
    def __init__(self):
        pass
class MainMenu:
    """
    Attributes
    ----------
    menus : dict["str", tk.Menu]
        Словарь хранящий структуры меню программы.
    """
    def __init__(self, master:tk.Tk, dictMenuView:dict) -> None:
        """
        Parameters
        ----------
        master : tk.Tk
        Основное окно для отображения меню программы

        dictMenuView : dict[str, dict]
            Структура меню программы
        """
        self.menus = {"Main":tk.Menu()}
        master.option_add("*tearOff", tk.FALSE)
        for nameMenu,  listMenu in dictMenuView.items():
            self.menus.update({nameMenu:tk.Menu()})
            self._createMenu(nameMenu, listMenu)
            self.menus["Main"].add_cascade(menu=self.menus[nameMenu], label=nameMenu)
    def _createMenu(self,nameMenu, dictNames):
        """
        Метод добавляет функционал в меню программы

        Parameters
        ----------
        nameMenu : str
        Имя текущего элемента меню 

        dictNames : dict{str, Callable}
        Словарь с названиями и методами функций
        """
        for name, com in dictNames.items():
            self.menus[nameMenu].add_command(label=name, command=com)
    def __call__(self):
        """
        Метод возвращает словарь компонентов меню программы.
        
        Returns
        -------
        dict{"str", dict}
        """
        return self.menus["Main"]
    
class BaseFrame(IFrame):
    """
    Базовый класс для фреймов программы.

    Arguments
    ---------
    frame:ttk.Frame
        Основной фрейм класса
    opts : dict
        Словарь с настройками внешнего вида frame.

        Ключи:
            'borderwidth':int
                Толщина границ окна
            'relief':int
            'width':int
                Ширина окна
            'height':int
                Высота окна
    """
    opts = {'borderwidth':1, 'relief':tk.SOLID, 'width':100, 'height':100}
    def __init__(self, master=None, **kwargs) -> None:
        self.frame = ttk.Frame(master, **self.opts)
    def grid(self, row, col, rowspan=1, colspan=1):  
        """
        Метод для размещения фрейма в окне.

        Parameters
        ---------
        row:int
            Строка, где располагается фрейм.
        col:int
            Столбец, где располагается фрейм.
        rowspan:int
            Кол-во строк от row, который займет фрейм
        colspan:int
            Кол-во столбцов от col, который займет фрейм
        """
        self.frame.grid(row=row, column=col, rowspan=rowspan, columnspan=colspan)
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        pass
    pass
class CurrentButton(IButton):
    """
    Имплементация интерфейса IButton

    Arguments
    ---------
    logic:bool
        Состояние объекта, если true состояние button меняется на ACTIVE, если false состояние button меняется на DISABLED
    button : ttk.Button
    """
    def __init__(self, master:ttk.Frame, name:str, func) -> None:
        """
        Parameters
        ---------
        master:ttk.Frame
            Основное окно для отображения меню программы
        name:str
            Название button
        func:function
            Функция вызываемая при нажатии button
        """
        self.logic = func
        self.button = ttk.Button(master, text = name, command=self.logic)
    pass
    def __setitem__(self, key, value):
        self.logic = value
        pass
    def active(self):
        self.button['state'] = tk.ACTIVE
        pass
    def deactive(self):
        self.button['state'] = tk.DISABLED
        pass
    def logic(self):
        pass
    def grid(self, x, y):
        """
        Метод для размещения фрейма в окне.

        Parameters
        ---------
        x:int
            Строка, где располагается фрейм.
        y:int
            Столбец, где располагается фрейм.
        """
        self.button.grid(row=x, column=y)
        pass
    def configure(self, command):
        self.button.configure(command=command)
class CustomButtonFrame(BaseFrame):
    buttons:List[CurrentButton]
    def __init__(self,master, *args) -> None:
        super().__init__(master)
        self.buttons = [CurrentButton(self.frame, "OK", self.apply),
                        CurrentButton(self.frame, "Cancel", self.deny)]
        pass
    pass
class LogicButtonFrame(BaseFrame):
    buttons:List[CurrentButton]

    def __init__(self,master) -> None:
        super().__init__(master)
        self.bFrame = ttk.Frame(master)
        self.buttons = [CurrentButton(self.bFrame, "Ok", self.apply),
                        CurrentButton(self.bFrame, "Cancel", self.deny)]
        pass
    def __getitem__(self, idx):
        match idx:
            case "OK": return self.buttons[0]
            case "Cancel": return self.buttons[1]
        pass
    def __setitem__(self, key, value):
        match key:
            case 'OK': self.buttons[0].configure(command = value)
            case 'Cancel': self.buttons[1].configure(command = value)
        pass
    def apply(self):
        pass
    def deny(self):
        pass
    def active(self):
        for i in range(len(self.buttons)):
            self.buttons[i].active()
    def deactive(self):
        for i in range(len(self.buttons)):
            self.buttons[i].deactive()
        pass
    def grid(self, xIndex, yIndex):
        """
        Метод для размещения фрейма в окне.

        Parameters
        ---------
        xIndex:int
            Строка, где располагается фрейм.
        yIndex:int
            Столбец, где располагается фрейм.
        """
        for i, btn in enumerate(self.buttons):
            btn.grid(0, i)
        self.bFrame.grid()
        pass
    pass
class States:
    '''
        0 - начальное состояние 

        1 - выбрана задача
        
        2 - выбраны аргументы
        
        0 -> 1 -> 2

        2 <- 1 <- 0
    '''
    class LiveStatus:
        active = True
        deactive = False

    class StartState(IState):
        name = "Начальное"
    class ChooseArgsState(IState):
        name = "Выбор параметров"
        pass
    class SolveState(IState):
        name = "Решение"
        pass
    class CompileState(IState):
        name = "Компиляция"
        pass
class LogicFrame(BaseFrame, IbaseLogic):
        """
        Фрейм, управляющий состоянием _nextFrame

        Arguments
        ---------
        _nextFrame:IFrame|LogicFrame|BaseFrame
            Фрейм следующего этапа программы
        _state:States
            Этап выполнения программы
        buttonArea:None|LogicButtonFrame
            Фрейм с кнопками управления "Ok", "Cancel"
        """
        _state = States.StartState()
        _nextFrame:IFrame|None
        def apply(self):
            """
            Метод управляющей передачей управления программой _nextFrame
            """
            self.buttonArea['OK'].deactive()
            self.state = self.nextState
            if self._nextFrame is not None: 
                self._nextFrame.active()
                self._nextFrame.buttonArea.active()

        def deny(self):
            """
            Метод управляющей передачей управления программой LogicFrame
            """
            self.state = self.prevState
            self.buttonArea['OK'].active()
            if self._nextFrame is None:
                pass
            if self._nextFrame is not None and self._nextFrame.live: 
                self._nextFrame.deactive()
                self._nextFrame.buttonArea.deactive()
        def __init__(self,master, nextFrame, **kwargs) -> None:
            self.profile = {}
            super().__init__(master)
            self._nextFrame = nextFrame
            for key, value in kwargs.items():
                self.profile[key] = value
            if "use_button" in self.profile:
                if self.profile["use_button"]:
                    self.buttonArea = LogicButtonFrame(self.frame)
                    self.buttonArea["OK"] = self.apply
                    self.buttonArea["Cancel"] = self.deny
            pass
        @property
        def state(self):
            return self._state
        @state.setter
        def state(self, newState):
            self._state = newState        
            match self.state.name:
                case self.prevState.name: 
                    self.active()
                case self.nextState.name: 
                    self.deactive()
        def grid(self, row, col, rowspan=1, colspan=1):
            """
        Метод для размещения фрейма в окне.

        Parameters
        ---------
        row:int
            Строка, где располагается фрейм.
        col:int
            Столбец, где располагается фрейм.
        rowspan:int
            Кол-во строк от row, который займет фрейм
        colspan:int
            Кол-во столбцов от col, который займет фрейм
            """
            if self.profile["use_button"]:
                self.buttonArea.grid(row+1, col)
            self.frame.grid(row=row, column=col, rowspan=rowspan, columnspan=colspan)
            self.frame.grid_rowconfigure(0, weight=1)
            self.frame.grid_columnconfigure(0, weight=1)
        def active(self):
            self.live = True
        def deactive(self):
            self.live = False
