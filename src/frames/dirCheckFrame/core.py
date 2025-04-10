from tkinter import Tk, ttk

from src.base.core import TasksManager
from src.base.core import LogicFrame, States
from src.frames.BDManager.core import BDManager

class CheckFrame(LogicFrame, TasksManager):
    id=1
    greetings = "<----------------Выберите тип задачи---------------->"
    name = "Выбор лабораторной работы"
    nextState = States.ChooseArgsState
    prevState = States.StartState
    def __init__(self, master,db:BDManager, nextFrame) -> None:
        super().__init__(master, nextFrame, use_button=False)
        self.db = db

        self.combobox = ttk.Combobox(self.frame, values=[task.value for task in self.TASKS], state="readonly", width=70, height=20, justify='center')
        self.combobox.set(self.greetings)
        self.combobox.grid(sticky='nswe')


    def grid(self, row, col, rowspan=1, colspan=1):
        self.frame['height'] = 20
        self.frame['width'] = 20
        self.frame.grid_configure(sticky ='we')
        super().grid(row, col, rowspan, colspan)
    #
    # def active(self):
    #     super().active()
    #
    #     self._nextFrame.countFunEntry.enity.delete(0, END)
    #     self._nextFrame.denyCnt()
    #     self._nextFrame.countFunEntry.enity["state"] = tk.ACTIVE
    # def deactive(self):
    #     super().deactive()
    #
    #     self._nextFrame.countFunEntry.enity.delete(0, END)
    #     self.db["TASKINFO"].add(1,f"'{self.combobox.get()}'")
    #     match self.combobox.get():
    #         case "Наискорейшее движение точки в сопротивляющейся среде":
    #             pass
    #         case "Линейные системы с квадратичным критерием качества":
    #             pass
    #         case "Энергетически оптимальное движение точки в сопротивляющейся среде":
    #             pass
    #         case "Метод динамического программирования":
    #             pass

    def close(self):
        self.db["TASKINFO"].remove(1)
        pass

def main():
    testtask = ["1", "2"]
    root = Tk()
    app = CheckFrame(root, None, testtask)
    app.grid(1,1)
    root.mainloop()
    pass
if __name__ == "__main__":
    main()
    pass