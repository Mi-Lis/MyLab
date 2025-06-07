from functools import partial
import tkinter as tk
from tkinter import ttk
from pathlib import Path

from src.base.core import TasksManager
from src.frames.dirArgFrame.functionmanager import FunctionManager
from src.base.core import LogicFrame, States
from src.frames.BDManager.core import BDManager
from src.frames.dirArgFrame.utils import MPLEntryFrames, savefig

class ArgFrame(LogicFrame, TasksManager):
    id=2
    name = "Выбор аргументов"
    prevState=States.ChooseArgsState
    nextState = States.SolveState
    pathfig = Path("MyLab/tempfig/")
    def updateFun(self, expr, i, name):
        savefig(i, expr, tex=True, name=name, figsize=(3, 0.25), path=self.pathfig)
        self.fm.updateFun(self.pathfig/(name+f'{i}.png'), i)
    # def changeFun(self, fs):

    def __init__(self, root,db:BDManager,task,taskId, nextFrame) -> None:
        super().__init__(root, nextFrame,  use_button=False)
        self.db = db
        self.task=task
        self.taskId = taskId
        self.mainframe = ttk.Frame(self.frame)
        self.warning = ttk.Label(self.mainframe)
        self.countFunFrame = ttk.Frame(self.mainframe)
        self.btnFrame = ttk.Frame(self.countFunFrame)
        self.countFunEntry = MPLEntryFrames(self.countFunFrame, "n", sep="=", columnspan=2)
        self.applyCountBtn = ttk.Button(self.btnFrame, command=self.applyCnt, text = "Ok", state=tk.DISABLED)

    def grid(self, row, col, rowspan=1, colspan=1):
        self.mainframe.grid(sticky='nsew')
        self.mainframe.grid_rowconfigure(0, weight=1)
        self.mainframe.grid_columnconfigure(0, weight=1)
        self.frame.grid_configure(sticky='nsew')
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_rowconfigure(0, weight=1)
        super().grid(row, col)
        self.countFunFrame.grid(sticky='n')
        self.countFunFrame.grid_rowconfigure(0, weight=1)
        # self.countFunFrame.grid_columnconfigure(0, weight=1)
        self.btnFrame.grid_rowconfigure(0, weight=1)
        self.btnFrame.grid_columnconfigure(0, weight=1)
        self.btnFrame.grid(sticky='nsew')

        self.applyCountBtn.grid(row=1, column=0)
        # self.denyCountBtn.grid(row=1, column=1)
        self.warning.grid()
    def applyCnt(self,*args,**kwargs):
        self.n = int(self.countFunEntry.text())
        self.db["TASKINFO"].update("n", self.n)
        self.sc = ttk.Scrollbar(self.mainframe, orient="vertical")
        self.fm = FunctionManager(self.mainframe, self.db, self.n, self.sc,task=self.task,**kwargs)
        self.sc["command"]=self.fm.canvas.yview
        self.sc.grid(row=0, column=1, rowspan=self.n+1, sticky='nsew')
        self.sc.grid_columnconfigure(0, weight=1)
        self.sc.grid_rowconfigure(0, weight=1)
        self.fm.canvas.update()
        try:
            self.applyCountBtn["state"] = tk.DISABLED
        except:self.warning["text"] = "Введите число больше нуля"
        self._nextFrame.n = self.n
        self._nextFrame.setPsi0()

        pass
    def disableCnt(self):
        self.applyCountBtn["state"] = tk.DISABLED
    def _denyCnt(self):
        self.applyCountBtn["state"] = tk.ACTIVE
        try:
            self.fm.close()
        except:
            pass
        pass
    def denyCnt(self):
          self._denyCnt()
          p1 = Path("MyLab/data/fs")
          p2 = Path("MyLab/data/dfs")
          files = p1.glob("**/*")
          [x.unlink() for x in files if x.is_file()]
          files = p2.glob("**/*")
          [x.unlink() for x in files if x.is_file()]
          pass

    def apply(self):
        # super().apply()
        # self.task = self.db["TASKINFO"].get("value")[0][0]
        bd = ""
        if self.fm.typeBk.get():
            bkLen = len(self.fm.x0sFrames)
            for i in range(self.n):
                bd+=f"{self.fm.x0sFrames[i].enity.get()}t"
            for i in range(2*self.n, bkLen):
                bd+=f"{self.fm.x0sFrames[i].enity.get()}t"

        else:
            for i, arg in enumerate(self.fm.x0sFrames):
                bd+=f"{arg.enity.get()}t"
        bd=bd[:-1]
        self.db["BORDERDB"].update_by_id(bd, id=self.taskId)
        print(bd)
        print(self.db["BORDERDB"].get())
        self._nextFrame.show()

    def deny(self):
        # super().deny()
        self.db["BORDERDB"].delete()

    def close(self):
        # self.db["BORDERDB"].delete()
        self.fm.close()
        pass


    def _configure_window(self, event, frame=None, canvas=None):
        if frame.winfo_reqwidth() != canvas.winfo_width():
            canvas.config(width = frame.winfo_reqwidth())
        if frame.winfo_reqheight() != canvas.winfo_height():
            canvas.config(height = frame.winfo_reqheight())

def main():
    root = tk.Tk()
    app = ArgFrame(root, None)
    app.grid(1,1)
    root.mainloop()
    pass
if __name__ == "__main__":
    main()
    pass