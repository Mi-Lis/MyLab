import json
from enum import Enum
import threading
from tkinter import Tk, ttk
import tkinter as tk


from pylatex import Document, Section, Plot, Package, Chapter, NoEscape, Figure, Math, Subsection

from src.frames.dirPlotFrame.core import EXPERIMENTSSTEP
from src.frames.BDManager.core import BDManager
from src.base.core import BaseFrame, TasksManager

STEPS = ["FUNS","BS", "HAMILT", "U", "PSI0"]

TEMPLATE = {
    "FUNS":"",
    "BS":"",
    "EXPERIMENTS":{},
}
class PDFMaker(TasksManager):
    class DOC(Enum):
        FUNS=r"Исходная система"
        BS=r"Граничные условия"
        HAMILT=r"Гамильтониан"
        U=r"Управляющая функция"
        EXPERIMENTS = r"Эксперимент"
        PSI0 = r"Начальные значения"
        FUNCT = r"Функционал"
    EXPERIMENTS = {}
    path = r''
    def __init__(self, TITLE,path = r'',*args, **kwargs):
        # Генерация текста
        self.doc = Document(documentclass="extreport", fontenc="T1,T2C")
        self.doc.packages.append(Package('breqn'))
        self.doc.preamble.append(NoEscape(r"\setcounter{page}{2}"))

        try:
            with open("MyLab/data/data.json", "r") as read_file:
                data = json.load(read_file)
                for key in data.keys():
                    TEMPLATE[key] = data[key]
            print(TEMPLATE)
        except:
            pass
        self.doc.packages.append(Package('gost'))
        self.doc.append(NoEscape(r"\tableofcontents"))
        self.doc.append(NoEscape(r"\intro"))
        match TITLE:
            case self.TASKS.naiskDvi.value:
                self.doc.append(NoEscape(r"В отчете представлен пример решения задачи оптимального управления при помощи принципа максимума Понтрягина. "))
                pass
            case self.TASKS.linSys.value:
                self.doc.append(NoEscape(r"В отчете представлен пример решения задачи оптимального управления при помощи принципа максимума Понтрягина. "))

                pass
            case self.TASKS.enOpt.value:
                self.doc.append(NoEscape(r"В отчете представлен пример решения задачи оптимального управления при помощи принципа максимума Понтрягина. "))
                pass
            case self.TASKS.metodDyn.value:
                self.doc.append(NoEscape(r"В отчете представлен пример решения задачи оптимального управления при помощи принципа Беллмана. "))
                pass
        with (self.doc.create(Chapter(TITLE))):
            self.doc.append(Section("Постановка задачи"))
            for step in self.DOC:
                match step:
                    case self.DOC.FUNS:
                        self.doc.append(Subsection(step.value))
                        self.doc.append(NoEscape(f"$${TEMPLATE[step.name]}$$"))
                    case self.DOC.EXPERIMENTS:
                        for  EXPERIMENT in TEMPLATE[step.name].keys():
                            self.doc.append(Section(step.value+f"№{EXPERIMENT}"))
                            for experimentStep in EXPERIMENTSSTEP:
                                match experimentStep:
                                    case self.DOC.FUNCT.name:
                                        self.doc.append(Subsection(self.DOC.FUNCT.value))
                                        self.doc.append(NoEscape("Функционал имеет вид:"))
                                        self.doc.append(NoEscape(f"$$ {TEMPLATE[step.name][EXPERIMENT][experimentStep]}  \\rightarrow \\operatorname{{min}}$$"))
                                        pass
                                    case self.DOC.PSI0.name:
                                        self.doc.append(Subsection(self.DOC.PSI0.value))
                                        match TITLE:
                                            case self.TASKS.metodDyn.value:
                                                self.doc.append(NoEscape(f"$$\\lambda = {TEMPLATE[step.name][EXPERIMENT][experimentStep][0]}, t_k={TEMPLATE[step.name][EXPERIMENT][experimentStep][1]}$$"))
                                            case _:
                                                self.doc.append(NoEscape(
                                            f"$$\\psi_0 = {TEMPLATE[step.name][EXPERIMENT][experimentStep][:-1]}, t_k={TEMPLATE[step.name][EXPERIMENT][experimentStep][-1]}$$"))
                                        pass
                                    case self.DOC.U.name:
                                        self.doc.append(Subsection(self.DOC.U.value))
                                        match TITLE:
                                            case self.TASKS.metodDyn.value:
                                                self.doc.append(NoEscape(f"$${TEMPLATE[step.name][EXPERIMENT][experimentStep]}$$"))
                                            case _:
                                                if TEMPLATE[step.name][EXPERIMENT][experimentStep][1]:
                                                    self.doc.append(NoEscape(r"Управляющая функция ограничена и имеет вид:"))
                                                else:
                                                    self.doc.append(NoEscape(r"Управляющая функция неограниченна и имеет вид:"))
                                                self.doc.append(NoEscape(f"$${TEMPLATE[step.name][EXPERIMENT][experimentStep][0]}$$"))
                                    case self.DOC.HAMILT.name:
                                        self.doc.append(Subsection(self.DOC.HAMILT.value))
                                        match TITLE:
                                            case self.TASKS.metodDyn.value:
                                                self.doc.append(NoEscape(f"""\\begin{{dmath*}}S(x(t), t) = {TEMPLATE[step.name][EXPERIMENT][experimentStep]}\\end{{dmath*}}
                                                """))
                                            case _:
                                                self.doc.append(NoEscape(f"$$H(x(t), \\psi(t), t) = {TEMPLATE[step.name][EXPERIMENT][experimentStep][0]}$$"))
                                                self.doc.append(NoEscape("Гамильтониан в конце движения принимает следующие значения:"))
                                                self.doc.append(NoEscape(f"$$H(x(t_k), \\psi(t_k), t_k) = {TEMPLATE[step.name][EXPERIMENT][experimentStep][1]}$$"))
                            self.doc.append(NoEscape(
                                r"Результаты расчетов представлены в соответствии с рисунком \ref{fig:%s}"%(EXPERIMENT)))
                            with self.doc.create(Figure(position='H')) as plot:
                                try:
                                    plot.add_image(fr"fig/plot{EXPERIMENT}.png", width=NoEscape(r'\linewidth'))
                                    plot.add_caption('Результат работы программы.')
                                    plot.append(NoEscape(r'\label{fig:%s}'%(EXPERIMENT)))  # Добавление метки
                                except:
                                    pass
                                
        self.doc.append(NoEscape(r"\conclusions"))
        match TITLE:
            case self.TASKS.naiskDvi.value:
                self.doc.append(NoEscape(
                    r"В отчете был представлен пример решения задачи оптимального управления при помощи принципа максимума Понтрягина."))
                pass
            case self.TASKS.linSys.value:
                self.doc.append(NoEscape(
                    r"В отчете был представлен пример решения задачи оптимального управления при помощи принципа максимума Понтрягина."))
                pass
            case self.TASKS.enOpt.value:
                self.doc.append(NoEscape(
                    r"В отчете был представлен пример решения задачи оптимального управления при помощи принципа максимума Понтрягина."))
                pass
            case self.TASKS.metodDyn.value:
                self.doc.append(NoEscape(
                    r"В отчете был представлен пример решения задачи оптимального управления при помощи принципа Беллмана."))
                pass
    def compile(self, **kwargs):
        self.doc.generate_pdf('MyLab/output/report', clean_tex=False)

class CompileFrame(BaseFrame):
    id=4
    name = "Генерация отчета"
    def __init__(self, master, db:BDManager,task) -> None:
        super().__init__(master)
        self.db = db
        self.task = task
        self.mainframe = ttk.Frame(self.frame)
        self.btn = ttk.Button(self.mainframe, text="Compile pdf", command=self.compile,state=tk.ACTIVE)
        self.finish_label = ttk.Label(self.mainframe, text = 'Не готово')
        self.pgb = ttk.Progressbar(self.mainframe, orient="horizontal", length=150, mode='indeterminate')

    def grid(self, row=0, col=0, rowspan=1, colspan=1):
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_configure(sticky='we')
        super().grid(row, col)
        self.btn.grid_columnconfigure(0, weight=1)
        self.btn.grid(row=5, column=0,columnspan=2, sticky='nsew')
        self.finish_label.grid(row=4, column=0,columnspan=2, sticky='nsew')
        self.pgb.grid(row=3, column=0,columnspan=2, padx=10, pady=10, sticky='nsew')
        self.mainframe.grid_rowconfigure(0, weight=1)
        self.mainframe.grid_columnconfigure(0, weight=1)
        self.mainframe.grid()

# Скомпилировать pdf файл
    def compile(self):
        def real_compile():
            self.btn['state'] = tk.DISABLED
            try:
                self.pgb.start(5)
                pdf = PDFMaker(self.task)
                pdf.compile()
                self.pgb.stop()
                self.finish_label['text'] = 'Готово'
            except:
                print("Не все данные задачи получены")
                self.finish_label.configure(text="Не все данные задачи получены")
                self.finish_label.after(5000, lambda: self.finish_label.configure(text="Не готово"))
                self.btn['state'] = tk.NORMAL
                self.pgb.stop()
            self.btn['state'] = tk.ACTIVE
        t = threading.Thread(target=real_compile, daemon=True)
        try:
            t.start()
        except:
            print("Не все данные задачи получены")
            self.finish_label.configure(text="Не все данные задачи получены")
            self.finish_label.after(5000, lambda: self.finish_label.configure(text="Не готово"))
            self.btn['state'] = tk.ACTIVE
            self.pgb.stop()



def main():
    root = Tk()
    app = CompileFrame(root)
    app.grid(1,1)
    root.mainloop()
    pass
if __name__ == "__main__":
    main()
    pass