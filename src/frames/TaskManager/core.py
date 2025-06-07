from functools import partial
from re import S
import tkinter as tk
from tkinter import ttk
import sqlite3
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox

from sympy import solve
# from src.base.core import TasksManager
# from src.frames.BDManager.core import BDManager
class UserTreeManager:
    userId = 0
    def __init__(self, db,parent_frame, on_user_select):
        self.db = db
        global users_listbox, current_user_id
        self.tree = ttk.Treeview(parent_frame, columns=("id", "F", "I", "O","g"), show="headings")
        self.tree.heading("id", text="ID")
        self.tree.heading("F", text="Фамилия")
        self.tree.heading("I", text="Имя")
        self.tree.heading("O", text="Отчество")
        self.tree.heading("g", text="Группа")
        
        self.tree.column("id", width=50)
        self.tree.column("F", width=100)
        self.tree.column("I", width=100)
        self.tree.column("O", width=100)
        self.tree.column("g", width=50)
        self.tree.pack(fill=tk.Y, expand=True)
        self.on_user_select = on_user_select
        self.tree.bind("<<TreeviewSelect>>", self.handle_selection)
        self.groups = self.db["GROUPS"].get()
        self.load_users()

    def load_users(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for id, user in enumerate(self.db["USERS"].get()):
            self.update_users_list(user[0])
            self.userId = user[0]

    def handle_selection(self, event):
        selected_item = self.tree.focus()
        if selected_item:
            self.on_user_select(int(selected_item))
        self.current_user = selected_item
    def update_users_list(self, id):
        v = self.db.execute(f"SELECT * FROM USERS WHERE id = {id}")
        if not v:
            return
        print(self.groups[int(v[0][-1])-1][1])
        print(v[0][:-1])
        self.tree.insert("",tk.END,iid=id,values=(*v[0][:-1], self.groups[int(v[0][-1])-1][1]))
        pass
    # Добавить нового пользователя
    def del_user(self, update_task):
        if not self.current_user:
            return
        
        self.db["USERS"].remove(int(self.current_user))

        self.load_users()
        update_task()
        pass
    def add_new_user(self):
        def getFIO():
            F = fEntry.get()
            I = iEntry.get()
            O = oEntry.get()
            g = gCombo.get()
            if F!='' and I!='' and O!='':
                user_id = self.add_user(f"'{F}'", f"'{I}'", f"'{O}'", g)
                if user_id:
                    self.update_users_list(user_id)
            FIOWindow.destroy()
        FIOWindow = tk.Tk()
        
        fLabel = ttk.Label(FIOWindow, text="Фамилия")
        iLabel = ttk.Label(FIOWindow, text="Имя")
        oLabel = ttk.Label(FIOWindow, text="Отчество")
        gLabel = ttk.Label(FIOWindow, text="Группа")
        
        fEntry = ttk.Entry(FIOWindow)
        iEntry = ttk.Entry(FIOWindow)
        oEntry = ttk.Entry(FIOWindow)
        gCombo = ttk.Combobox(FIOWindow, values=[v[0] for v in self.db["GROUPS"].get("value")])        
        
        fLabel.grid()
        fEntry.grid()

        iLabel.grid()
        iEntry.grid()
        
        oLabel.grid()
        oEntry.grid()
        
        gLabel.grid()
        gCombo.grid()
        okButton = ttk.Button(FIOWindow,command=getFIO, text="Готово")
        okButton.grid()
        FIOWindow.grid(column=0)

    # # Обновление списка пользователей в интерфейсе
    # def refresh_users_list(self, users_listbox):
    #     users_listbox.delete(0, tk.END)
    #     for user in self.get_users():
    #         users_listbox.insert(tk.END, f"{user[0]}: {user[1:]}")

    # Добавить нового пользователя
    def add_user(self, F, I, O,g):
        try:
            self.userId+=1
            gId = self.db.execute(f"SELECT id FROM GROUPS WHERE value = {g}")[0][0]
            self.db["USERS"].add(self.userId,F,I,O,gId)
            return self.userId
        except sqlite3.IntegrityError:
            messagebox.showerror("Ошибка", "Пользователь с таким именем уже существует.")
            return None

class TaskTreeManager:
    OPTIONS = {'varepsilon':r'1e-5', 'num':r'50'}
    METHODS = {'diffmethod':r"RK45", "method":r"hybr"}
    def __init__(self,db, parent_frame, editor_frame):
        self.db = db
        self.tree = ttk.Treeview(parent_frame, columns=("id", "task_id","task"), show="headings")
        self.tree.heading("id", text="ID")
        self.tree.heading("task_id", text="Номер задачи")
        self.tree.heading("task", text="Задача")
        # self.tree.heading("status", text="Статус")

        self.tree.column("id", width=50)
        self.tree.column("task_id",  width=50)
        self.tree.column("task",  width=100)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.editor_frame = editor_frame
        self.current_user_id = None
        self.tree.bind("<<TreeviewSelect>>", self.handle_selection)
        print("Tasks")
        print(self.db["TASKS"].get())
        try:
            self.taskId = len(self.db["TASKS"].get())
        except:
            self.taskId = 0
    def handle_selection(self, event):
        selected_item = self.tree.focus()
        values = self.tree.item(selected_item, 'values')
        try:
            self.current_task = values[1]
        except:
            pass
       
        pass
    def set_user(self, user_id):
        self.current_user_id = user_id
        self.load_tasks()

    def load_tasks(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        if not self.current_user_id:
            return
        id_task_by_user = self.db.execute(f"SELECT id,taskId FROM TASKS WHERE userID={self.current_user_id}")
        
        taskname = self.db.execute(f"SELECT * FROM TASKNAME")
        for id,taskId in enumerate(id_task_by_user):
            self.tree.insert("", tk.END,iid=id, values=(id+1,taskId[0],taskname[taskId[1]-1][1]))
    def get_task_templates(self):
        return [v[0] for v in self.db["TASKNAME"].get("value")] + ["Другое..."]

    def add_inline_task(self):
        if self.current_user_id is None:
            messagebox.showwarning("Ошибка", "Выберите пользователя!")
            return

        for widget in self.editor_frame.winfo_children():
            widget.destroy()

        combo = ttk.Combobox(self.editor_frame, values=self.get_task_templates(), state="readonly")
        combo.pack(pady=5, fill=tk.X)
        
        def on_select(event=None):
            task_name = combo.get()
            taskId = self.db.execute(f"SELECT id FROM TASKNAME WHERE value='{task_name}'")[0][0]
            print("taskId")
            print(taskId)
            if task_name == "Другое...":
                task_name = simpledialog.askstring("Новая задача", "Введите название задачи:")
            if task_name:
                self.taskId+=1
                
                self.db["SOLVEINFO"].add(self.taskId, self.OPTIONS["varepsilon"],self.OPTIONS["num"],f"'{self.METHODS["diffmethod"]}'",f"'{self.METHODS["method"]}'")
                self.db["BORDERDB"].add(self.taskId)
                self.db["TASKINFO"].add(self.taskId,self.taskId,self.taskId,taskId,0, False, False)
                self.db["TASKS"].add(self.taskId,self.current_user_id, taskId)
                self.load_tasks()
            combo.destroy()

        combo.bind("<<ComboboxSelected>>", on_select)
class AppTaskManager(tk.Tk):
    def __init__(self, db,treeSelect, screenName = None, baseName = None, className = "Tk", useTk = True, sync = False, use = None):
        super().__init__(screenName, baseName, className, useTk, sync, use)
        self.title("MyLab")
        self.geometry("800x400")

        # --- Левая панель — пользователи ---
        left_frame = tk.Frame(self)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(left_frame, text="Пользователи").pack(anchor=tk.W)
        self.user_manager = UserTreeManager(db,left_frame, self.on_user_selected)
        self.add_user_button = ttk.Button(left_frame, text="Добавить пользователя", command=self.user_manager.add_new_user)
        self.add_user_button.pack(pady=5, fill=tk.X)


        # --- Правая панель — задачи ---
        right_frame = tk.Frame(self)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(right_frame, text="Задачи").pack(anchor=tk.W)

        self.editor_frame = tk.Frame(right_frame)
        self.editor_frame.pack(pady=5, fill=tk.X)

        self.task_manager = TaskTreeManager(db,right_frame, self.editor_frame)

        self.add_task_button = ttk.Button(right_frame, text="Добавить задачу", command=self.task_manager.add_inline_task)
        self.add_task_button.pack(pady=5, fill=tk.X)
        self.next = treeSelect
        self.next_button = ttk.Button(self, text="Далее", command=self.next)
        self.next_button.pack(side=tk.BOTTOM, anchor='se')
        del_user = partial(self.user_manager.del_user, self.task_manager.load_tasks)
        self.del_user_button = ttk.Button(left_frame, text="Удалить пользователя", command=del_user)
        self.del_user_button.pack(pady=5, fill=tk.X)
    def on_user_selected(self, user_id):
        self.task_manager.set_user(user_id)
    

#     pass
# class TaskManager(ttk.Frame):
#     userId=0
#     def __init__(self, db:BDManager,master = None, **kwargs):
#         super().__init__(master, **kwargs)
#         self.db = db
#         print(self.db["USERS"])
        
#         current_user_id = None
#         # Левая панель: пользователи
#         left_frame = tk.Frame(master)
#         left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

#         tk.Label(left_frame, text="Пользователи").pack(anchor=tk.W)
#         users_listbox = tk.Listbox(left_frame, width=30)
#         users_listbox.pack(fill=tk.Y, expand=True)
#         users_listbox.bind("<Double-Button-1>", lambda e: self.on_user_select(e, users_listbox, tree))
#         self.refresh_users_list(users_listbox)

#         btn_add_user = ttk.Button(left_frame, text="Добавить пользователя",
#                                 command=lambda: self.add_new_user(users_listbox))
#         btn_add_user.pack(pady=5, fill=tk.X)

#         # Правая панель: задачи
#         right_frame = tk.Frame(master)
#         right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

#         tk.Label(right_frame, text="Задачи").pack(anchor=tk.W)
#         columns = ("id", "task_name", "status")
#         tree = ttk.Treeview(right_frame, columns=columns, show="headings")
#         tree.heading("id", text="ID")
#         tree.heading("task_name", text="Задача")
#         tree.heading("status", text="Статус")
#         tree.pack(fill=tk.BOTH, expand=True)

#         btn_add_task = tk.Button(right_frame, text="Добавить задачу",
#                                 command=lambda: self.add_new_task(tree))
#         btn_add_task.pack(pady=5, fill=tk.X)

#     # Получить список пользователей
#     def get_users(self):
#         conn = sqlite3.connect(DB_NAME)
#         cur = conn.cursor()
#         print(self.db["USERS"])
#         users = self.db["USERS"].get()
#         conn.close()
#         return users


#     # Загрузить задачи для выбранного пользователя
#     def load_tasks(self, user_id):
#         # conn = sqlite3.connect(DB_NAME)
#         # cur = conn.cursor()
#         # cur.execute("SELECT id, task_name, status FROM tasks WHERE user_id=?", (user_id,))
#         # tasks = cur.fetchall()
#         # conn.close()
#         # return tasks
#         pass

#     # Сохранить новую задачу
#     def save_task(self, user_id, task_name, status="Активна"):
#         # conn = sqlite3.connect(DB_NAME)
#         # cur = conn.cursor()
        
#         # cur.execute("INSERT INTO tasks (user_id, task_name, status) VALUES (?, ?, ?)",
#         #             (user_id, task_name, status))
#         # conn.commit()
#         # conn.close()
#         pass


#     # При выборе пользователя обновляем задачи
#     def on_user_select(self, event, users_listbox, tree):
#         global current_user_id
#         selection = users_listbox.curselection()
#         if not selection:
#             return
#         selected_line = users_listbox.get(selection[0])
#         try:
#             user_id = int(selected_line.split(':')[0])
#             current_user_id = user_id

#             # Очистить старые задачи
#             for row in tree.get_children():
#                 tree.delete(row)

#             # Загрузить новые
#             for task in self.load_tasks(user_id):
#                 tree.insert("", tk.END, values=(task[0], task[1], task[2]))
#         except Exception as e:
#             messagebox.showerror("Ошибка", f"Не удалось загрузить данные: {e}")

#     # Добавить нового пользователя
#     def add_new_user(self, users_listbox):
#         def getFIO():
#             F = fEntry.get()
#             I = iEntry.get()
#             O = oEntry.get()
#             # F = simpledialog.askstring("Добавить пользователя", "Введите имя:")
#             # I = simpledialog.askstring("Добавить пользователя", "Введите имя:")
#             # O = simpledialog.askstring("Добавить пользователя", "Введите имя:")
            
#             if F!='' and I!='' and O!='':
#                 user_id = self.add_user(f"'{F}'", f"'{I}'", f"'{O}'")
#                 if user_id:
#                     self.refresh_users_list(users_listbox)
#             FIOWindow.destroy()
#         FIOWindow = tk.Tk()
        
#         fLabel = ttk.Label(FIOWindow, text="Фамилия")
#         iLabel = ttk.Label(FIOWindow, text="Имя")
#         oLabel = ttk.Label(FIOWindow, text="Отчество")
        
#         fEntry = ttk.Entry(FIOWindow)
#         iEntry = ttk.Entry(FIOWindow)
#         oEntry = ttk.Entry(FIOWindow)
        
#         fLabel.grid()
#         fEntry.grid()

#         iLabel.grid()
#         iEntry.grid()
        
#         oLabel.grid()
#         oEntry.grid()
        
#         okButton = ttk.Button(FIOWindow,command=getFIO)
#         okButton.grid()
#         FIOWindow.grid(column=0)
        
#     # Добавить новую задачу для текущего пользователя
#     def add_new_task(self, tree):
#         global current_user_id
#         if current_user_id is None:
#             messagebox.showwarning("Ошибка", "Выберите пользователя!")
#             return
        
#         taskCombobox = ttk.Combobox(values=self.db["TASKNAME"].get("name"))
#         task_name = taskCombobox.get()
#         if task_name:
#             self.save_task(current_user_id, task_name)
#             self.on_user_select(None, users_listbox, tree)  # Обновить список задач



# DB_NAME = "multi_user_tasks.db"
# current_user_id = None  # Глобальная переменная для хранения ID текущего пользователя

# # Основное окно
# def main():
#     app = tk.Tk()
#     t = TaskManager(master = app)
#     app.mainloop()

# if __name__ == "__main__":
#     main()