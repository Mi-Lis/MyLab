import pathlib
import sqlite3

class BD:
    """
    Класс для взаимодействия с таблицей в базе данных.

    Arguments
    ---------
    NAME:str
        Название базы данных
    rowNumber:int
        Кол-во строк в таблице
    prompt:str
        SQL-запрос
    ARGS:list[str]
        Список названий аргументов таблицы
    """
    NAME:str
    rowNumber:int
    prompt:str
    ARGS:list[str]
    def __init__(self, bdInfo:dict, connection:sqlite3.Connection=None):
        """
        Конструктор создает таблицу в базе данных
        Parameters
        ----------
        bdInfo:dict
            Словарь со структурой таблицы
        connection:sqlite3.Connection
            Экземпляр класса sqlite3.Connection 
        """
        self.ARGS = []
        self.connection = connection
        self.cursor = self.connection.cursor()
        self.values = []
        argInfo = None
        for key, value in bdInfo.items():
            self.NAME = key
            argInfo = value
        
        self.rowNumber = len(argInfo.keys())
        prompt = "CREATE TABLE IF NOT EXISTS {0} ({1});"
        temp = ""
        i = 0
        for key, value in argInfo.items():
            if i<self.rowNumber-1:
                temp+=key+' '+value+','
            else:
                temp+=key+' '+value
            self.values.append(key.capitalize())
            i+=1
        prompt = prompt.format(self.NAME, temp)
        self.cursor.execute(prompt)
        pass

    def check_execute(attribute_name):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                func(self, *args, **kwargs)
                # Проверяем, что атрибут класса имеет ожидаемое значение
                cursor = getattr(self, attribute_name)
                try:
                    cursor.execute("COMMIT")
                except sqlite3.IntegrityError:
                    cursor.execute("ROLLBACK")
            return wrapper
        return decorator
    @check_execute('cursor')
    def delete(self):
        """
        Метод удаляет таблицу из БД.
        """
        prompt = f"""DELETE FROM {self.NAME}"""
        self.cursor.execute(prompt)
        pass
    @check_execute('cursor')
    def add(self, *items):
        """
        Метод добавляет новую строку в таблицу.
        """
        noneValues = []
        while len(items)+len(noneValues)<len(self.values):
            noneValues.append(0)
        prompt = "INSERT INTO {0} ({1}) VALUES ({2});"
        args = ", ".join(self.values)
        values = ", ".join([str(item) for item in items+tuple(noneValues)])
        prompt = prompt.format(self.NAME, args,values)
        try: self.cursor.execute(prompt)
        except sqlite3.IntegrityError: 
            self.delete()
            self.cursor.execute(prompt)
    @check_execute('cursor')
    def update(self, *args):
        prompt = f"UPDATE {self.NAME} SET {",".join([f"{val}='{str(arg)}'" for val, arg in zip(args[::2], args[1::2])])};"
        self.cursor.execute(prompt)
        pass
    def get(self, *args):
        """
        Метод получает данные из таблицы, если args = None получает всю таблицу, если args != None получает данные выбранных столбцов.
        """
        if args != ():
            for arg in args:
                if not arg.capitalize() in self.values:
                    return None
        if args == ():
            prompt = f"SELECT * FROM {self.NAME};"
        else:
            prompt = f"SELECT {", ".join([str(i) for i in args])} FROM {self.NAME}"
        return self.cursor.execute(prompt).fetchall()
        pass
    def update_by_id(self, *args, id):
        """
        Обновляет значения в строке с указанным id.
        TODO: Добавить обработку исключений args == None, args содержит недопустимые значения.
        """
        prompt = f"UPDATE {self.NAME} SET {",".join([f"{val}='{str(arg)}'" for val, arg in zip(self.values[1:], args)])} WHERE id = {id};"
        self.cursor.execute(prompt)
        pass
    def remove(self, id):
        """
        Удаляет строку с указанным id.
        """
        if id>self.rowNumber:
            return None
        prompt = f"DELETE FROM {self.NAME} WHERE id={id};"
        self.cursor.execute(prompt)
    @check_execute('cursor')
    def execute(self, prompt):
        """
        Выполняет пользовательский sql-запрос.
        """
        return self.cursor.execute(prompt).fetchall()
    def __str__(self):
        return str(self.get())
        pass
    def updateConnection(self, newConnection):
        self.connection = newConnection
        self.cursor = self.connection.cursor()
        pass
    pass

class BDManager:
    """
    Класс для работы с таблицами в базе данных.
    """
    PATH = "MyLab/data"
    NAME = "functionList.db"
    def __init__(self, bdDict:dict, path=None):
        """
        Parameters
        ----------
        bdDict:dict
            Структура базы данных
        parh:str
            Путь к папке сохранение .sql файла
        """
        self.bdDict  = bdDict
        if path:
            self.PATH = path
        p = pathlib.Path(self.PATH)/self.NAME
        if not p.exists():
            with p.open("w") as f:
                pass
        self.connection = sqlite3.Connection(p)
        self.data = {name:BD({name:bd}, connection = self.connection) for name, bd in bdDict.items()}
        pass
    def update(self):
        p = pathlib.Path(self.PATH)/self.NAME
        self.connection = sqlite3.Connection(p)
        # for key in self.data.keys():
        #     self.data[key].updateConnection(self.connection)
        # self.data = {name: BD({name: bd}, connection=self.connection) for name, bd in self.bdDict.items()}
    def __getitem__(self, item):
        """
        Получение таблицы по названию.
        """
        return self.data[item]
    def __del__(self):
        """
        Удаление базы данных.
        """
        for bd in self.data.values():
            bd.delete()
        self.connection.close()
    
    pass
if __name__ == '__main__':
    bd = {
            "FUNDB":
            {
                "id":"INTEGER PRIMARY KEY",
                "TexView": "Text Not Null",
                "SymView": "Text"
            }
            ,
            "TASKINFO":
            {
                "id":"INT PRIMARY KEY",
                "value":"TEXT NOT NULL"
            }
            ,
           "BORDERDB":
            {
               "id":"INTEGER PRIMARY KEY",
               "value":"TEXT NOT NULL"
            }
            ,
            "FUNCTIONALDB":
            {
                "id":"INTEGER PRIMARY KEY",
                "value":"TEXT NOT NULL"
            }
        }  
    
    bd = BDManager(bd)
    bd["FUNDB"].add(1,2)
    bd["FUNDB"].add(2,3)
    print(bd["FUNDB"].get())
    bd["FUNDB"].remove(2)
    print(bd["FUNDB"])
    pass