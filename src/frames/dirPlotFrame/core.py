from collections.abc import MutableSequence
from itertools import product
import json
from functools import partial, reduce
import operator
from pathlib import Path
import sqlite3
from tkinter import Tk, ttk, BooleanVar

import numpy as np
import scipy as sc
import sympy as sp

from src.base.core import TasksManager
from src.frames.BDManager.core import BDManager
from src.frames.dirPlotFrame.solvers import SolversAgent
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


class PlotAgent:
    def setScatter(self, axs, x, y, label):
        axs.scatter(x, y, label=label, color="red")
        axs.legend(loc='best')
        pass
    def addPlot(self, axs, x, y, label):
        axs.plot(x, y, label=label, color="red")
        axs.legend()
    def setPlot(self, axs, x, y, label, tags):
        xlabel, ylabel = tags

        axs.plot(x, y, label=ylabel+f"({xlabel})", color="black")
        axs.set_title(label=label, size=10)
        axs.set_ylabel(ylabel+f"({xlabel})", rotation=0, size=10)
        axs.set_xlabel(xlabel, rotation=0, size=10)
        axs.yaxis.set_label_coords(-.13, .95)
        axs.legend(loc='best')
        # axs.yaxis.set_label_coords(-.1, .95)
        # axs.xaxis.set_label_coords(1.05, -0.025)
        pass
    def __init__(self, *args, **kwargs) -> None:
        bps = []

        t, xs, u = args
        
        labels = {  0:u"Координата материальной точки",
                    1:u"Скорость материальной точки",
                    2:""}
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.style.use('seaborn-v0_8-whitegrid')
        if isinstance(xs[0], np.ndarray):
            self.fig, self.axs = plt.subplots(2,len(xs[0]), figsize=(9, 7))
            if kwargs["balanceFunction"]:
                bp = kwargs["balanceFunction"]
                for ti in product(*[t]*len(xs[0])):
                    bps.append([*bp(*ti)])
                bps = np.array(bps)
            for i in range(len(xs[0])):
                self.setPlot(self.axs[0][i], t, xs[0:,i], labels[i], ('t',f'$x_{i+1}$'))
            for i, j in combinations(range(len(xs[0])), 2):
                self.setPlot(self.axs[1][i], xs[0:,i], xs[0:,j], fr"Фазовый портрет $x_{i+1}$ $x_{j+1}$", (f'$x_{i+1}$', f'$x_{j+1}$'))
                if kwargs["balancePoint"]:
                    self.setScatter(self.axs[1][i], bps[:,i], bps[:,j], label="Точки равновесия")
                    # self.axs[1][i].scatter(bps[:,i],bps[:,j], color = 'r',marker='o', s=1, label="Точки равновесия")
                if kwargs["balanceLine"]:
                    self.addPlot(self.axs[1][i], bps[:,i], bps[:,j], label="Точки равновесия")
                    # self.axs[1][i].plot(bps[:,i],bps[:,j], color = 'r', label="Точки равновесия")
            self.setPlot(self.axs[-1][-1], t, u, u"Оптимальное управление", ('t', 'u'))
        else:
            self.fig, self.axs = plt.subplots(2,len(xs))
            for i, x in enumerate(xs):
                self.setPlot(self.axs[0][i], t, [x(i) for i in t], labels[i], ('t', f'x_{i+1}'))
            for i, j in combinations(range(len(xs)), 2):
                self.setPlot(self.axs[1][i], [xs[i](l) for l in t], [xs[j](i) for i in t], f"Фазовый портрет x{i+1}(t) x{j+1}(t)", (f'x_{i+1}', f'x_{j+1}'))
            self.setPlot(self.axs[-1][-1], t, [u(i) for i in t], u"Управляемая сила", ('t', 'u'))
        self.fig.tight_layout()
        plt.style.use("default")
        pass
    def getFig(self):
        return self.fig, self.axs
    pass
class ComputeAgent:
    def __init__(self, dfs, fs, bs, t0, tk):
        xs = [sp.Function(f'x_{i+1}') for i in range(len(fs))]
        xbs = bs[0:2*len(fs)]
        self.fs = r"\begin{cases} "
        for f, df in zip(fs, dfs):
            self.fs+=sp.latex(sp.Eq(df, f))+r"\\"
        self.fs+=r"\end{cases}"

        self.bs = r"\begin{cases} "
        for i, x in enumerate(xs):
            self.bs+=sp.latex(sp.Eq(x(t0),xbs[i]))+r" \quad "+sp.latex(sp.Eq(x(tk),xbs[i+2]))+r"\\"
        self.bs+=r"\end{cases}"
        pass
    def get_data(self):
        return self.fs, self.bs
    pass


class Mediator:    
    pass
class ConcreateMediator:
    pass


class PlotFrameModel:
    def __init__(self):
        self.psi0 = []
        self.task = task
        self.taskId = taskId
        self.db = db
        self.dictExperiments = {}
        self.dictAdditional = {}
        self.countExperements = 0
        self.plotList = []
        self.fs = []
        pass
    pass

class PlotFrameView(ttk.Frame):
    def __init__(self, master = None):
        super().__init__(master)
        self.plotFrame = None
        self.psi0Frame = None

        self.sysFrame = ttk.Frame(self)
        self.plotFrame = ttk.Frame(self)
        self.psiFrame = ttk.Frame(self)
        self.addExperimentButton = ttk.Button(self.psiFrame, text="Добавить эксперимент")
        self.removeExperimentButton = ttk.Button(self.psiFrame, text="Убрать эксперимент", state=tk.DISABLED)
        self._psiFrame = ttk.Frame(self.psiFrame)
        self.warningLabel = ttk.Label(self.frame, text="Пожалуйста, выберите начальные значения", justify="center")
        self.solvebtn = ttk.Button(self.frame, text = "Solve")

        self.figSys = Figure(figsize=(5, 5))
        self.figSys.set_facecolor(u'#f0f0f0')

        self.axsSys = self.figSys.subplots(nrows=1, ncols=1)
        self.sysCanvas = FigureCanvasTkAgg(self.figSys, self.sysFrame)

    pass
    def addExperimentButtonFunction(self, f):
        self.addExperimentButton['command'] = f
    def removeExperimentButtonFunction(self, f):
        self.removeExperimentButton['command'] = f
    def changeState(self):
        
        pass
    def grid(self, row, col, rowspan=1, colspan=1, **kwargs):
        super().grid(**kwargs)
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
class ExperimentGUI:
    pass
class ExperimentData:
    pass
class Experiment:
    def __init__(self, *args):
        self.eps = 1e-5
        self.num = 50
        self.method = 'hybr' 
        self.diffmethod='RK45'
        self.findBalancePoint = False
        pass
    def solve(self):
        pass
    pass
class ExperimentBuilder:
    experiment:Experiment
    def setParams(self, **kwargs):
        for key, value in kwargs.items():
            if value:
                match key:       
                    case 'fs':
                        self.experiment.fs = value
                    case 'N':
                        self.experiment.n = int(kwargs[key])
                    case 'Varepsilon':
                        self.experiment.eps = float(value)
                    case 'Num':
                        self.experiment.num = int(value)
                    case 'Diffmethod':
                        self.experiment.diffmethod = value
                    case 'Method':
                        self.experiment.method = value
                    case 'Findbalancepoint':
                        self.experiment.findBalancePoint=True
        pass
    def getExperiment(self)->Experiment:
        pass
    pass
class PontryaginExperimentBuilder(ExperimentBuilder):
    experiment:Experiment
    def __init__(self):
        super().__init__()
        self.experiment = PontryaginExperiment()
    def reset(self):
        self.experiment = PontryaginExperiment()
    def setParams(self, **kwargs):
        super().setParams(**kwargs)
        for key, value in kwargs.items():
            match key:
                case 'tkCurrent':
                    self.experiment.tkCurrent = value
                case 'limit':
                    self.experiment.is_limit = value
                case 'Isbkfun':
                    if value == 'True':
                        self.experiment.isBkFun = True
                        self.experiment.phi = kwargs['phi']
                        pass
                    else:
                        self.experiment.bs = kwargs['bs'][int(kwargs['N']):2*int(kwargs['N'])]
                        pass
        pass
    def getExperiment(self):
        return self.experiment
    pass
class BellmanExperimentBuilder(ExperimentBuilder):
    def __init__(self):
        super().__init__()
        self.experiment = BellmanExperiment()
    def reset(self):
        self.experiment = BellmanExperiment()
    def setParams(self, **kwargs):
        super().setParams(**kwargs)
        for key, value in kwargs.items():
            match key:
                case 'bs':
                    self.experiment.bs = kwargs['bs'][int(kwargs['N']):2*int(kwargs['N'])]
                case 'tk':
                    self.experiment.tk = float(value)
                case 'lam':
                    self.experiment.lam = float(value)
                case 'psi0':
                    self.experiment.psi0 = value
    def getExperiment(self):
        return self.experiment
    pass
class PontryaginExperiment(Experiment):
    def __init__(self, *args):
        super().__init__()
        self.phi = None
        self.isBkFun = False
        pass
    def solve(self, dfs, fs, bs, alpha1=0, alpha2=0, args=(),u_opt=None, **kwargs):
        def is_linear(expr, vars):
            for xy in product(*[vars]*len(vars)):
                    try: 
                        if not sp.Eq(sp.diff(expr, *xy), 0):
                            return False
                    except TypeError:
                        return False
            return True

        t0 = bs[-1]
        b0 = bs[0:self.n]


        xs = [sp.Function(f'x_{i}') for i in range(1, self.n+1)]
        self.psis = list(sp.symbols(f'psi1:{self.n+1}', cls=sp.Function))
        t = sp.Symbol('t')
        u = sp.Function('u')(t)
        self.functional = sp.Integral(alpha1+alpha2*sp.Pow(u, 2), (t, t0, sp.Symbol("t_k")))
        H = sum([psi(t)*f for psi, f in zip(self.psis, fs)])-(alpha1+alpha2*sp.Pow(u, 2))

        if self.phi:
            self.phi = sp.lambdify([x(t) for x in xs], self.phi)

        if H.diff(u, 2) == 0:
            if self.is_limit:
                if (alpha1 + alpha2 * sp.Pow(u, 2)) == 1:
                    u_opt = sp.sign(H.diff(u))
                else:
                    u_opt = sp.Piecewise((-1, H.diff(u)< -1), (1, H.diff(u)>1), (H.diff(u), True))
            else:
                u_opt = H.diff(u)
        else:
            if self.is_limit:
                u_opt_part = sp.solve(H.diff(u), u)[0]
                u_opt = sp.Piecewise((u_opt_part, sp.Abs(u_opt_part) <= 1), (1, u_opt_part > 1), (-1, u_opt_part < -1))
                # u_opt = sp.sign(H.diff(u))
            else:
                u_opt = sp.solve(H.diff(u), u)[0]

        if H.diff(u, 2) == 0:
            if self.is_limit:
                c = sp.Symbol('c')

                F =  sp.lambdify((*[x(t) for x in xs], *[psi(t) for psi in self.psis]),
                     [H.subs(u, c).diff(psi(t)).subs(c, u_opt) for psi in self.psis] + [-H.subs(u, c).diff(x(t)).subs(c, u_opt) for x in xs],'scipy',dummify=True)
            else:
                F = sp.lambdify((*[x(t) for x in xs], *[psi(t) for psi in self.psis]),
                                [H.subs(u, u_opt).diff(psi(t)) for psi in self.psis] + [-H.subs(u, u_opt).diff(x(t)) for
                                                                                        x in xs], ('scipy', 'numpy'),
                                dummify=True)
        else:
            F =  sp.lambdify((*[x(t) for x in xs], *[psi(t) for psi in self.psis]),
                 [H.subs(u, u_opt).diff(psi(t)) for psi in self.psis]+[-H.subs(u, u_opt).diff(x(t)) for x in xs],('scipy','numpy'),dummify=True)

        self.Hf = sp.lambdify((*[x(t) for x in xs], *[psi(t) for psi in self.psis]),H.subs(u, u_opt))
        def F_(*x):
            t, x0 = x
            return F(*x0)

        def Nev(psi0):
            # Проверка корректности конечного времени
            if psi0[-1] < 0:
                return [1e9] * (self.n+1)  # Возвращаем большое число, если время отрицательное

            # Интегрирование системы
            tn = np.linspace(t0, psi0[-1], self.num)
            sol = solve_ivp(F_, [t0, psi0[-1]], np.concatenate((b0,psi0[:-1])), t_eval=tn, method=self.diffmethod, atol=self.eps)
            psiT = sol.y[:, -1]  # Последнее значение решения
            # Выделение переменных состояния
            xf = psiT[:self.n]
            # Возвращаем разность между текущим и целевым состоянием
            if self.isBkFun:
                if self.tkCurrent:
                    return np.concatenate(([self.phi(*xf)]*self.n,[self.Hf(*xf,*psiT[self.n:-1], args[-1])]))
                else:
                    return np.concatenate(([self.phi(*xf)]*self.n, [self.Hf(*xf, *psiT[self.n:])]))
            else:
                if self.tkCurrent:
                    return np.concatenate(([i - k for i, k in zip(xf, self.bk)],[self.Hf(*self.bk,*psiT[self.n:-1], args[-1])]))
                else:
                    return np.concatenate(([i - k for i, k in zip(xf, self.bk)], [self.Hf(*self.bk, *psiT[self.n:])]))

        psi0 = args
        if self.tkCurrent:
            self.psi0 = args[:-1]
        else:
            self.psi0 = args
        y0 = sc.optimize.root(Nev, psi0, method=self.method, tol=self.eps)['x']
        u = sp.lambdify((*[x(t) for x in xs], *[psi(t) for psi in self.psis]), u_opt, modules='numpy')
        if self.tkCurrent:
            tn = np.linspace(t0, args[-1], self.num)
        else:
            tn = np.linspace(t0, y0[-1], self.num)


        if self.findBalancePoint:
            b = sp.solve(fs, [x(t) for x in xs])

            c = sp.symbols(f'c_1:{self.n+1}')
            self.bp = {sp.Symbol(x.name):ci for x,ci in zip(xs,c)}
            defaultbp = self.bp
            su = sp.Function('u')
            xsym = {x(t):sp.Symbol(x.name) for x in xs}

            self.bp.update(sp.solve([sp.Eq(f.subs(xsym).subs({su(t):sp.Symbol(su.name)}),0) for f in fs]))

            kwargs["balanceLine"] = False
            kwargs["balancePoint"] = False
            for key, value in self.bp.items():
                self.bp[key] = self.bp[key].subs({ci:0 for ci in c})
                if self.bp[key] in c:
                    kwargs["balanceLine"] = True
            if not kwargs["balanceLine"]:
                kwargs["balancePoint"] = True
            self.bp = list(self.bp.values())[:-1]
            kwargs["balanceFunction"] = sp.lambdify(c, self.bp)
        y_toch = solve_ivp(F_, [t0, tn[-1]], np.concatenate((b0, y0[:-1])), t_eval=tn, method = self.diffmethod).y.T
        self.psin = y_toch[:,self.n: self.n*2]

        xn = y_toch[:,0:self.n]
        self.xk = xn[-1]
        if self.isBkFun:
            if self.phi(*self.xk)>=0.1:
                return
        else:
            if max([abs(i-j)**2 for i,j in zip(self.xk,self.bk)])>=0.1:
                return
        u = [u(*x, *psi) for x, psi in zip(xn, self.psin)]
        if self.tkCurrent:
            self.tk = args[-1]
        else:
            self.tk = y0[-1]
        self.plot = PlotAgent(tn, xn, u, **kwargs)
        self.H = sp.latex(H)
        self.u_opt = sp.latex(sp.Eq(sp.Function('u')(sp.Symbol('t')), u_opt))
        def get_data(self):
            if self.isBkFun:
                return (self.H,self.Hf(*self.xk, *self.psin[-1])),(self.u_opt, self.is_limit),sp.latex(sp.Eq(sp.Symbol("J"),self.functional)),self.psi0+[self.tk]
            else:
                return (self.H,self.Hf(*self.bk, *self.psin[-1])),(self.u_opt, self.is_limit),sp.latex(sp.Eq(sp.Symbol("J"),self.functional)),self.psi0+[self.tk]

        return super().solve()
    pass
class BellmanExperiment(Experiment):
    def __init__(self):
        super().__init__()
    def solve(self):
        t0 = self.bs[-1]
        xs = [sp.Function(f'x_{i}') for i in range(1, self.n+1)]
        bordersk = self.bs[0:self.n]
        S = sp.Function('S')(t, *[x(t) for x in xs])
        u  = sp.Symbol('u')
        fys = self.fs[:self.n]
        fys = [f.subs({x(t):sp.Symbol(x.name) for x in xs}).subs(sp.Function('u')(t), u) for f in fys]
        ds = [sp.Derivative(S, x(t)) for x in xs]
        self.functional = sp.Integral(u**2, (t, 0, sp.Symbol("t_k")))+sp.Symbol("lambda")*fs[-1]
        H = sum(f*s for f, s in zip(fys, ds))+u**2
        # if self.findBalancePoint:
        #     b = sp.solve(self.fs, [x(t) for x in xs])
        #     c = sp.symbols(f'c_1:{self.n+1}')
        #     self.bp = {sp.Symbol(x.name):ci for x,ci in zip(xs,c)}
        #     defaultbp = self.bp
        #     su = sp.Function('u')
        #     xsym = {x(t):sp.Symbol(x.name) for x in xs}

        #     self.bp.update(sp.solve([sp.Eq(f.subs(xsym).subs({su(t):sp.Symbol(su.name)}),0) for f in fys]))
        #     kwargs["balanceLine"] = False
        #     kwargs["balancePoint"] = False
        #     for key, value in self.bp.items():
        #         self.bp[key] = self.bp[key].subs({ci:0 for ci in c})
        #         if self.bp[key] in c:
        #             kwargs["balanceLine"] = True
        #     if not kwargs["balanceLine"]:
        #         kwargs["balancePoint"] = True

        #     self.bp = list(self.bp.values())[:-1]
        #     kwargs["balanceFunction"] = sp.lambdify(c, self.bp)
        power_sym = sp.symbols(f'p_1:{len(xs)+1}')
        xij = []
        c = 0
        ss = []
        for ij in product(*(range(len(fys)+1) for _ in range(len(xs)))):
            if sum(ij) <= len(fys):
                xij.append(reduce(operator.mul, [sp.Symbol(x.name)**powsym for x, powsym in zip(xs, power_sym)]))
                xij[-1] = xij[-1].subs({p:i for p, i in zip(power_sym, ij)})
                c+=1
                ss.append(sp.Function(f's_{c}'))

        expr = sum(s(t)*x for s, x in zip(ss, xij))

        u_kr = sp.solve(H.diff(u), u)[0]

        dxexpr = {S.diff(x(t)):expr.diff(sp.Symbol(x.name)) for x in xs}
        H = H.subs(u, u_kr).subs(dxexpr).expand()
        ST = self.lam*self.fs[-1].subs(t, tk).subs({x(tk):sp.Symbol(x.name) for x in xs})
        expandExpr = expr.subs({s(t):sp.Symbol(s.name) for s in ss})-ST
        systemEq = []
        for x in xij[::-1]:
            systemEq.append(expandExpr.coeff(x))
            expandExpr -= expandExpr.coeff(x) * x
            expandExpr = expandExpr.expand()

        y0Solve = sp.solve(systemEq, [sp.Symbol(s.name) for s in ss],dict=True)[0]
        y0 = []
        for key, value in y0Solve.items():
            y0.append(value)

        PolyH = sp.Poly(H, [sp.Symbol(x.name) for x in xs])

        ns = [sp.lambdify((*(sp.Symbol(s.name) for s in ss),t), f.subs({s(t):sp.Symbol(s.name) for s in ss})) for f in list(PolyH.as_dict().values())]
        def _ns(*x):
            t,y = x
            return [n(*y,t) for n in ns]

        nt = np.linspace(t0, tk, self.num)

        f = solve_ivp(_ns,
                      [t0, tk],
                      y0,
                      t_eval=nt,
                      method=self.diffmethod,
                      rtol=1e-8,  # Уменьшаем допустимую ошибку
                      atol=1e-8,
                      ).y.T

        sSpline = []
        for i in range(len(f[0])):
            sSpline.append(CubicSpline(nt, f[:,i]))
        for i, f in enumerate(fys):
            fys[i] = f.subs(u, u_kr.subs(dxexpr)).subs({sp.Symbol(x.name):x(t) for x in xs})
        lfys = []

        for f in fys:
            lfys.append(sp.lambdify((t, *(s(t) for s in ss), *(x(t) for x in xs)), f.subs(dxexpr)))
        def dfys(t, x):
            return [f(t, *(s(t) for s in sSpline), *x) for f in lfys]
        u = sp.lambdify((t, *(s(t) for s in ss), *(sp.Symbol(x.name) for x in xs)), u_kr.subs(dxexpr))
        xs = solve_ivp(dfys, [t0, tk], bordersk, t_eval=nt, method=self.diffmethod).y.T
        _u = u(nt, *[s(nt) for s in sSpline], *[xs[0:,i] for i in range(self.n)])

        bbs = ([0]*self.n+self.bs[0:self.n])
        bbs.reverse()

        self.u_opt = sp.latex(sp.Eq(sp.Function('u')(sp.Symbol('t')),u_kr.subs(dxexpr)))
        self.H = sp.latex(H)
        self.plot = PlotAgent(nt, xs, _u, **kwargs)
        def get_data(self):
            return self.H,self.u_opt, sp.latex(sp.Eq(sp.Symbol("J"),self.functional)),self.psi0

    pass
class Experiments(MutableSequence):
    experiments:list[Experiment] = []
    def __getitem__(self, index):
        return self.experiments[index]
    def __setitem__(self, index, item:Experiment):
        self.experiments[index] = item
    def __delitem__(self, index):
        self.experiments = self.experiments[:index]+self.experiments[index+1:]
    def __iter__(self):
        for experiment in self.experiments:
            yield experiment
    def __len__(self):
        return len(self.experiments)
    def insert(self, item:Experiment):
        self.experiments(item)
class GUI:
    pass
class App:
    expirements: Experiments
    plot:GUI
    def solve(self):
        for experiment in self.experiments:
            self.plot.get(experiment.solve())
    pass
class Command:
    app:App
    def execute(self):
        pass
class AddExperimentCommand(Command):
    def execute(self):
        self.app.add()
        pass
    pass
class RemoveExperimentCommand(Command):
    def execute(self):
        self.app.remove()
        pass
    pass
class ApplyExperimentCommand(Command):
    def execute(self):
        self.app.apply()
        pass
    pass
class SolveExperimentCommand(Command):
    def execute(self):
        self.app.solve()
        pass
    pass
class PlotFrameController:
    def __init__(self, model:PlotFrameModel, view:PlotFrameView):
        self.model = model
        self.view = view
        self.view.addExperimentButtonFunction(self.addExperiment) 
        self.view.removeExperimentButtonFunction(self.removeExperiment) 
        pass
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
    pass


class PlotFrameRef:
    def __init__(self, frame):
        self.model = PlotFrameModel()
        self.view = PlotFrameView(frame)
        self.controller = PlotFrameController(self.model, self.view)
        pass








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
    def __init__(self, master, db:BDManager,task,taskId, nextFrame=None):
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
        self.phi = None
        self.root = master
        self.task = task
        self.taskId = taskId
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
        print(self.task)
        self.n = self.db["TASKINFO"].get("N")[0][0]
        self.isBkFun = self.db["TASKINFO"].get("Isbkfun")[0][0]
        print(self.isBkFun)
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
            match self.task:
                case self.TASKS.metodDyn.value:
                    for i, file in enumerate(p):
                        if i<self.n+1:
                            with open(file, "rb") as f:
                                self.fs[i]=dill.load(f)
                case _:
                    for i, file in enumerate(p):
                        if i < self.n:
                            with open(file, "rb") as f:
                                self.fs[i] = dill.load(f)
        d = Path("MyLab/data/dfs/").glob('**/*')
        for i, file in enumerate(d):
            if i<self.n:
                with open(file, "rb") as f:
                    self.dfs[i]=dill.load(f)
        self.update()
        pass
    def setPsi0(self):
        # self.task = self.db.execute()["TASKINFO"].get("taskNameId")[0][0]
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
        bd = self.db.execute(f"SELECT value FROM BORDERDB WHERE id={self.taskId}")[0][0].split(sep="t")
        match self.task:
            case  self.TASKS.metodDyn.value:
                t0 = bd[self.n]
                bd0 = [float(i) for i in bd[:self.n]]
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

                #  = self.db["BORDERDB"].get()[-1][1]
                
                t0 = bd[-1]
                bd0 = [float(i) for i in bd[:self.n]]
                if self.isBkFun == "True":
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
                    bdk = [float(i) for i in bd[self.n:2 * self.n]]
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
        l = ""
        l+=self.to_fs_latex(lhs, rhs, usetex)+self.to_bs_latex(usetex)
        print(l)
        return l
    def close(self):
        pass
    def solve(self):
        # self.bs = [float(i[1]) for i in self.db["BORDERDB"].get()]
        self.bs = self.db.execute(f"SELECT value FROM BORDERDB WHERE id={self.taskId}")[0][0]
        self.bs = [float(i) for i in self.bs.split(sep="t")]
    
        self.kwargs.update({opt:val for opt, val in zip(self.db["TASKINFO"].values+self.db["SOLVEINFO"].values, *[self.db.execute(f"SELECT * FROM TASKINFO WHERE id={self.taskId}")[0]+self.db.execute(f"SELECT * FROM SOLVEINFO WHERE id={self.taskId}")[0]])})
        print(self.kwargs)
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