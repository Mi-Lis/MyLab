from functools import reduce
from itertools import combinations, product
import operator
import sympy as sp
import scipy as sc
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import matplotlib as mpl

from src.base.core import TasksManager

t = sp.Symbol('t')
t0 = sp.Symbol('t0')
ti = sp.Symbol('ti')
tk = sp.Symbol('tk', real=True)
x1 = sp.Function('x_1')
x2 = sp.Function('x_2')
x3 = sp.Function('x_3')
x10 = sp.Symbol('x10')
x20 = sp.Symbol('x20')
x1k = sp.Symbol('x1k')
x2k = sp.Symbol('x2k')
c = sp.Symbol('c')
t0 = 0
# tk = 1
x10 = 1
x20 = 0
x1k = 2
x2k = 1

class SolversAgent(TasksManager):
    typeAlg:str
    def __init__(self, typeAlg, dfs, fs, bs, psi0=[], **kwargs):
        self.typeAlg = typeAlg
        self.dfs = dfs
        self.fs = fs
        self.bs = bs
        self.psi0 = psi0
        self.kwargs = kwargs
        pass
    def solver(self):
        match self.typeAlg:
            case self.TASKS.naiskDvi.value:
                return PontryaginMovementOptimal(self.dfs, self.fs, self.bs, self.psi0, **self.kwargs)
            case self.TASKS.linSys.value:
                return PontryaginCustomOptimal(self.dfs, self.fs, self.bs, self.psi0, **self.kwargs)
            case self.TASKS.enOpt.value:
                return PontryaginEnergyOptimal(self.dfs, self.fs, self.bs, self.psi0, **self.kwargs)
            case self.TASKS.metodDyn.value:
                return  BellmanCompute(self.dfs, self.fs, self.bs, self.psi0, **self.kwargs)

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
class PontryaginCompute:
    def __init__(self, dfs, fs, bs, alpha1=0, alpha2=0, args=(),u_opt=None, **kwargs) -> None:
        def is_linear(expr, vars):
            for xy in product(*[vars]*len(vars)):
                    try: 
                        if not sp.Eq(sp.diff(expr, *xy), 0):
                            return False
                    except TypeError:
                        return False
            return True
        self.eps = 1e-5
        self.num = 50
        self.method = 'hybr' 
        self.diffmethod='RK45'
        self.findBalancePoint = False
        self.phi = None
        self.isBkFun = False
        for key, value in kwargs.items():
            match key:
                case 'tkCurrent':
                    self.tkCurrent = kwargs[key]
                case 'limit':
                    self.is_limit = kwargs[key]
                case 'N':
                    self.n = int(kwargs[key])
                case 'Varepsilon':
                    if value!='':
                        self.eps = float(value)
                case 'Num':
                    if value!='':
                        self.num = int(value)
                case 'Diffmethod':self.diffmethod = value
                case 'Method':self.method = value
                case 'Findbalancepoint':
                    if value:
                        self.findBalancePoint=True
                case 'Isbkfun':
                    if value == 'True':
                        self.isBkFun = True
                        self.phi = kwargs['phi']
                        pass
                    else:
                        self.bk = bs[self.n:2*self.n]
                        pass
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
class PontryaginCustomOptimal(PontryaginCompute):
    def __init__(self, dfs, fs, bs:list, args=(), **kwargs):
        alpha2 = bs[-2]
        alpha1 = bs[-1]
        bs = bs[:-2]
        print(bs)
        super().__init__(dfs, fs, bs, alpha1, alpha2, args, **kwargs)
class PontryaginEnergyOptimal(PontryaginCompute):
    def __init__(self, dfs, fs, bs, args = (), **kwargs):
        alpha1 = 1
        alpha2 = 1
        super().__init__(dfs, fs, bs, alpha1, alpha2, args, **kwargs)
class PontryaginMovementOptimal(PontryaginCompute):
    def __init__(self, dfs, fs, bs, args = (), **kwargs):
        alpha1 = 1
        alpha2 = 0
        super().__init__(dfs, fs, bs, alpha1, alpha2, args, **kwargs)
class PontryaginSymbolCustom:
    pass
class BellmanCompute:
    def __init__(self, dfs, fs, bs,args=(), **kwargs):
        self.eps = 1e-5
        self.num = 50
        self.method = 'hybr'
        self.diffmethod='RK45'
        self.findBalancePoint = False
        for key, value in kwargs.items():
            match key:
                case 'N':
                    self.n = int(kwargs[key])
                case 'Varepsilon':
                    if value!='':
                        self.eps = float(value)
                case 'Num':
                    if value!='':
                        self.num = int(value)
                case 'Diffmethod':self.diffmethod = value
                case 'Method':self.method = value
                case 'Findbalancepoint':
                    if value == 'True':
                        value = True
                    else:
                        value = False
                    if bool(value):
                        self.findBalancePoint=True
        lam, tk = args
        self.psi0 = args
        t0 = bs[-1]
        xs = [sp.Function(f'x_{i}') for i in range(1, self.n+1)]
        bordersk = bs[0:self.n]
        S = sp.Function('S')(t, *[x(t) for x in xs])
        u  = sp.Symbol('u')
        fys = fs[:self.n]
        fys = [f.subs({x(t):sp.Symbol(x.name) for x in xs}).subs(sp.Function('u')(t), u) for f in fys]
        ds = [sp.Derivative(S, x(t)) for x in xs]
        self.functional = sp.Integral(u**2, (t, 0, sp.Symbol("t_k")))+sp.Symbol("lambda")*fs[-1]
        H = sum(f*s for f, s in zip(fys, ds))+u**2
        if self.findBalancePoint:
            b = sp.solve(fs, [x(t) for x in xs])
            c = sp.symbols(f'c_1:{self.n+1}')
            self.bp = {sp.Symbol(x.name):ci for x,ci in zip(xs,c)}
            defaultbp = self.bp
            su = sp.Function('u')
            xsym = {x(t):sp.Symbol(x.name) for x in xs}

            self.bp.update(sp.solve([sp.Eq(f.subs(xsym).subs({su(t):sp.Symbol(su.name)}),0) for f in fys]))
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
        ST = lam*fs[-1].subs(t, tk).subs({x(tk):sp.Symbol(x.name) for x in xs})
        expandExpr = expr.subs({s(t):sp.Symbol(s.name) for s in ss})-ST
        systemEq = []
        for x in xij[::-1]:
            systemEq.append(expandExpr.coeff(x))
            expandExpr -= expandExpr.coeff(x) * x
            expandExpr = expandExpr.expand()

        y0Solve = sp.solve(systemEq, [sp.Symbol(s.name) for s in ss],dict=True)[0]
        y0 = []
        for key, value in y0Solve.items():
            y0.append(y0Solve[key])

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

        bbs = ([0]*self.n+bs[0:self.n])
        bbs.reverse()

        self.u_opt = sp.latex(sp.Eq(sp.Function('u')(sp.Symbol('t')),u_kr.subs(dxexpr)))
        self.H = sp.latex(H)
        self.plot = PlotAgent(nt, xs, _u, **kwargs)
    def get_data(self):
        return self.H,self.u_opt, sp.latex(sp.Eq(sp.Symbol("J"),self.functional)),self.psi0

if __name__=='__main__':
    kwargs = {"balanceFunction":None, "balancePoint":None, "balanceLine":None}
    u = sp.Function('u')(t)
    f0 = 1+1*u**2
    f1 = x2(t)
    f2 = -x2(t)+u
    
    xs = [x1, x2]
    fys = [f1, f2]
    dfys = [x1.diff(t), x2.diff(t)]
    # PontryaginCompute(dfys, fys, [1,0,0,0], 1,1, [3,2,3])
    BellmanCompute(dfys, fys, [1,0,0,10], **kwargs)
    plt.show()
    pass

# class PontryaginSymbol(ComputeAgent):
#     def __init__(self, dfs, fs, bs) -> None:

#         x1 = sp.Function('x_1')
#         x2 = sp.Function('x_2')
#         u1 = sp.Function('u')
#         t = sp.Symbol('t')
#         xs = x1, x2
#         x10, x20, x1k, x2k, t0 = bs
#         L1 = sp.exp(-(x1k-x10+x2k-x20+t0))
#         L2 = sp.exp(-(x10-x1k+x20-x2k-t0))

#         def ind1(t, ti):
#             if t>ti:
#                 return 1
#             else:
#                 return 0
#         def ind0(t, ti):
#             if t<=ti:
#                 return 1
#             else:
#                 return 0
#         if L1>=(sp.exp(t0)*(1-x2k)*(1+x20)) and x10>0:
#             eqs = [(i-j).subs(u1(t), -1) for i, j in zip(dfs, fs)]
#             firstpart = sp.dsolve(eqs, [x1(t), x2(t)], ics={x1(t0):x10, x2(t0):x20})
#             x11 = sp.lambdify(t, firstpart[0].rhs, modules=["numpy", "sympy"])
#             x21 = sp.lambdify(t, firstpart[1].rhs, modules=["numpy", "sympy"])
#             if (x1k-x10>0 and x2k-x20>0):
#                 u = lambda t: -1
#                 tkc = -(x1k-x10+x2k-x20)-t0
#                 t = np.linspace(t0, tkc)
#                 x1 = lambda t: x11(t)
#                 x2 = lambda t: x21(t)

#             else:
#                 u = lambda t: -1 if t<tic else 1

#                 eqs = [(i-j).subs(u1(t), 1) for i, j in zip(dfs, fs)]
#                 secpart = sp.dsolve(eqs, [x1(t), x2(t)])
#                 cij = sp.solve([firstpart[0].rhs.subs(t, ti)-secpart[0].rhs.subs(t, ti), firstpart[1].rhs.subs(t, ti)-secpart[1].rhs.subs(t, ti)])
#                 secpart = [sp.Eq(secpart[0].lhs, secpart[0].rhs.subs(cij[0])),
#                            sp.Eq(secpart[1].lhs, secpart[1].rhs.subs(cij[0]))]
#                 D = sp.sqrt(4*L1**2-4*(1+x20)*(1-x2k)*sp.exp(t0))
#                 tic = sp.ln((2*L1+D)/(2*(1+x2k)))
#                 tkc = x1k-x10+x2k-x20+2*tic-t0

#                 self.u_opt = sp.latex(sp.Eq(sp.Function('u')(t),sp.Piecewise((-1, t<float(tic.evalf())), (1, True))))

#                 secpart = [sp.Eq(secpart[0].lhs, secpart[0].rhs.subs(ti, tic)),
#                 sp.Eq(secpart[1].lhs, secpart[1].rhs.subs(ti, tic))]

#                 x12 = sp.lambdify(t, secpart[0].rhs, modules=["numpy", "sympy"])
#                 x22 = sp.lambdify(t, secpart[1].rhs, modules=["numpy", "sympy"])

#                 t1 = np.linspace(t0, float(tic.evalf()))
#                 t2 = np.linspace(float(tic.evalf()), float(tkc.evalf()))
#                 t = np.hstack((t1, t2))
#                 x1 = lambda t: x11(t)*ind0(t, tic.evalf())+x12(t)*ind1(t, tic.evalf())
#                 x2 = lambda t: x21(t)*ind0(t, tic.evalf())+x22(t)*ind1(t, tic.evalf())

#         if L2>=(sp.exp(t0)*(1+x2k)*(1-x20)) and x10<0:
#             eqs = [(i-j).subs(u1(t), 1) for i, j in zip(dfs, fs)]
#             firstpart = sp.dsolve(eqs, [x1(t), x2(t)], ics={x1(t0):x10, x2(t0):x20})
#             x11 = sp.lambdify(t, firstpart[0].rhs, modules=["numpy", "sympy"])
#             x21 = sp.lambdify(t, firstpart[1].rhs, modules=["numpy", "sympy"])
#             if (-x1k+x10<0 and -x2k+x20<0):
#                 u = lambda t: 1
#                 tkc = (x1k-x10+x2k-x20)+t0
#                 t = np.linspace(t0, tkc)
#                 x1 = lambda t: x11(t)
#                 x2 = lambda t: x21(t)
#             else:
#                 u = lambda t: 1 if t<tic else -1

#                 eqs = [(i-j).subs(u1(t), -1) for i, j in zip(dfs, fs)]
#                 secpart = sp.dsolve(eqs, [x1(t), x2(t)])
#                 cij = sp.solve([firstpart[0].rhs.subs(t, ti)-secpart[0].rhs.subs(t, ti), firstpart[1].rhs.subs(t, ti)-secpart[1].rhs.subs(t, ti)])
#                 secpart = [sp.Eq(secpart[0].lhs, secpart[0].rhs.subs(cij[0])),
#                     sp.Eq(secpart[1].lhs, secpart[1].rhs.subs(cij[0]))]
#                 D = sp.sqrt(4*L2**2-4*(1+x2k)*(1-x20)*sp.exp(t0))
#                 tic = sp.ln((2*L2+D)/(2*(1+x2k)))
#                 tkc = x10-x1k+x20-x2k+2*tic+t0

#                 self.u_opt = sp.latex(sp.Eq(sp.Function('u')(t),sp.Piecewise((1, t<float(tic.evalf())), (-1, True))))

#                 secpart = [sp.Eq(secpart[0].lhs, secpart[0].rhs.subs(ti, tic)),
#                     sp.Eq(secpart[1].lhs, secpart[1].rhs.subs(ti, tic))]
#                 x12 = sp.lambdify(t, secpart[0].rhs, modules=["numpy", "sympy"])
#                 x22 = sp.lambdify(t, secpart[1].rhs, modules=["numpy", "sympy"])

#                 t1 = np.linspace(t0, float(tic.evalf()))
#                 t2 = np.linspace(float(tic.evalf()), float(tkc.evalf()))
#                 t = np.hstack((t1, t2))
#                 x1 = lambda t: x11(t)*ind0(t, tic.evalf())+x12(t)*ind1(t, tic.evalf())
#                 x2 = lambda t: x21(t)*ind0(t, tic.evalf())+x22(t)*ind1(t, tic.evalf())

#         super().__init__(dfs, fs, bs, t0, float(tkc.evalf()))
#         self.H = ""
#         self.tic = float(tic.evalf())
#         self.tkc = float(tkc.evalf())
#         self.plot = PlotAgent(t,(x1,x2),u)
#     def get_data(self):
#         return *super().get_data(), self.H, self.u_opt, self.plot.getFig()
#         pass
