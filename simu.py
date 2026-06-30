import os
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import random
from typing import Literal
from tqdm import tqdm
from itertools import product
import statistics as stat

test=True

#configuration : classe de paramètres. par défaut short term 
class config:
    #initialisation
    def __init__(
        self,
        workdir=os.path.dirname(os.path.abspath(__file__)),
        datadir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        term='short',
        simu_title=None,
        B0=None,
        b0=None,
        epsiloneq=None,
        Lambda=None, 
        kappaeq=None,
        tauepsilon=None, 
        taukappa=None, 
        deltaepsilon=None,
        deltakappa=None, 
        inter_kappa=None,
        inter_epsilon=None, 
        nu=None,
        dt=0.01,
        tfin=50
        ):
            self.workdir = workdir
            # prefer an existing executable (dynamo.exe on Windows, dynamo elsewhere)
            exe_candidates = [os.path.join(workdir, "dynamo.exe"), os.path.join(workdir, "dynamo")]
            self.exec_path = None
            for p in exe_candidates:
                if os.path.exists(p):
                    self.exec_path = p
                    break
            # fallback to the expected name if none found (will raise a clear error later)
            if self.exec_path is None:
                self.exec_path = os.path.join(workdir, "dynamo.exe")
            self.datadir = datadir #dossier dans lequel sont stockées toutes les simulations

            # initialize randomized defaults when not provided
            if B0 is None :
                B0=10**random.uniform(-3, 1)
            if b0 is None :
                b0=10**random.uniform(-3, 1)
            if epsiloneq is None:
                epsiloneq = random.choice([-0.1,-0.01,0,0.01,0.1])
            if Lambda is None:
                Lambda = random.choice([0.1,1])
            if kappaeq is None:
                kappaeq = random.choice([0.01,0.1,1])
            if tauepsilon  is None:
                tauepsilon = random.choice([10**2,10**3,10**4])
            if taukappa is None:
                taukappa = random.choice([10**3,10**4,10**5])
            if deltaepsilon is None:
                deltaepsilon = random.choice([10**(-3),10**(-4),10**(-5)])
            if deltakappa is None:
                deltakappa = random.choice([10**(-3),10**(-4),10**(-5)])
            if inter_kappa is None :
                inter_kappa=True
            if inter_epsilon is None : 
                inter_epsilon=True
            if nu is None:
                nu = 10**random.uniform(-6, -3)
            if simu_title is None :
                simu_title="simu_"

            simu_number = 1
            while os.path.exists(os.path.join(self.datadir, simu_title + str(simu_number))):
                simu_number += 1
            self.folder = os.path.join(self.datadir, simu_title + str(simu_number))
            os.makedirs(self.folder, exist_ok=True)
            self.config_in = os.path.join(self.folder,"configuration.in")
            self.stat_file = os.path.join(self.folder,"minimas.txt")

            self.B0=B0
            self.b0=b0

            self.epsiloneq=5 if term=="long" else epsiloneq
            self.Lambda=Lambda
            self.kappaeq=kappaeq

            self.tauepsilon=tauepsilon
            self.deltaepsilon=deltaepsilon
            self.taukappa=taukappa
            self.deltakappa=deltakappa

            self.inter_kappa=inter_kappa
            self.inter_epsilon=inter_epsilon

            self.nu=nu

            self.dt=dt
            self.tfin=tfin
            self.term = term
            if term == 'short':
                self.tauepsilon = 0
                self.taukappa = 0
                self.deltaepsilon = 0
                self.deltakappa = 0
                self.nu = 0
            elif term == 'mid':
                self.nu=0

    #solutions analytiques stationnaires
    def get_eq(self):
        epsilon = self.epsiloneq
        Lambda = self.Lambda
        kappa = self.kappaeq

        tol=1e-10 # x=0--> abs(x)<tol ; x<0--> x<-tol ; x>0--> x>tol
        div=1e10

        c1=(kappa**3)-Lambda
        c2=(kappa**2)*epsilon + 2*Lambda
        c3=-Lambda
        Delta=c2**2-4*c1*c3

        sol=[]

        def landau():
            return [(0, 1)]
        
        def verif_cond(x):
            if (max(0,-epsilon/kappa)-x<-tol) and (x-1<-tol) and x>0 and abs(x)<div : 
                b_eq=sqrt(x)
                B_eq=sqrt((epsilon+kappa*x)/Lambda)
                sol.append((B_eq,b_eq))

        if abs(c1)<tol : #pour éviter de diviser par 0 plus tard
                if abs(c2)<tol : sol=landau()
                else : 
                    x=-c3/c2
                    verif_cond(x)
        elif Delta<-tol : sol=landau()
        elif abs(Delta)<tol : 
            x=-c2/(2*c1)
            verif_cond(x)
        else : 
            xplus=(-c2+sqrt(Delta))/(2*c1)
            verif_cond(xplus)
            xmoins=(-c2-sqrt(Delta))/(2*c1)
            verif_cond(xmoins)
        if len(sol)==0 : sol=landau()
        return sol
        
    #écrire configuration.in
    def write_config_file(self, differentfile=None):
        eq=self.get_eq()
        (Beq,beq)=eq[0]
        text = f"""
            B0 = {self.B0}
            b0 = {self.b0}

            epsiloneq = {self.epsiloneq}
            Lambda = {self.Lambda}
            kappaeq = {self.kappaeq}

            tauepsilon = {self.tauepsilon}
            taukappa = {self.taukappa}
            deltaepsilon = {self.deltaepsilon}
            deltakappa = {self.deltakappa}

            inter_kappa = {'true' if self.inter_kappa else 'false'}
            inter_epsilon = {'true' if self.inter_epsilon else 'false'}

            term = {self.term}

            nu = {self.nu}

            dt = {self.dt}
            tfin = {self.tfin}

            Beq = {Beq}
            beq = {beq}
        """.lstrip()
        target_file = self.config_in if differentfile is None else differentfile
        target_dir = os.path.dirname(target_file)
        os.makedirs(target_dir, exist_ok=True)
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(text)

    def delete_folder(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    #lanceur de simulation de base, mettre le output avec configuration.in
    def run(self, outputname='output.txt', save=False):
        self.dt=min(self.dt, 0.01/max(self.epsiloneq, self.Lambda, self.kappaeq))
        os.makedirs(self.folder, exist_ok=True)
        self.write_config_file()
        # The C++ program expects the directory containing configuration.in as argv[1].
        # Pass `datadir` (not the full config file path).
        if not os.path.exists(self.exec_path):
            raise FileNotFoundError(f"Executable not found: {self.exec_path}\nCompile the C++ code first (e.g. run `make` in {self.workdir}).")
        subprocess.run([self.exec_path, self.folder], cwd=self.workdir, check=True)
        data_path = os.path.join(self.folder, outputname)
        try:
            data = np.genfromtxt(data_path)
            return data
        finally:
            if not save:
                self.delete_folder()
    
    #plot temporel
    possible_plots = Literal["Bb","B","b","epsilon","kappa"] #quel plot faire en fonction du temps
    def plot_time(self, data, type: possible_plots, eq=False, show=True, name=None, minimas=None):  
        t = data[:, 0]
        B = data[:, 1]
        b = data[:, 2]
        ncols = data.shape[1]
        epsilon = None
        kappa = None
        if ncols > 3:
            kappa = data[:, 3]
        if ncols > 4:
            epsilon = data[:, 4]
        fig, ax = plt.subplots(figsize=(10,5))
        fig.tight_layout()
        
        if ((type == "epsilon" or type == "kappa") and not (ncols > 3)):
            raise ValueError("Simulation" + self.term + "terme incompatible avec le type de plot voulu (" + type + ")")

        is_B=(type=="Bb" or type=="B")
        is_b=(type=="Bb" or type=="b")
        if is_B : ax.plot(t, B, color='blue',   linestyle='-', lw=1.5, label="B(t)")
        if is_b : ax.plot(t, b, color='orange',   linestyle='-', lw=1.5, label="b(t)")
        if eq :
            eq_solutions = self.get_eq()
            if eq_solutions is not None:
                for idx, (Beq,beq) in enumerate(eq_solutions) :
                    sign = '+' if len(eq_solutions) > 1 and idx == 0 else '-' if len(eq_solutions) > 1 else ''
                    linestyle='--' if idx==0 else '-.'
                    if is_B:
                        labelB = "$B_{eq}^{(" + sign + ")}$" if sign else "$B_{eq}$"
                        ax.plot(t, Beq*np.ones_like(t), color='blue', linestyle=linestyle, lw=2, label=labelB)
                    if is_b:
                        labelb = "$b_{eq}^{(" + sign + ")}$" if sign else "$b_{eq}$"
                        ax.plot(t, beq*np.ones_like(t), color='orange', linestyle=linestyle, lw=2, label=labelb)
            else : ax.plot([0], [0], color='none', label="Negative discriminant ; no equilibrium solution.")
        if ((type=="B") or type=="Bb") and (self.term=='mid') and (minimas is not None) : 
                for (centre,duree) in minimas : 
                    plt.axvline(x=centre-0.5*duree, color='red', linestyle='--', lw=1.5)
                    plt.axvline(x=centre+0.5*duree, color='red', linestyle='--', lw=1.5)
        if type=="epsilon" and self.inter_epsilon and (not self.term=='short'):
            ax.plot(t, epsilon, color='green',   linestyle='-', lw=1.5, label=r"$\varepsilon(t)$")
            ax.set_ylabel("Dynamo number")
        if type=="kappa" and (not self.term=='short'):
            ax.plot(t, kappa, color='red',   linestyle='-', lw=1.5, label=r"$\kappa(t)$")
            ax.set_ylabel("Coupling factor")

        ax.set_xlabel("t")
        ax.grid(True)
        ax.legend(fontsize=8)

        try:
            os.makedirs(self.folder, exist_ok=True)
        except Exception:
            pass
        if name is None :
            savefile = os.path.join(self.folder, f"plot_{type}.png")
        else : 
            savefile=os.path.join(self.folder, name)
        plt.savefig(savefile)
        if show : plt.show()
        plt.close(fig)

    def stat_analysis(self, data):
        t = data[:, 0]
        B = data[:, 1]
        eq=self.get_eq()
        (B_eq,b_eq)=eq[0]
        if B_eq==0 : B_eq=max(B)
        threshold=0.1*B_eq
        B_minimas=[x<threshold for x in B]

        minimas = []
        start = None

        for i, val in enumerate(B_minimas + [False]): 
            if (val) and (start is None) and (t[i]>10/max(abs(self.epsiloneq),0.01)):
                start = i
            elif (not val) and (start is not None):
                if ((t[i-1]-t[start])>200) :
                    minimas.append((t[start]+0.5*(t[i-1]-t[start]), t[i-1] - t[start]))
                    start = None
                else : start = None
        
        return minimas
    
    def plot_histograms(self,minimas_list,differentfolder=None, show=False, name=None):
        fig, ax = plt.subplots(figsize=(10,5))
        fig.tight_layout()

        for minimas in minimas_list : 
            lengths = [duree for (_, duree) in minimas]
            ax.hist(lengths, bins=max(10,round(len(lengths)/10)))
        
        ax.set_xlabel("Lenghts")
        ax.set_ylabel("Frenquency")
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=8)

        try:
            os.makedirs(self.folder, exist_ok=True)
        except Exception:
            pass
        if differentfolder is None :
            if name is None : savefile = os.path.join(self.folder, f"plot_minimas.png")
            else : savefile = os.path.join(self.folder, name)
        else : 
            if name is None : savefile = os.path.join(differentfolder, f"plot_minimas.png")
            else : savefile = os.path.join(differentfolder, name)
        plt.savefig(savefile)
        if show : plt.show()
        plt.close(fig)

    def write_stat_file(self,minimas,differentfile=None):
        text = f"{'No.':<5}{'Centre':<15}{'Durée':<15}"
        for idx, (centre, duree) in enumerate(minimas):
            text += f"\n{idx:<5}{centre:<15.6f}{duree:<15.6f}"
        target_file = self.stat_file if differentfile is None else differentfile
        target_dir = os.path.dirname(target_file)
        os.makedirs(target_dir, exist_ok=True)
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(text)

import numpy as np

def read_output(outputfile):
    data = np.genfromtxt(outputfile)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data

def read_minimas(minimasfile):
    minimas = []

    with open(minimasfile, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.split()

        if len(parts) >= 3:
            _, centre, duree = parts[:3]
            minimas.append((float(centre), float(duree)))

    return minimas

def analyze_minimas(minimas):
    [(centre,duree)]=minimas

    avg_duree=stat.mean(duree)
    stdev_duree=stat.pstdev(duree)
    
    fins=[(centre[i]+0.5*duree[i]) for i in minimas]
    ecarts=[fins[i+1]-fins[i] for i in fins].pop()
    avg_ecarts=stat.mean(ecarts)
    stdev_ecarts=stat.pstdev(ecarts)



if test:
    workdir = os.path.dirname(os.path.abspath(__file__))
    datadir=os.path.join(workdir, "data/tests")
    cfg=config(datadir=datadir, term='mid', tfin=400)
    cfg.taukappa=10000
    cfg.deltakappa=cfg.kappaeq*1e-3
    cfg.tauepsilon=10000
    cfg.deltaepsilon=max(1e-5,abs(cfg.epsiloneq*1e-3))
    cfg.nu=1
    data=cfg.run(save=True)
    minimas=cfg.stat_analysis(data)
    minimas_list=[minimas]
    cfg.plot_time(data, type='Bb', show=True,minimas=minimas)
    cfg.plot_time(data, type='epsilon', show=True)
    cfg.plot_time(data, type='kappa', show=True)
    cfg.write_stat_file(minimas)
    cfg.plot_histograms(minimas_list=minimas_list,show=True)
    
