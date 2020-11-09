import copy
import numpy as np
import matplotlib.pyplot as plt

class Individuo:
    def __init__(self, solucion, velocidad):
        self._solucion = solucion
        self._velocidad = velocidad
        self._b = copy.deepcopy(solucion)
        self._b_fitness = np.inf

class PSO:
    def __init__(self,
    cantidad_individuos,
    dimensiones,
    ro, #Tamaño de vecindad
    phi1_max,
    phi2_max,
    v_max,
    problema,
    generaciones):
        self._cantidad_individuos = cantidad_individuos
        self._dimensiones = dimensiones
        self._ro = ro
        self._phi1_max = phi1_max
        self._phi2_max = phi2_max
        self._v_max = v_max
        self._problema = problema
        self._generaciones = generaciones
        self._individuos = []
        self._rango = self._problema.MAX_VALUE - self._problema.MIN_VALUE
        self._mejor = np.inf

    def crearIndividuos(self):
        for i in range(self._cantidad_individuos):
            solucion = np.random.random(size = self._dimensiones) * self._rango + self._problema.MIN_VALUE
            velocidad = np.random.random(size = self._dimensiones) * self._v_max * 2 + self._v_max
            individuo = Individuo(solucion, velocidad)
            individuo._b_fitness = self._problema.fitness(individuo._solucion)
            self._individuos.append(individuo)

    def mejorIndividuo(self):
        for i in self._individuos:
            fitness = self._problema.fitness(i._solucion)
            if fitness < self._mejor:
                self._mejor = fitness

    def run(self):
        self.crearIndividuos()
        self.mejorIndividuo()
        generacion = 0
        bests = []
        gens = []
        while (generacion <= self._generaciones):
            for idx in range(len(self._individuos)):
                h = 0
                for i in range(-self._ro // 2, self._ro // 2 + 1):
                    if i == 0:
                        continue
                    elif i == -self._ro // 2:
                        h = copy.deepcopy(self._individuos[(idx + i) % len(self._individuos)])
                    elif self._problema.fitness(self._individuos[(idx + i) % len(self._individuos)]._solucion) < self._problema.fitness(h._solucion):
                        h = copy.deepcopy(self._individuos[(idx + i) % len(self._individuos)])
                phi1 = np.random.random(size = self._dimensiones) * self._phi1_max
                phi2 = np.random.random(size = self._dimensiones) * self._phi2_max
                self._individuos[idx]._velocidad = (self._individuos[idx]._velocidad +
                np.multiply(phi1, self._individuos[idx]._b - self._individuos[idx]._solucion) +
                np.multiply(phi2, h._solucion - self._individuos[idx]._solucion))
                for i in range(self._dimensiones):
                    if abs(self._individuos[idx]._velocidad[i]) > self._v_max:
                        self._individuos[idx]._velocidad[i] = self._v_max / (self._individuos[idx]._velocidad[i])
                self._individuos[idx]._solucion = self._individuos[idx]._solucion + self._individuos[idx]._velocidad
                fitness_individuo = self._problema.fitness(self._individuos[idx]._solucion)
                if fitness_individuo < self._individuos[idx]._b_fitness:
                    self._individuos[idx]._b = copy.deepcopy(self._individuos[idx]._solucion)
                    self._individuos[idx]._b_fitness = fitness_individuo
                    if fitness_individuo < self._mejor:
                        self._mejor = fitness_individuo
            
            if generacion % 100 == 0:
                bests.append(self._mejor_historico._fitness)
                gens.append(generacion)
            generacion += 1
        return gens, bests

class Sphere:
    MIN_VALUE = -5.12
    MAX_VALUE = 5.12
    def __init__(self):
        pass
    def fitness(self, vector):
        z = 0
        for dimension in vector:
            z += dimension**2
        return z

class Rosenbrock:
    MIN_VALUE = -2.048
    MAX_VALUE = 2.048
    def __init__(self):
        pass
    def fitness(self, vector):
        z = 0
        for dimension in range(len(vector)-1):
            z += 100 * (vector[dimension + 1] - vector[dimension]**2)**2 
            + (vector[dimension]-1)**2
        return z

class Rastrigin:
    MIN_VALUE = -5.12
    MAX_VALUE = 5.12
    def __init__(self):
        pass
    def fitness(self, vector):
        z = 0
        for dimension in vector:
            z += (vector[dimension]**2 - 10 * np.cos(2 * np.pi * vector[dimension])) + \
                (vector[dimension + 1]**2 - 10 * np.cos(2 * np.pi * vector[dimension + 1])) + 20
        return z

class Quartic:
    MIN_VALUE = -1.28
    MAX_VALUE = 1.28
    def __init__(self):
        pass
    def fitness(self, vector):
        z = 0
        for dimension in vector:
            z += dimension**4
        return z

def avgs(arr):
    avgs = [0]*len(arr[0])
    for arrs in arr:
        i = 0
        while i < len(avgs):
            avgs[i] += arrs[i]
            i += 1
    j = 0
    while j < len(avgs):
        avgs[j] /= len(arr)
        j += 1
    return avgs  


def runPSO(figura, nombre):
    runs = 5
    fig, axs = plt.subplots(2,2)
    fig.suptitle('Particle Swarm Optimization : (2, 4 y 8 dim) - '+nombre)
    bests = []
    gens = []
    avg = []
    i=0

    fig = figura
    rango = fig.MAX_VALUE - fig.MIN_VALUE
    cantidad_individuos = 30
    ro = 8
    phi1_max = 1.7
    phi2_max = 2.0
    v_max = rango * 0.01
    generaciones = 2000
    dimensiones = 2
    #PSO - 2 dim
    pso = PSO(cantidad_individuos, dimensiones, ro, phi1_max, phi2_max,
    v_max, fig, generaciones)
    
    while i < runs:
        print("----------------Run: "+str(i+1)+"--------------------")
        aux = []
        gens, aux = pso.run()
        bests.append(aux)
        i += 1
    
    avg = avgs(bests)
    axs[0,0].plot(gens, avg, 'r')
    axs[1,1].plot(gens, avg, 'r')
    axs[0,0].set_title('2 Dimensiones')
    #------------------------------------------------------------------
    bests = []
    gens = []
    avg = []
    i=0

    dimensiones = 4
    #PSO - 4 dim
    pso = PSO(cantidad_individuos, dimensiones, ro, phi1_max, phi2_max,
    v_max, fig, generaciones)

    while i < runs:
        print("----------------Run: "+str(i+1)+"--------------------")
        aux = []
        gens, aux = pso.run()
        bests.append(aux)
        i += 1
    
    avg = avgs(bests)
    axs[0,1].plot(gens, avg, 'g')
    axs[1,1].plot(gens, avg, 'g')
    axs[0,1].set_title('4 Dimensiones')
    #------------------------------------------------------------------
    bests = []
    gens = []
    avg = []
    i=0
    
    dimensiones = 8
    #PSO - 8 dim
    pso = PSO(cantidad_individuos, dimensiones, ro, phi1_max, phi2_max,
    v_max, fig, generaciones)

    while i < runs:
        print("----------------Run: "+str(i+1)+"--------------------")
        aux = []
        gens, aux = pso.run()
        bests.append(aux)
        i += 1
    
    avg = avgs(bests)
    axs[1,0].plot(gens, avg, 'b')
    axs[1,1].plot(gens, avg, 'b')
    axs[1,0].set_title('8 Dimensiones')
    axs[1,1].set_title('Comparativa')
    for ax in fig.get_axes():
        ax.label_outer()
    plt.show()

def main():
    sp = Sphere()
    ro = Rosenbrock()
    ra = Rastrigin()
    qu = Quartic()
    runPSO(sp, 'Sphere')
    runPSO(ro, 'Rosenbrock')
    runPSO(ra, 'Rastrigin')
    runPSO(qu, 'Quartic')

if  __name__ == '__main__':
    main()
