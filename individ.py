import math
import random


class Individ:
    left = 0
    right = 0
    precision = 0
    number_of_bits = 0

    # coeficientii polinomului
    a = 0
    b = 0
    c = 0

    def __init__(self, chromosome):
        self.val = Individ.decode(chromosome)
        self.chromosome = chromosome
        self.fit = self.fitness()
        self.selection_probability = 0

    @staticmethod
    def initialize_individ(l, r, p, a, b, c):
        Individ.left = l
        Individ.right = r
        Individ.precision = p
        Individ.number_of_bits = math.ceil(math.log(((r - l) * (10 ** p)), 2))
        Individ.a = a
        Individ.b = b
        Individ.c = c

    @staticmethod
    def decode(list_of_bits):
        pwr = 0
        value = 0
        for i in range(len(list_of_bits) - 1, -1, -1):
            value += list_of_bits[i] * (2 ** pwr)
            pwr += 1
        value = (Individ.right - Individ.left) / ((2 ** Individ.number_of_bits) - 1) * value + Individ.left
        return value

    @staticmethod
    def generate_chromosome():
        crm = []
        for i in range(Individ.number_of_bits):
            crm.append(random.randint(0, 1))
        return crm

    def fitness(self):
        return Individ.a * (self.val ** 2) + Individ.b * self.val + Individ.c

    @property
    def show_chromosome(self):
        return ''.join(map(str, self.chromosome))

    def reset_individ(self):
        self.val = Individ.decode(self.chromosome)
        self.fit = self.fitness()
