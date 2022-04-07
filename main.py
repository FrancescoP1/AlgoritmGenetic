import copy
import random
import matplotlib.pyplot as plt
from individ import Individ
global a
global b
global c


def print_population(pop_list):
    ct = 1
    for individ in pop_list:
        g.write('{0}: {1} x= {2:.6f} f= {3}\n'.format(ct, individ.show_chromosome, individ.val, individ.fit))
        ct += 1


def generate_first_population(pop_size):
    pop = []
    for i in range(pop_size):
        pop.append(Individ(Individ.generate_chromosome()))
    return pop


def get_total_fitness(pop):
    total_fit = 0
    for individ in pop:
        total_fit += individ.fit
    return total_fit


def generate_selection_probability(pop):
    # generam probabilitatile de selectie pt fiecare individ
    total_fit = get_total_fitness(pop)
    select_prob = []
    for individ in pop:
        select_prob.append(individ.fit/total_fit)
        individ.selection_probability = individ.fit/total_fit
    return select_prob


def show_selection_probabilities(pop):
    g.write('Probabilitati selectie:\n')
    ct = 1
    for individ in pop:
        g.write('cromozom   {0} probabilitate {1}\n'.format(ct, individ.selection_probability))


def generate_selection_intervals(pop):
    # generare intervale selectie
    sel_intervals = [0]
    for i in range(len(pop)):
        sel_intervals.append(sel_intervals[i] + pop[i].selection_probability)
    return sel_intervals


def show_selection_intervals(sel_intervals):
    g.write('Intervale probabilitati selectie: \n')
    for element in sel_intervals:
        g.write('{0} '.format(element))
    g.write('\n')


def search_element(el, left, right, select_prob):
    # binary search to find the corresponding interval for each chromosome
    if el <= select_prob[left]:
        return left
    if el > select_prob[right]:
        return right
    middle = int((left + right) / 2)
    if select_prob[middle] <= el < select_prob[middle + 1]:
        return middle
    if el < select_prob[middle]:
        return search_element(el, left, middle - 1, select_prob)
    if el >= select_prob[middle + 1]:
        return search_element(el, middle + 1, right, select_prob)


def select_population(pop, selection_prob, wrt):
    # etapa de selectie a populatiei, selectam indivizii ce vor participa la urmatoarea generatie
    new_pop = []
    for i in range(len(pop)):
        u = random.uniform(0, 1)
        chromosome_selected = search_element(u, 0, len(selection_prob) - 1, selection_prob)
        new_pop.append(pop[chromosome_selected])
        if wrt:
            g.write('u= {0}, selectam cromozomul {1}\n'.format(u, chromosome_selected + 1))
    return new_pop


def recombine_chromosomes(ind1, ind2, break_point):
    # recombinare a 2 cromozomi
    aux = ind1.chromosome[: break_point]
    aux2 = ind2.chromosome[: break_point]
    aux.extend(ind2.chromosome[break_point:])
    aux2.extend(ind1.chromosome[break_point:])
    ind2.chromosome = aux
    ind2.reset_individ()
    ind1.chromosome = aux2
    ind1.reset_individ()


def recombine_population(pop, rec_chance, wrt):
    # recombinam cromozomii din populatie
    # wrt seminfica daca scriem in fisier sau nu
    index_rec = []
    if wrt:
        g.write('Probabilitatea de incrucisare {0}\n'.format(rec_chance))
    for i in range(len(pop)):
        u = random.uniform(0, 1)
        if wrt:
            g.write('{0}: {1} u={2}'.format(i+1, pop[i].show_chromosome, u))
        if u < rec_chance:
            if wrt:
                g.write('<{0} participa'.format(rec_chance))
            index_rec.append(i)
        if wrt:
            g.write('\n')

    # random recombination order
    random.shuffle(index_rec)
    lg = len(index_rec)
    # recombinam 2 cate 2
    for i in range(0, lg, 2):
        if i + 1 < lg:
            individ1 = pop[index_rec[i]]
            individ2 = pop[index_rec[i + 1]]
            # generare breakpoint
            break_point = random.randint(0, Individ.number_of_bits)
            if wrt:
                g.write('Recombinare dintre cromozomul {0} cu cromozomul {1}:\n'.format(index_rec[i] + 1, index_rec[i + 1] + 1))
                g.write('{0} {1} punct {2}\n'.format(individ1.show_chromosome, individ2.show_chromosome, break_point))
            recombine_chromosomes(individ1, individ2, break_point)
            if wrt:
                g.write('Rezultat   {0}   {1}\n'.format(individ1.show_chromosome, individ2.show_chromosome))

    if wrt:
        g.write('Dupa recombinare:\n')
        print_population(pop)
    return pop


def mutate_chromosome(ind1, mutation_probability):
    # mutatie deasa a unui cromozom
    modified = False
    for i in range(Individ.number_of_bits):
        u = random.uniform(0, 1)
        if u < mutation_probability:
            modified = True
            if ind1.chromosome[i] == 0:
                ind1.chromosome[i] = 1
            else:
                ind1.chromosome[i] = 0
    return modified


def mutate_population(pop, mutation_probability, wrt):
    # aplicam mutatia asupra tuturor elementelor din pop, cu probabilitatea mutation_probability
    modified_chromosomes = []
    if wrt:
        g.write('Probabilitatea de mutatie pentru fiecare gena: {0}\n'.format(mutation_probability))
    for i in range(len(pop)):
        has_mutated = mutate_chromosome(pop[i], mutation_probability)
        if has_mutated:
            pop[i].reset_individ()
            modified_chromosomes.append(i)
    if len(modified_chromosomes) and wrt:
        g.write('Au fost modificati cromozomii:\n')
        for chromosome in modified_chromosomes:
            g.write('{0}\n'.format(chromosome + 1))
    if wrt:
        g.write('Dupa mutatie:\n')
        print_population(pop)
    return pop


def select_fittest_element(pop):
    # selectam cel mai "fit" element din pop
    if len(pop):
        max_fit = pop[0].fit
        fittest = pop[0]
        for i in range(1, len(pop)):
            if pop[i].fit > max_fit:
                max_fit = pop[i].fit
                fittest = pop[i]
        return fittest
    return -1


def select_worse_element(pop):
    # selectam cel mai "non-fit" element din pop
    if len(pop):
        worse_fit = pop[0].fit
        worst = pop[0]
        for i in range(1, len(pop)):
            if pop[i].fit < worse_fit:
                worse_fit = pop[i].fit
                worst = pop[i]
        return worst
    return -1


def swap_worse_for_fittest(pop1, pop2):
    # schimbam cel mai slab individ din pop2, cu cel mai bun din pop1
    el1 = select_fittest_element(pop1)
    # el2 = select_fittest_element(pop2)
    #if el1.fit > el2.fit:
    el3 = select_worse_element(pop2)
    pop2.remove(el3)
    pop2.append(el1)
    return pop2


def genetic_algorithm(current_population, recombination_probability, mutation_probability):
    previous_population = copy.deepcopy(current_population)
    generate_selection_probability(current_population)
    select_intervals = generate_selection_intervals(current_population)
    new_pop = select_population(current_population, select_intervals, False)
    recombined_pop = recombine_population(new_pop, recombination_probability, False)
    mutated_pop = mutate_population(recombined_pop, mutation_probability, False)
    # criteriul elitist:
    final_pop = swap_worse_for_fittest(previous_population, mutated_pop)
    return final_pop


if __name__ == '__main__':

    f = open("input.txt", "r")
    g = open("output.txt", "w")
    input_list = f.read().split()

    population_size = int(input_list[0])
    left = float(input_list[1])
    right = float(input_list[2])
    # aX^2 + bX + C
    a = float(input_list[3])
    b = float(input_list[4])
    c = float(input_list[5])
    precision = int(input_list[6])
    rec_prob = float(input_list[7])
    mutation_prob = float(input_list[8])
    generations = int(input_list[9])
    f.close()
    Individ.initialize_individ(left, right, precision, a, b, c)

    # Initial population
    first_population = generate_first_population(population_size)
    prev_population = copy.deepcopy(first_population)

    g.write('Populatia initiala:\n')
    print_population(first_population)
    selection_probabilities = generate_selection_probability(first_population)
    show_selection_probabilities(first_population)
    selection_intervals = generate_selection_intervals(first_population)
    show_selection_intervals(selection_intervals)
    new_population = select_population(first_population, selection_intervals, True)
    g.write('Dupa selectie:\n')
    print_population(new_population)

    pop_after_recomb = recombine_population(new_population, rec_prob, True)
    pop_after_mutation = mutate_population(pop_after_recomb, mutation_prob, True)
    new_population = swap_worse_for_fittest(prev_population, new_population)
    # print('first_pop: {0} vs second_pop: {1}'.format(prev_max.fit, fittest.fit))
    # print(combine_lists([0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1], 17))

    g.write('Evolutia maximului: \n')
    max_evolution = []
    m1 = select_fittest_element(prev_population)
    m2 = select_fittest_element(new_population)
    max_evolution.append(m1.fit)
    max_evolution.append(m2.fit)
    g.write('{0}\n'.format(m1.fit))
    g.write('{0}\n'.format(m2.fit))
    gen = 2
    while gen < generations:
        intermediary_population = genetic_algorithm(new_population, rec_prob, mutation_prob)
        fittest_element = select_fittest_element(new_population)
        g.write('{0}\n'.format(fittest_element.fit))
        new_population = copy.deepcopy(intermediary_population)
        #print(gen)
        max_evolution.append(fittest_element.fit)
        gen += 1

    gens = list(range(1, 51))
    print(gens)
    print(max_evolution)
    plt.plot(gens, max_evolution)
    plt.ylabel('Maximul')
    plt.xlabel('Generatia')
    plt.title('Evolutia maximului')
    plt.show()

