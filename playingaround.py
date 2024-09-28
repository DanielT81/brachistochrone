import multiprocessing as mp
import math
import sys

print('x'+'1')
'''
sys.set_int_max_str_digits(1000000)

# Funktion für aufwändige Berechnungen
def calc_factorial(number):
    return math.factorial(number)

if __name__ == '__main__':
    # Erstelle eine Liste von Aufgaben
    numbers = [100000, 100000, 100000, 100000, 98273, 29837]

    # Erstelle einen Pool von Prozessen
    pool = mp.Pool(processes=16)  # 4 Prozesse, entsprechend der Anzahl der CPU-Kerne

    # Führe die Aufgaben parallel aus
    results = pool.map(calc_factorial, numbers)

    print(results)
'''