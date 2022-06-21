import math


def find_whole_fraction(number):
    if number >= 0:
        whole = int(number)
        fraction = number - int(number)
    else:
        whole = math.floor(number)
        fraction = number - math.floor(number)
    if fraction < 0.001:
        fraction = 0.
    elif fraction > 0.999:
        whole += 1
        fraction = 0.
    return whole, fraction
