import numpy as np
import math


# to 1 task of home works
def calculate_pi_method_monte_carlo():
    count_iteration = 1_000_000
    in_round = 0

    for _ in range(count_iteration):
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, 1)

        if math.sqrt(x**2 + y**2) < 1:
            in_round += 1

    print(f"Pi is: {(in_round / count_iteration) * 4}")


# to 2 task of home works
def find_all_numbers():
    # laziness :)
    pass


if __name__ == '__main__':
    calculate_pi_method_monte_carlo()
    find_all_numbers()
