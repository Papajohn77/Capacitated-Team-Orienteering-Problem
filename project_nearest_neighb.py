import math
import copy
import random
from solution_drawer import *

class Warehouse:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

class Customer:
    def __init__(self, id, x, y, demand, service_time, profit):
        self.id = int(id)
        self.x = float(x)
        self.y = float(y)
        self.demand = int(demand)
        self.service_time = int(service_time)
        self.profit = int(profit)
        self.is_routed = False

class Vehicle:
    def __init__(self, warehouse, capacity, duration):
        self.route = [warehouse]
        self.capacity = capacity
        self.duration = duration

class Solution:
    def __init__(self):
        self.profit = 0
        self.routes = []

class SwapMove:
    def __init__(self):
        self.move_duration_change = None
        self.first_route_pos = None
        self.second_route_pos = None
        self.first_cust_pos = None
        self.second_cust_pos = None
        self.duration_change_first_route = None
        self.duration_change_second_route = None

    def init(self):
        self.move_duration_change = 10 ** 9
        self.first_route_pos = None
        self.second_route_pos = None
        self.first_cust_pos = None
        self.second_cust_pos = None
        self.duration_change_first_route = None
        self.duration_change_second_route = None

class RelocationMove:
    def __init__(self):
        self.move_duration_change = None
        self.origin_route_pos = None
        self.target_route_pos = None
        self.origin_cust_pos = None
        self.target_cust_pos = None
        self.duration_change_origin_route = None
        self.duration_change_target_route = None

    def init(self):
        self.move_duration_change = 10 ** 9
        self.origin_route_pos = None
        self.target_route_pos = None
        self.origin_cust_pos = None
        self.target_cust_pos = None
        self.duration_change_origin_route = None
        self.duration_change_target_route = None

def create_customers(filename):
    customers = []
    with open(filename) as customers_file:
        for line in customers_file:
            customers.append(Customer(*line[:-1].split(';')))
    return customers

def create_distance_matrix(warehouse, customers):
    nodes = [warehouse, *customers]
    distance_matrix = [[0 for _ in range(len(nodes))] for _ in range(len(nodes))]
    for node_1 in nodes:
        for node_2 in nodes:
            distance_matrix[node_1.id][node_2.id] = math.sqrt(math.pow(node_1.x - node_2.x, 2) + math.pow(node_1.y - node_2.y, 2))
    return distance_matrix

def create_vehicles(num_of_vehicles, warehouse, capacity, duration):
    vehicles = []
    for _ in range(num_of_vehicles):
        vehicles.append(Vehicle(warehouse, capacity, duration))
    return vehicles

def reset_customers(customers):
    for cust in customers:
        cust.is_routed = False

def solve(solution, warehouse, customers, distance_matrix, vehicles, rcl_size, demand_weight):
    while True:
        rcl = []

        for vehicle in vehicles:
            for cust in customers:
                if cust.is_routed:
                    continue

                if cust.demand > vehicle.capacity:
                    continue

                if cust.service_time + distance_matrix[vehicle.route[-1].id][cust.id] + distance_matrix[cust.id][warehouse.id] > vehicle.duration:
                    continue

                value = cust.profit / (cust.demand * demand_weight + (cust.service_time + distance_matrix[vehicle.route[-1].id][cust.id]))

                if len(rcl) < rcl_size:
                    rcl.append((value, cust, vehicle))
                    rcl.sort(key=lambda x: x[0], reverse=True)
                elif value > rcl[-1][0]:
                    rcl.pop(-1)
                    rcl.append((value, cust, vehicle))
                    rcl.sort(key=lambda x: x[0], reverse=True)
        
        if len(rcl) == rcl_size:
            max, max_cust, max_vehicle = rcl[random.randint(0, rcl_size - 1)]

            solution.profit += max_cust.profit
            max_vehicle.capacity -= max_cust.demand
            max_vehicle.duration -= max_cust.service_time + distance_matrix[max_vehicle.route[-1].id][max_cust.id]
            max_vehicle.route.append(max_cust)
            max_cust.is_routed = True
        else:
            break
    
    for vehicle in vehicles:
        vehicle.duration -= distance_matrix[vehicle.route[-1].id][warehouse.id]
        vehicle.route.append(warehouse)
        solution.routes.append(vehicle.route)

def find_best_swap_move(distance_matrix, vehicles, swap_move):
    for first_route_idx in range(len(vehicles)):
        vehicle_1 = vehicles[first_route_idx]
        for second_route_idx in range(first_route_idx, len(vehicles)):
            vehicle_2 = vehicles[second_route_idx]
            for first_cust_idx in range(1, len(vehicle_1.route) - 1):
                start_of_second_cust_idx = 1
                if first_route_idx == second_route_idx:
                    start_of_second_cust_idx = first_cust_idx + 1

                for second_cust_idx in range(start_of_second_cust_idx, len(vehicle_2.route) - 1):

                    A = vehicle_1.route[first_cust_idx - 1]
                    B = vehicle_1.route[first_cust_idx]
                    C = vehicle_1.route[first_cust_idx + 1]

                    D = vehicle_2.route[second_cust_idx - 1]
                    E = vehicle_2.route[second_cust_idx]
                    F = vehicle_2.route[second_cust_idx + 1]

                    move_duration_change = None
                    duration_change_first_route = None
                    duration_change_second_route = None

                    if first_route_idx == second_route_idx:
                        if first_cust_idx == second_cust_idx - 1:
                            removed_duration = distance_matrix[A.id][B.id] + distance_matrix[B.id][E.id] + distance_matrix[E.id][F.id]
                            added_duration = distance_matrix[A.id][E.id] + distance_matrix[E.id][B.id] + distance_matrix[B.id][F.id]
                            move_duration_change = added_duration - removed_duration
                        else:
                            removed_duration_1 = distance_matrix[A.id][B.id] + distance_matrix[B.id][C.id]
                            added_duration_1 = distance_matrix[A.id][E.id] + distance_matrix[E.id][C.id]
                            removed_duration_2 = distance_matrix[D.id][E.id] + distance_matrix[E.id][F.id]
                            added_duration_2 = distance_matrix[D.id][B.id] + distance_matrix[B.id][F.id]
                            move_duration_change = added_duration_1 + added_duration_2 - removed_duration_1 - removed_duration_2
                    else:
                        if vehicle_1.capacity < B.demand - E.demand:
                            continue

                        if vehicle_2.capacity < E.demand - B.demand:
                            continue

                        removed_duration_1 = distance_matrix[A.id][B.id] + distance_matrix[B.id][C.id]
                        added_duration_1 = distance_matrix[A.id][E.id] + distance_matrix[E.id][C.id]
                        removed_duration_2 = distance_matrix[D.id][E.id] + distance_matrix[E.id][F.id]
                        added_duration_2 = distance_matrix[D.id][B.id] + distance_matrix[B.id][F.id]

                        duration_change_first_route = added_duration_1 - removed_duration_1 - B.service_time + E.service_time
                        duration_change_second_route = added_duration_2 - removed_duration_2 - E.service_time + B.service_time

                        if vehicle_1.duration < duration_change_first_route:
                            continue

                        if vehicle_2.duration < duration_change_second_route:
                            continue

                        move_duration_change = added_duration_1 + added_duration_2 - removed_duration_1 - removed_duration_2

                    if move_duration_change < swap_move.move_duration_change:
                        store_best_swap_move(first_route_idx, second_route_idx, first_cust_idx, second_cust_idx, move_duration_change, duration_change_first_route,duration_change_second_route, swap_move)

def store_best_swap_move(first_route_idx, second_route_idx, first_cust_idx, second_cust_idx, move_duration_change, duration_change_first_route, duration_change_second_route, swap_move):
    swap_move.move_duration_change = move_duration_change
    swap_move.first_route_pos = first_route_idx
    swap_move.second_route_pos = second_route_idx
    swap_move.first_cust_pos = first_cust_idx
    swap_move.second_cust_pos = second_cust_idx
    swap_move.duration_change_first_route = duration_change_first_route
    swap_move.duration_change_second_route = duration_change_second_route

def apply_swap_move(vehicles, swap_move):
    vehicle_1 = vehicles[swap_move.first_route_pos]
    vehicle_2 = vehicles[swap_move.second_route_pos]
    route_1 = vehicle_1.route
    route_2 = vehicle_2.route
    B = route_1[swap_move.first_cust_pos]
    E = route_2[swap_move.second_cust_pos]
    route_1[swap_move.first_cust_pos] = E
    route_2[swap_move.second_cust_pos] = B

    if (route_1 == route_2):
        vehicle_1.duration -= swap_move.move_duration_change
    else:
        vehicle_1.duration -= swap_move.duration_change_first_route
        vehicle_2.duration -= swap_move.duration_change_second_route
        vehicle_1.capacity += B.demand - E.demand
        vehicle_2.capacity += E.demand - B.demand

def find_best_relocation_move(distance_matrix, vehicles, relocation_move):
    for origin_route_idx in range(len(vehicles)):
        vehicle_1 = vehicles[origin_route_idx]
        for target_route_idx in range(len(vehicles)):
            vehicle_2 = vehicles[target_route_idx]
            for origin_cust_idx in range(1, len(vehicle_1.route) - 1):
                for target_cust_idx in range(len(vehicle_2.route) - 1):

                    if target_route_idx == origin_route_idx and (target_cust_idx == origin_cust_idx or target_cust_idx == origin_cust_idx - 1):
                        continue

                    A = vehicle_1.route[origin_cust_idx - 1]
                    B = vehicle_1.route[origin_cust_idx]
                    C = vehicle_1.route[origin_cust_idx + 1]

                    F = vehicle_2.route[target_cust_idx]
                    G = vehicle_2.route[target_cust_idx + 1]

                    if origin_route_idx != target_route_idx:
                        if vehicle_2.capacity < B.demand:
                            continue

                    added_duration = distance_matrix[A.id][C.id] + distance_matrix[F.id][B.id] + distance_matrix[B.id][G.id]
                    removed_duration = distance_matrix[A.id][B.id] + distance_matrix[B.id][C.id] + distance_matrix[F.id][G.id]

                    origin_route_duration_change = distance_matrix[A.id][C.id] - distance_matrix[A.id][B.id] - distance_matrix[B.id][C.id] - B.service_time
                    target_route_duration_change = distance_matrix[F.id][B.id] + distance_matrix[B.id][G.id] - distance_matrix[F.id][G.id] + B.service_time

                    if vehicle_1.duration < origin_route_duration_change:
                        continue

                    if vehicle_2.duration < target_route_duration_change:
                        continue

                    move_duration_change = added_duration - removed_duration

                    if (move_duration_change < relocation_move.move_duration_change):
                        store_best_relocation_move(origin_route_idx, target_route_idx, origin_cust_idx, target_cust_idx, move_duration_change, origin_route_duration_change, target_route_duration_change, relocation_move)

def store_best_relocation_move(origin_route_idx, target_route_idx, origin_cust_idx, target_cust_idx, move_duration_change, origin_route_duration_change, target_route_duration_change, relocation_move):
    relocation_move.move_duration_change = move_duration_change
    relocation_move.origin_route_pos = origin_route_idx
    relocation_move.origin_cust_pos = origin_cust_idx
    relocation_move.target_route_pos = target_route_idx
    relocation_move.target_cust_pos = target_cust_idx
    relocation_move.duration_change_origin_route = origin_route_duration_change
    relocation_move.duration_change_target_route = target_route_duration_change

def apply_relocation_move(vehicles, relocation_move):
    vehicle_1 = vehicles[relocation_move.origin_route_pos]
    vehicle_2 = vehicles[relocation_move.target_route_pos]
    origin_route = vehicle_1.route
    target_route = vehicle_2.route

    B = origin_route[relocation_move.origin_cust_pos]

    if origin_route == target_route:
        del origin_route[relocation_move.origin_cust_pos]

        if (relocation_move.origin_cust_pos < relocation_move.target_cust_pos):
            target_route.insert(relocation_move.target_cust_pos, B)
        else:
            target_route.insert(relocation_move.target_cust_pos + 1, B)

        vehicle_1.duration -= relocation_move.move_duration_change
    else:
        del origin_route[relocation_move.origin_cust_pos]
        target_route.insert(relocation_move.target_cust_pos + 1, B)

        vehicle_1.duration -= relocation_move.duration_change_origin_route
        vehicle_2.duration -= relocation_move.duration_change_target_route
        vehicle_1.capacity += B.demand
        vehicle_2.capacity -= B.demand

def del_n_per_route(solution, distance_matrix, vehicles, n):
    for vehicle in vehicles:
        for _ in range(n):
            min = float('inf')
            min_pos = -1

            for pos in range(1, len(vehicle.route) - 1):
                prev_cust = vehicle.route[pos - 1]
                cust = vehicle.route[pos]
                next_cust = vehicle.route[pos + 1]

                value = cust.profit / (cust.demand + (cust.service_time + distance_matrix[prev_cust.id][cust.id] + distance_matrix[cust.id][next_cust.id]))

                if value < min:
                    min = value
                    min_pos = pos

            if min != float('inf'):
                prev_cust = vehicle.route[min_pos - 1]
                cust = vehicle.route[min_pos]
                next_cust = vehicle.route[min_pos + 1]

                solution.profit -= cust.profit
                vehicle.capacity += cust.demand
                vehicle.duration += cust.service_time + distance_matrix[prev_cust.id][cust.id] + distance_matrix[cust.id][next_cust.id] - distance_matrix[prev_cust.id][next_cust.id]

                vehicle.route.remove(cust)
                cust.is_routed = False

def fill_with_unserved(solution, customers, distance_matrix, vehicles):
    while True:
        max = float('-inf')
        max_pos = -1
        max_cust = -1
        max_vehicle = -1
        max_additional_duration = -1

        for unserved_cust in customers:
            if unserved_cust.is_routed:
                continue

            for vehicle in vehicles:
                for pos in range(1, len(vehicle.route)):
                    prev = vehicle.route[pos - 1]
                    next = vehicle.route[pos]
                    additional_duration = distance_matrix[prev.id][unserved_cust.id] + distance_matrix[unserved_cust.id][next.id] - distance_matrix[prev.id][next.id]

                    if unserved_cust.demand > vehicle.capacity:
                        continue

                    if unserved_cust.service_time + additional_duration > vehicle.duration:
                        continue

                    value = unserved_cust.profit / (unserved_cust.demand + (unserved_cust.service_time + additional_duration))

                    if value > max:
                        max = value
                        max_pos = pos
                        max_cust = unserved_cust
                        max_vehicle = vehicle
                        max_additional_duration = additional_duration
        
        if max != float('-inf'):
            solution.profit += max_cust.profit
            max_vehicle.capacity -= max_cust.demand
            max_vehicle.duration -= max_cust.service_time + max_additional_duration
            max_vehicle.route.insert(max_pos, max_cust)
            max_cust.is_routed = True
        else:
            break

def swap_served_with_unserved(solution, customers, distance_matrix, vehicles):
    while True:
        max = float('-inf')
        max_pos = -1
        max_cust = -1
        max_vehicle = -1
        max_additional_demand = -1
        max_additional_duration = -1

        for unserved_cust in customers:
            if unserved_cust.is_routed:
                continue

            for vehicle in vehicles:
                for pos in range(1, len(vehicle.route) - 1):
                    prev_cust = vehicle.route[pos - 1]
                    cust = vehicle.route[pos]
                    next_cust = vehicle.route[pos + 1]

                    additional_demand = unserved_cust.demand - cust.demand

                    if additional_demand > vehicle.capacity:
                        continue

                    served_duration = cust.service_time + distance_matrix[prev_cust.id][cust.id] + distance_matrix[cust.id][next_cust.id]
                    unserved_duration = unserved_cust.service_time + distance_matrix[prev_cust.id][unserved_cust.id] + distance_matrix[unserved_cust.id][next_cust.id]
                    additional_duration = unserved_duration - served_duration

                    if additional_duration > vehicle.duration:
                        continue

                    value_served = cust.profit / (cust.demand + served_duration)

                    value_unserved = unserved_cust.profit / (unserved_cust.demand + unserved_duration)

                    value_percentage_change = ((value_unserved - value_served) / value_served)

                    if value_percentage_change > 0 and value_percentage_change > max:
                        max = value_percentage_change
                        max_pos = pos
                        max_cust = unserved_cust
                        max_vehicle = vehicle
                        max_additional_demand = additional_demand
                        max_additional_duration = additional_duration

        if max != float('-inf'):
            cust_to_be_changed = max_vehicle.route[max_pos]
            solution.profit += max_cust.profit - cust_to_be_changed.profit
            max_vehicle.capacity -= max_additional_demand
            max_vehicle.duration -= max_additional_duration
            max_vehicle.route[max_pos] = max_cust
            cust_to_be_changed.is_routed = False
            max_cust.is_routed = True
        else:
            break


if __name__ == '__main__':
    random.seed(10)
    warehouse = Warehouse(0, 23.142, 11.736)
    customers = create_customers('customers.csv')
    distance_matrix = create_distance_matrix(warehouse, customers)
    vehicles = create_vehicles(6, warehouse, 150, 200)

    max_profit = float('-inf')
    max_solution = []
    max_vehicles = []

    rcl_size = 4
    demand_weight = 2.5
    for _ in range(525):
        solution = Solution()
        reset_customers(customers)
        vehicles_copy = copy.deepcopy(vehicles)
        solve(solution, warehouse, customers, distance_matrix, vehicles_copy, rcl_size, demand_weight)

        swap_move = SwapMove()
        while True:
            swap_move.init()

            find_best_swap_move(distance_matrix, vehicles_copy, swap_move)
            if swap_move.move_duration_change < -0.01:
                apply_swap_move(vehicles_copy, swap_move)
            else:
                break

        relocation_move = RelocationMove()
        while True:
            relocation_move.init()

            find_best_relocation_move(distance_matrix, vehicles_copy, relocation_move)
            if relocation_move.move_duration_change < -0.01:
                apply_relocation_move(vehicles_copy, relocation_move)
            else:
                break

        n = 2
        del_n_per_route(solution, distance_matrix, vehicles_copy, n)
        fill_with_unserved(solution, customers, distance_matrix, vehicles_copy)
        swap_served_with_unserved(solution, customers, distance_matrix, vehicles_copy)

        if solution.profit > max_profit:
            max_profit = solution.profit
            max_solution = solution
            max_vehicles = vehicles_copy

    file = open("solution.txt", "w+")

    file.write(f'Total Profit\n{max_profit}\n')
    for i, route in enumerate(max_solution.routes):
        file.write(f'Route {i + 1}\n')
        route_customers = ''
        for cust in route:
            route_customers += f'{cust.id} '
        file.write(f'{route_customers[:-1]}\n')

    file.close()

    SolutionDrawer.draw(max_solution, customers)
