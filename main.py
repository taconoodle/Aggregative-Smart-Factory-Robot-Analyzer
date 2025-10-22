import csv
import math

######################## SMOKE CODE ##############################

def avg_speed(min_speed, max_speed):
    min_speed = float(min_speed)
    max_speed = float(max_speed)
    if min_speed > max_speed:
        print("The min speed cannot be greater than the max speed")
        return 0

    robot_avg = get_avg_speeds()

    filtered_avg = {} # dictionary to filter the average speeds to the desired range

    for robot_id, (avg, count) in robot_avg.items():
        if min_speed < avg <= max_speed:  # Checking each robot id avg speed if it is in the desired range
            filtered_avg[robot_id] = (avg, count)

    return filtered_avg

def idle_ratio(min_percentage, max_percentage):
    min_percentage = float(min_percentage)
    max_percentage = float(max_percentage)
    if min_percentage > max_percentage:
        print("The min percentage cannot be greater than the max percentage.")
        return {}

    robot_data = {}

    # Read and group by robot
    with open("data/smart_factory_robots.csv") as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        for row in reader:
            robot_id = int(row[0])
            current_time = float(row[1])
            idle_state = row[9].strip()

            if robot_id not in robot_data:
                robot_data[robot_id] = []
            robot_data[robot_id].append((current_time, idle_state))

    # Sort by time
    for robot_id in robot_data:
        robot_data[robot_id].sort(key=lambda x: x[0])

    # Compute idle ratios
    robot_ratios = {}
    for robot_id, rows in robot_data.items():
        if len(rows) < 2:
            continue

        idle_time = 0.0
        total_time = 0.0

        for i in range(1, len(rows)):
            prev_time, prev_idle_state = rows[i - 1]
            curr_time, _ = rows[i]
            dt = curr_time - prev_time
            total_time += dt
            if prev_idle_state == "True":
                idle_time += dt
            ratio = idle_time / total_time
            robot_ratios[robot_id] = (ratio, len(rows))

    filtered_robots = {} # dictionary to filter to the desired range

    for robot_id, (ratio, count) in robot_ratios.items():
        if min_percentage <= ratio <= max_percentage:  # Checking each robot id avg speed if it is in the desired range
            filtered_robots[robot_id] = (ratio, count)

    return filtered_robots



########################## TACO CODE #############################

DATAPATH = "data/smart_factory_robots_200.csv"
#"C:/Users/Ιωάννης Βλάσσης/Downloads/smart_factory_robots.csv"

def get_avg_speeds():
    #Dictionary of the form: robot_info[robot ID] = [cumulative speed, count of measurements]
    robot_info = {}

    with open(DATAPATH) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            #Get the data from the datafile
            robot_id = int(row[0])
            vx = float(row[6])
            vy = float(row[7])

            #Given formula
            speed = math.sqrt(vx ** 2 + vy ** 2)

            #If we don't initialize, the addition below, will fail
            if robot_id not in robot_info:
                robot_info[robot_id] = [0, 0]

            robot_info[robot_id][0] += speed
            robot_info[robot_id][1] += 1

    #Dictionary of the form: robot_avgs[robot ID] = [avg speed, count of measurements]
    robot_avgs = {}
    for robot_id, (sum_speed, count) in robot_info.items():
        avg = sum_speed / count
        robot_avgs[robot_id] = [avg, count]

    return robot_avgs

def top_speed(robot_count: int):
    sorted_avg_speeds = dict(sorted(get_avg_speeds().items(), key=lambda x: x[1], reverse=True))

    print(f'{'Rank'}  {'robot_id'}  {'Μέση Ταχύτητα (m/s)'}  {'Αριθμός Εγγραφών'}')

    i = 0
    #NA DW GIATI THELEI PARENTHESH
    for robot_id, (avg, count) in sorted_avg_speeds.items():
        if i == robot_count:
            break
        i += 1

        print(f'{i:<4}  {robot_id:<8}  {avg:<19.4f}  {count:<15}')

    return sorted_avg_speeds

def collisions(start_time, end_time):
    #Dictionary of the form: robots_collided[robot ID] = [collision count, time of first collision, time of last collision]
    robots_collided = {}

    with open(DATAPATH) as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            robot_id = int(row[0])
            current_time = float(row[1])
            goal_status = row[8]

            if (start_time <= current_time < end_time) and (goal_status == "collision detected"):
                #If it's the robot's first entry we have to initialize the values
                if robot_id not in robots_collided: robots_collided[robot_id] = [0, current_time, 0]

                robots_collided[robot_id][0] += 1
                robots_collided[robot_id][2] = current_time

        print(f'Robot ID  Collision Count  Timeframe')
        for robot_id, (count, first_collision_time, last_collision_time) in robots_collided.items():
            print(f'{robot_id:<8}  {count:<15}  {f'{first_collision_time:.1f}-{last_collision_time:.1f}':<9}')

    return robots_collided




######################## GENERAL CODE ############################

def menu():
    while True:
        choice = input(f'Please Choose your option:')
        match choice:
            case '1':
                min_speed = input(f'Please choose your minimum speed:')
                max_speed = input(f'Please choose your maximum speed:')
                avg_speeds = avg_speed(min_speed, max_speed)
                if not avg_speeds:
                    print("No robots found in the specified speed range.")
                else:
                    print(f"{'robot_id':<10} {'Μέση Ταχύτητα (m/s)':<25} {'Αριθμός Εγγραφών'}")
                    for robot_id, (avg, count) in avg_speeds.items():
                        print(f"{robot_id:<8} {avg:<19.4f} {count:<15}")
            case '2':
                robot_count = int(input(f'Please choose how many robots:'))
                top_speed(robot_count)
            case '3':
                min_percentage = input(f'Please choose the minimum percentage of time:')
                max_percentage = input(f'Please choose the maximum percentage of time:')
                filtered = idle_ratio(min_percentage, max_percentage)
                if not filtered:
                    print("No robots found in the specified idle ratio range.")
                else:
                    print(f"{'robotID':<10} {'Ποσοστό Αδράνειας':<20} {'Αριθμός Εγγραφών'}")
                    for robot_id, (ratio, count) in filtered.items():
                        print(f"{robot_id:<8} {ratio:<19.4f} {count:<15}")
            case '4':
                start_time = float(input(f'Please choose the start time:'))
                end_time = float(input(f'Please choose the end time:'))
                collisions(start_time, end_time)
            case '5':
                deadlock_steps = input(f'Please choose the minimum amount of steps the robot was in a deadlock for:')
                # deadlocks(deadlock_steps)
            case '6':
                return
                # dominance()
            case '7':
                active_steps = input(f'Please choose the minimum number of the robot\'s active steps:')
                average_displacement_per_step = input(f'Please choose the robot\'s average displacement per step:')
                # iceberg(active_steps, average_displacement_per_step)
            case '8':
                similarity = input(f'Please enter the cosine similarity the robots should have')
                # similar_robots(similarity)
            case '9':
                critical_distance = input(f'Please enter the maximum distance two robots can reach without danger of crashing:')
                # proximity_events()
            case '10':
                robot_A = input(f'Please choose the first robot:')
                robot_B = input(f'Please choose the second robot:')
                time_lags = input(f'Please choose the lag time:')
                # lagged_corr()


menu()