import csv
import logging
import logging.config


######################## SMOKE CODE ##############################

def avg_speed(min_speed, max_speed):
    if min_speed > max_speed:
        print("The min speed cannot be greater than the max speed")
        return 0

    robot_avg = get_avg_speeds()

    filtered_avg = {} # dict filtered_avg[robot_id] = (avg_speed, count)
    for robot_id, (avg, count) in robot_avg.items():
        if min_speed < avg <= max_speed:  # Checking each robot  avg speed if in range
            filtered_avg[robot_id] = (avg, count)

    return filtered_avg


def idle_ratio(min_percentage, max_percentage):
    if min_percentage > max_percentage:
        print("The min percentage cannot be greater than the max percentage.")
        return {}

    robot_data = {} # dict robot_data[robot_id] = (current_time, idle_state)

    # Read and group by robot
    with open(DATAPATH) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            robot_id = int(row[0])
            current_time = float(row[1])
            idle_state = row[9]

            if robot_id not in robot_data:
                robot_data[robot_id] = []
            robot_data[robot_id].append((current_time, idle_state))

    # Sort by time
    for robot_id in robot_data:
        robot_data[robot_id].sort(key=lambda x: x[0])

    # Compute idle ratios
    robot_ratios = {} # dict robot_ratios[robot_id] = (ratio, count)
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

    filtered_robots = {} # dict to filter to the range

    for robot_id, (ratio, count) in robot_ratios.items():
        if min_percentage <= ratio <= max_percentage:  # Checking each robot avg speed if in range
            filtered_robots[robot_id] = (ratio, count)

    return filtered_robots


def read_deadlock_data():
    robot_data = {} # dict robot_data[robot_id] = (current_time, deadlock_state)

    # Read and group info by robot id
    with open(DATAPATH) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            robot_id = int(row[0])
            current_time = float(row[1])
            deadlock_state = row[12]

            if robot_id not in robot_data:
                robot_data[robot_id] = []

            robot_data[robot_id].append((current_time, deadlock_state))

        # Sort by time
        for robot_id in robot_data:
            robot_data[robot_id].sort(key=lambda x: x[0])

        return robot_data


def deadlocks(deadlock_steps):
    w = deadlock_steps

    robot_data = read_deadlock_data()

    deadlock_results = {} # dict deadlock_results[robot_id] = (start_time, end_time, counter)

    for robot_id, rows in robot_data.items():
        in_deadlock = False
        start_time = None
        counter = 0

        for i, (time, deadlock) in enumerate(rows):

            if deadlock == "True":
                # if deadlock occurs 1st time in a row starts the streak count
                if not in_deadlock:
                    in_deadlock = True
                    start_time = time
                    counter = 1
                else: # if not deadlock happens doesn't 1st time in a row add to the streak count
                    counter += 1
            else:
                # if deadlock doesn't occur but already on the desired range streak keep the info
                if in_deadlock and counter >= w:
                    end_time = rows[i - 1][0]
                    if robot_id not in deadlock_results:
                        deadlock_results[robot_id] = []
                    deadlock_results[robot_id].append((start_time, end_time, counter))
                in_deadlock = False
                counter = 0

        # If deadlock continues until the end keep the info
        if in_deadlock and counter >= w:
            end_time = rows[-1][0]
            if robot_id not in deadlock_results:
                deadlock_results[robot_id] = []
            deadlock_results[robot_id].append((start_time, end_time, counter))

    return deadlock_results


def read_robot_positions():

    robot_data = {} # dict robot_data[robot_id] =(current_time, px, py, pz, idle_state)

    # Read and group info by robot id.
    with open(DATAPATH) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            robot_id = int(row[0])
            current_time = float(row[1])
            px = float(row[3])
            py = float(row[4])
            pz = float(row[5])
            idle_state = row[9]

            if robot_id not in robot_data:
                robot_data[robot_id] = []
            robot_data[robot_id].append((current_time, px, py,pz, idle_state))

    # Sort by time for each robot
    for robot_id in robot_data:
        robot_data[robot_id].sort(key=lambda x: x[0])

    return robot_data


def iceberg(active_steps, average_displacement_per_step):

    if active_steps <= 0:
        print("Invalid active steps")
        return 0

    robot_data = read_robot_positions()
    iceberg_results = {} # dict iceberg_results[robot_id] = (avg_disp, streak)

    for robot_id, rows in robot_data.items():
        total_dis = 0
        streak = 0

        for i in range(1, len(rows)):
            # get two consecutive steps
            prev_time, prev_x, prev_y, prev_z, prev_idle = rows[i - 1]
            curr_time, curr_x, curr_y, curr_z, curr_idle = rows[i]
            # check if both are active and compute displacement
            if prev_idle == "False" and curr_idle == "False":
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                dz = curr_z - prev_z
                displacement = (dx ** 2 + dy ** 2 + dz ** 2)**0.5
                total_dis += displacement
                streak += 1

        if streak >= active_steps:
            avg_disp = total_dis / streak
            if avg_disp > average_displacement_per_step:
                iceberg_results[robot_id] = (avg_disp, streak)

    return iceberg_results


def read_robot_positions_by_time():
    # dict positions_by_time[current_time][robot_id] = [robot,px,py,pz]
    positions_by_time = {}

    # Read and group info by robot id.
    with open(DATAPATH) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            robot_id = int(row[0])
            current_time = float(row[1])
            px = float(row[3])
            py = float(row[4])
            pz = float(row[5])

            if current_time not in positions_by_time:
                positions_by_time[current_time] = {}

            positions_by_time[current_time][robot_id] = (px, py, pz)

    return positions_by_time


def proximity_events(critical_distance):

    if critical_distance < 0:
        print("distance can't be negative")
        return []

    positions_by_time = read_robot_positions_by_time()
    events = [] # list of tuples (t, r1, r2, dist)

    # For each current_time get the list of robot IDs
    for current_time, robots in sorted(positions_by_time.items()):
        robot_ids = list(robots.keys())
        n = len(robot_ids)
        # Compare all pairs on their distance
        for i in range(n):
            robot_id1 = robot_ids[i]
            x1, y1, z1 = robots[robot_id1]
            for j in range(i + 1, n):
                robot_id2 = robot_ids[j]
                x2, y2, z2 = robots[robot_id2]
                dx = x2 - x1
                dy = y2 - y1
                dz = z2 - z1
                dist = (dx ** 2 + dy ** 2 + dz ** 2)**0.5
                # if in the desired distance add in the list
                if dist < critical_distance:
                    events.append((current_time, robot_id1, robot_id2, dist))

    return events

########################## TACO CODE #############################

DATAPATH = "data/smart_factory_robots.csv"
# DATAPATH = "C:/Users/Ιωάννης Βλάσσης/Downloads/smart_factory_robots.csv"

def get_avg_speeds():
    # Dictionary of the form: robot_avgs[robot ID] = [average speed, count of measurements]
    robot_avgs = {}

    with open(DATAPATH) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            # Get the data from the datafile
            robot_id = int(row[0])
            vx = float(row[6])
            vy = float(row[7])

            # Given formula
            speed = (vx ** 2 + vy ** 2)**0.5

            # If we don't initialize, the addition below, will fail
            if robot_id not in robot_avgs:
                robot_avgs[robot_id] = [0, 0]

            # Get the current average speed and measurement counts
            current_avg = robot_avgs[robot_id][0]
            current_count = robot_avgs[robot_id][1]
            # Use the current values to calculate the new ones
            robot_avgs[robot_id][0] = ((current_avg * current_count) + speed) / (current_count + 1)
            robot_avgs[robot_id][1] = current_count + 1

    return robot_avgs

def top_speed(robot_count: int):
    # First, sort the dictionary containing the average speeds by ascending avg speed order
    sorted_avg_speeds = dict(sorted(get_avg_speeds().items(), key=lambda x: x[1], reverse=True))

    # Create a dictionary to hold the number of the requested robots. Will be returned
    top_avg_speeds = {}

    i = 0
    for robot_id, (avg, count) in sorted_avg_speeds.items():
        # Stop adding robots if we reached the desired number
        if i == robot_count:
            break

        # Move the next robot to the return dictionary
        top_avg_speeds[robot_id] = [avg, count]
        i += 1

    return top_avg_speeds

def collisions(start_time, end_time=None):
    # Dictionary of the form: robots_collided[robot ID] = [collision count, time of first collision, time of last collision]
    robots_collided = {}

    with open(DATAPATH) as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            # Retrieve the needed robot data from the csv file
            robot_id = int(row[0])
            current_time = float(row[1])
            goal_status = row[8]

            # Check if the robot crashed inside the requested time frame
            if (start_time <= current_time) and ((end_time is None) or (current_time < end_time)) and (goal_status == "collision detected"):
                # If it's the robot's first entry we have to initialize the values
                if robot_id not in robots_collided:
                    robots_collided[robot_id] = [0, current_time, 0]

                # Increase the number of collisions
                robots_collided[robot_id][0] += 1
                # Update the time of last collision to be the same as this collision
                robots_collided[robot_id][2] = current_time

    return robots_collided

# To be implemented down the line. Compares two robots
# def is_better(robot_a, robot_b):

def dominance():
    # Dictionary of the form: dominant_robots[robot_id] = [avg_speed, idle_ratio, collision_count]
    dominant_robots = {}

    with open(DATAPATH) as file:
        reader = csv.reader(file)
        next(reader)

        # Set containing all the existing robot IDs
        robot_ids = set()
        for row in reader:
            # Add each ID in the set
            robot_ids.add(int(row[0]))

    # Get the data the required data for the comparisons
    robot_avgs = get_avg_speeds()
    robot_idle_ratios = idle_ratio(min_percentage=0, max_percentage=100)
    robot_collisions = collisions(0)

    # Compare each robot with all the others, stop when you find one that's better
    for base_robot_id in robot_ids:
        for candidate_robot_id in robot_ids:
            base_avg = robot_avgs[base_robot_id][0]
            candidate_avg = robot_avgs[candidate_robot_id][0]

            base_idle_ratio = robot_idle_ratios[base_robot_id][0]
            candidate_idle_ratio = robot_idle_ratios[candidate_robot_id][0]

            base_collisions = robot_collisions[base_robot_id][0]
            candidate_collisions = robot_collisions[candidate_robot_id][0]
            # The actual comparison
            if (base_avg <= candidate_avg and
                base_idle_ratio >=candidate_idle_ratio and
                base_collisions >= candidate_collisions
            ):
                # If the robot is better, add it to the dictionary of dominant robots
                dominant_robots[candidate_robot_id] = [candidate_avg, candidate_idle_ratio, candidate_collisions]
                break

    return dominant_robots

def similar_robots(similarity_ratio: float):
    # Dictionary of the form: similar_robots[frozenset([robot_A_id, robot_B_id])] = similarity_ratio
    robot_similarities = {}

    with open(DATAPATH) as file:
        reader = csv.reader(file)
        next(reader)

        # Set containing all the available robot IDs
        robot_ids = set()
        for row in reader:
            # Add each ID in the set
            robot_ids.add(int(row[0]))

        all_robot_measurements = get_all_measurements()

        # Compare each robot with all the other robots and find their similarity
        for robot_a in robot_ids:
            for robot_b in robot_ids:
                # We don't need to calculate the same pair twice
                if (robot_a == robot_b) or (frozenset([robot_a, robot_b]) in robot_similarities.keys()):
                    continue

                similarity = calc_similarity(robot_a, robot_b, all_robot_measurements)
                # Keep it only if it meets the similarity requirement
                if similarity >= similarity_ratio:
                    robot_similarities[frozenset([robot_a, robot_b])] = similarity
                else:
                    robot_similarities[frozenset([robot_a, robot_b])] = None
    return robot_similarities

def calc_similarity(robot_a_id, robot_b_id, all_robot_measurements):
    # Get the speed measurements of the robots
    robot_a_measurements = all_robot_measurements[robot_a_id]
    robot_b_measurements = all_robot_measurements[robot_b_id]

    # Initialize the values that will be used in the cosine similarity formula
    numerator = 0
    denominator_a = 0
    denominator_b = 0

    # To find the similarity, we must compare the corresponding measurements
    # Therefore we iterate through each measurement of robot A and see if there is a corresponding for robot B
    for curr_time, (vx_a, vy_a) in robot_a_measurements.items():

        if curr_time not in robot_b_measurements:
            continue

        vx_b = robot_b_measurements[curr_time][0]
        vy_b = robot_b_measurements[curr_time][1]

        numerator += (vx_a * vx_b) + (vy_a * vy_b)
        denominator_a += vx_a ** 2 + vy_a ** 2
        denominator_b += vx_b ** 2 + vy_b ** 2

    # See the cosine similarity formula given
    return numerator / ((denominator_a ** 0.5) * (denominator_b ** 0.5))

def get_measurements(robot_id):
    # Dictionary of the form: robot_measurements[measurement_time] = [vx, vy]
    robot_measurements = {}

    with open(DATAPATH) as file:
        reader = csv.reader(file)
        next(reader)

        # We have to find all the rows in the datafile containing measurements for the given robot
        for row in reader:
            row_id = int(row[0])

            # All the robot measurements are gathered together in the data file.
            # This means that if the ID in the current row is not the robot's, but we have already got data for that robot,
            # then that robot's data is over
            if (row_id != robot_id) and robot_measurements:
                break

            # If the robot's ID matches, get the measurement data
            if row_id == robot_id:
                measurement_time = float(row[1])
                vx = float(row[6])
                vy = float(row[7])
                robot_measurements[measurement_time] = [vx, vy]
    return robot_measurements

def get_all_measurements():
    # Dictionary of the form: robot_measurements[measurement_time] = {measurement_time : [vx, vy]}
    robot_measurements = {}

    with open(DATAPATH) as file:
        reader = csv.reader(file)
        next(reader)

        # We have to find all the rows in the datafile containing measurements for the given robot
        for row in reader:
            robot_id = int(row[0])
            measurement_time = float(row[1])
            vx = float(row[6])
            vy = float(row[7])

            if robot_id not in robot_measurements:
                # Dictionary of the form: robot_measurements[robot_id][measurement_time] = [vx, vy]
                robot_measurements[robot_id] = {}

            # Get the measurement data
            robot_measurements[robot_id][measurement_time] = [vx, vy]
    return robot_measurements

def lagged_corr(robot_a, robot_b, lag):
    # Variables that track the best possible lag for the pair of robots and the correlation corresponding to it
    best_lag = None
    best_correlation = None

    # Get all the robots' measurements and their average speeds first, for efficiency
    all_robot_measurements = get_all_measurements()
    all_robot_avg_speeds = get_avg_speeds()

    # For every possible lag from -lag to lag, calculate the correlation the pair of robots' have
    for curr_lag in range(-lag, lag + 1):
        correlation = calc_correlation(robot_a, robot_b, curr_lag, all_robot_measurements, all_robot_avg_speeds)
        # If the new lag gives the best results update the best one
        if (best_correlation is None) or (correlation > best_correlation):
            best_correlation = correlation
            best_lag = curr_lag

    return [best_lag, best_correlation]

def calc_correlation(robot_a, robot_b, lag, robot_measurements, robot_avg_velocities):
    numerator = 0
    denominator_a = 0
    denominator_b = 0

    # Get the average speed of the pair of robots and the times at which there are measurements
    available_measure_times = [time for time in robot_measurements[robot_a].keys()]
    avg_speed_a = robot_avg_velocities[robot_a][0]
    avg_speed_b = robot_avg_velocities[robot_b][0]

    # Set the required range of indexes that will be used to iterate through the measurements, depending on the sign of the lag
    if lag >= 0:
        measurements_range = range(0, len(available_measure_times) - lag)
    else:
        measurements_range = range(lag, len(available_measure_times))

    for i in measurements_range:
        # Velocity of a robot is: sqrt(vx^2 + vy^2)
        velocity_a = (robot_measurements[robot_a][available_measure_times[i]][0] ** 2 +
                      robot_measurements[robot_a][available_measure_times[i]][1] ** 2) ** 0.5
        lag_velocity_b = (robot_measurements[robot_b][available_measure_times[i + lag]][0] ** 2 +
                      robot_measurements[robot_b][available_measure_times[i + lag]][1] ** 2) ** 0.5

        # Calculate the correlation using the formula given
        numerator += (velocity_a - avg_speed_a) * (lag_velocity_b - avg_speed_b)
        denominator_a += (velocity_a - avg_speed_a) ** 2
        denominator_b += (lag_velocity_b - avg_speed_b) ** 2

    return numerator / ((denominator_a ** 0.5) * (denominator_b ** 0.5))



######################## GENERAL CODE ############################

def menu():

    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger(__name__)

    while True:
        choice = input(f'Please Choose your option:')
        logger.info("User selected menu option: %s", choice)
        match choice:
            case '1':
                min_speed = float(input(f'Please choose your minimum speed:'))
                max_speed = float(input(f'Please choose your maximum speed:'))
                logger.info("User requested avg_speed with min=%.4f max=%.4f", min_speed, max_speed)
                avg_speeds = avg_speed(min_speed, max_speed)
                if not avg_speeds:
                    print('No robots found in the specified speed range.')
                else:
                    print(f'{'robot_id':<8} {'Μέση Ταχύτητα (m/s)':<19} {'Αριθμός Εγγραφών':<15}')
                    for robot_id, (avg, count) in avg_speeds.items():
                        print(f'{robot_id:<8} {avg:<19.4f} {count:<15}')
            case '2':
                robot_count = int(input(f'Please choose how many robots:'))

                top_speed_robots = top_speed(robot_count)
                print(f'{'Rank'}  {'robot_id'}  {'Μέση Ταχύτητα (m/s)'}  {'Αριθμός Εγγραφών'}')

                i = 0
                for robot_id, (avg, count) in top_speed_robots.items():
                    print(f'{i:<4}  {robot_id:<8}  {avg:<19.4f}  {count:<15}')
                    i += 1

            case '3':
                min_percentage = float(input(f'Please choose the minimum percentage of time:'))
                max_percentage = float(input(f'Please choose the maximum percentage of time:'))
                logger.info("User requested idle_ratio between %.2f and %.2f", min_percentage, max_percentage)
                filtered = idle_ratio(min_percentage, max_percentage)
                if not filtered:
                    print("No robots found in the specified idle ratio range.")
                else:
                    print(f'{'robotID':<8} {'Ποσοστό Αδράνειας':<19} {'Αριθμός Εγγραφών':<15}')
                    for robot_id, (ratio, count) in filtered.items():
                        print(f'{robot_id:<8} {ratio:<19.4f} {count:<15}')
            case '4':
                start_time = float(input(f'Please choose the start time:'))
                end_time = float(input(f'Please choose the end time:'))
                robot_collisions = collisions(start_time, end_time)

                print(f'Robot ID  Collision Count  Timeframe')
                for robot_id, (count, first_collision_time, last_collision_time) in robot_collisions.items():
                    print(f'{robot_id:<8}  {count:<15}  {f'{first_collision_time:.1f}-{last_collision_time:.1f}':<9}')

            case '5':
                deadlock_steps = int(input(f'Please choose the minimum amount of steps the robot was in a deadlock for:'))
                logger.info("User requested deadlocks with minimum steps=%d", deadlock_steps)
                deadlocks_results = deadlocks(deadlock_steps)
                if not deadlocks_results:
                    print('No deadlock streak found for the desired amount of steps.')
                else:
                    print(f'{'robotID':<8} {'Έναρξη':<19} {'Λήξη':<19} {'Μήκος':<9}')
                    for robot_id, streak in deadlocks_results.items():
                        for start, end, length in streak:
                            print(f'{robot_id:<8} {start:<19.1f} {end:<19.1f} {length:<9}')
            case '6':
                dominant_robots = dominance()
                print(f'Robot ID  Average Speed  Idle Ratio  Collision Count')

                for robot_id, (avg, idle_ratio, collisions) in dominant_robots.items():
                    print(f'{robot_id:<8}  {avg:<13.4f }  {idle_ratio:<10.4f}  {collisions:<15}')
            case '7':
                active_steps = int(input(f'Please choose the minimum number of the robot\'s active steps:'))
                average_displacement_per_step = float(input(f'Please choose the robot\'s average displacement per step:'))
                logger.info("User requested iceberg with active_steps=%d avg_disp_threshold=%.2f", active_steps, average_displacement_per_step)
                results = iceberg(active_steps, average_displacement_per_step)
                if not results:
                    print('No robots found meeting the Iceberg criteria.')
                else:
                    print(f'{'robotID':<10} {'Μέση Μετατόπιση':<20} {'Αριθμός Εγγραφών':<10}')
                    for robot_id, (avg_disp, count) in results.items():
                        print(f'{robot_id:<10} {avg_disp:<20.2f} {count:<10}')
            case '8':
                similarity_threshold = float(input(f'Please enter the cosine similarity the robots should have: '))
                robot_similarities = similar_robots(similarity_threshold)

                print(f'Robot A ID  Robot B ID  Cosine Similarity')

                for id_pair, similarity_ratio in robot_similarities.items():
                    if similarity_ratio is None:
                        continue
                    id_a, id_b = id_pair
                    print(f'{id_a:<10}  {id_b:<10}  {similarity_ratio:<17.2f}')
            case '9':
                critical_distance = float(input(f'Please enter the maximum distance two robots can reach without danger of crashing:'))
                logger.info("User requested proximity events with critical distance=%.2f", critical_distance)
                event_results = proximity_events(critical_distance)
                if not event_results:
                    print('No robots found in critical distance.')
                else:
                    print(f'{'Χρόνος':<10} {'Robot 1':<8} {'Robot 2':<8} {'Απόσταση':<10}')
                    for time, robot_id1, robot_id2, dist in event_results:
                        print(f'{time:<10.2f} {robot_id1:<8} {robot_id2:<8} {dist:<10.2f}')
            case '10':
                robot_a = int(input(f'Please choose the first robot:'))
                robot_b = int(input(f'Please choose the second robot:'))
                time_lag = int(input(f'Please choose the lag time:'))
                best_lag, best_correlation = lagged_corr(robot_a, robot_b, time_lag)

                print(f'Robot A ID  Robot B ID  Best lag  Correlation at best lag')
                print(f'{robot_a:<10}  {robot_b:<10}  {best_lag:<8}  {best_correlation:<23.4f}')
menu()

