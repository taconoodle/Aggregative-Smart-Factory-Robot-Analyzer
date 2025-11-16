import csv
import logging
import logging.config
import os
import heapq
import sys
import tempfile
from collections import defaultdict

FILENAME = "smart_factory_robots_5000x.csv"
CACHEPATH = f"data/cache/{FILENAME}"
DATAPATH = f"data/{FILENAME}"

###################################### CACHING ######################################

def chunk_csv(filepath):
    chunksize = 10000
    sorted_chunks = []

    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)

        # Append each row in a chunk
        chunk = []
        for row in reader:
            chunk.append(row)
            # When a chunk fills sort and save it in a tempfile
            if len(chunk) >= chunksize:
                # Sort the lines in the chunk based on robot_id and current_time
                chunk.sort(key = lambda x: (int(x[0]), float(x[1])))

                # Create a temporary file to keep the chunk and write the chunk in it
                with tempfile.NamedTemporaryFile(delete=False, mode='w', newline="") as temp_file:
                    writer = csv.writer(temp_file)
                    writer.writerow(header)
                    writer.writerows(chunk)
                    sorted_chunks.append(temp_file.name)
                chunk = []

        # Handle the chunk that remains when file ends
        if chunk:
            chunk.sort(key=lambda x: (int(x[0]), float(x[1])))
            with tempfile.NamedTemporaryFile(delete=False, mode='w', newline="") as temp_file:
                writer = csv.writer(temp_file)
                writer.writerow(header)
                writer.writerows(chunk)
                sorted_chunks.append(temp_file.name)

    return header, sorted_chunks

def yield_data(file, offset=None):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        if offset:
            f.seek(offset)
        for row in reader:
            yield row

def merge_sorted_chunks(sorted_chunks):
    row_generators = []

    # Get the generators that will generate the rows for the merge
    for temp_file in sorted_chunks:
        row_generators.append(yield_data(temp_file))

    # Create a generator that gives you the next sorted row
    merged_chunks = heapq.merge(*row_generators, key=lambda row: (int(row[0]), float(row[1])))
    return merged_chunks

def sort_csv_by_chunking(filepath):
    header, sorted_chunks = chunk_csv(filepath)
    merged_chunks = merge_sorted_chunks(sorted_chunks)

    # Create a temporary file and write the sorted data in it
    with open(f'data/temp.csv', 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in merged_chunks:
            writer.writerow(row)

    # Replace the original data file with the sorted one
    os.remove(filepath)
    os.rename('data/temp.csv', filepath)

def cache_data():
    # No need to cache if we already have a cache and the data hasn't changed
    data_mtime = os.path.getmtime(DATAPATH)
    if os.path.exists(CACHEPATH):
        with open(CACHEPATH, 'r') as cache_file:
            reader = csv.reader(cache_file)
            cache_mtime = float(next(reader)[0])

            if data_mtime == cache_mtime:
                return

    # If we reach this line, we have to create the cache from scratch
    # First sort the data file and get its modification time
    sort_csv_by_chunking(DATAPATH)
    data_mtime = os.path.getmtime(DATAPATH)

    # Open the data file and the cache file
    header = ['robot_id', 'offset', 'count',
              'avg_speed', 'idle_ratio', 'total_collisions', 'avg_displacement', 'active_steps']
    with open(DATAPATH, 'r') as data_file, open(CACHEPATH, 'w', newline="\n") as cache_file:
        writer = csv.writer(cache_file, delimiter=',')

        # First put the mtime of the data file and the header in the cache
        data_file.readline()
        writer.writerow([data_mtime])
        writer.writerow(header)

        # Initialize variables
        prev_robot_id = 0
        count = 0
        prev_offset = data_file.tell()
        while True:
            # Get the next line
            current_offset = data_file.tell()
            input_row = data_file.readline()
            #If there are no lines, we are done. Write the last robot data in the cache
            if not input_row:
                writer.writerow([prev_robot_id, prev_offset, count, None, None, None, None, None])
                break

            input_row = input_row.strip('\n').split(',')

            current_robot_id = int(input_row[0])

            #When the ID changes, it means the previous ID's data is over. Write it in the file
            if current_robot_id != prev_robot_id:
                writer.writerow([prev_robot_id, prev_offset, count, None, None, None, None, None])

                # Re-initialize the variables for the next ID
                prev_robot_id = current_robot_id
                prev_offset = current_offset
                count = 0

            count += 1

def yield_cache():
    with open(CACHEPATH, 'r') as cache:
        reader = csv.reader(cache)
        next(reader)
        next(reader)
        for row in reader:
            yield row

##################################################################

def get_measurements(robot_id):
    # Get the offset of the robot_id's measurements from the cache
    for cache_id, cache_offset, *_ in yield_cache():
        if int(cache_id) == robot_id:
            offset = int(cache_offset)
            break

    # Get the robot's measurements
    for row in yield_data(DATAPATH, offset):
        row_id = int(row[0])
        # When the ID changes the measurements are finished
        if row_id != robot_id:
            break

        # robot_id, current_time, current_time_step, px, py, pz, vx, vy, goal_status, idle, linear, rotational, deadlock_bool, robot_body_contact = row
        yield row

def cache_avg_speeds():
    cache_rows = []
    with open(CACHEPATH) as cache:
        reader = csv.reader(cache)
        # Skip mtime
        mtime = next(reader)
        # Skip header
        header = next(reader)
        # Get all robots' measurement counts
        for row in reader:
            cache_rows.append(row)


    with open(DATAPATH, 'r') as data_file, open(CACHEPATH, 'w', newline='\n') as cache_file:
        reader = csv.reader(data_file)
        writer = csv.writer(cache_file)

        writer.writerow(mtime)
        writer.writerow(header)

        next(reader)
        prev_robot_id = 0
        total_speed = 0
        robot_count = int(cache_rows[0][2])
        for row in reader:
            # Get the data from the datafile
            robot_id = int(row[0])

            if robot_id != prev_robot_id:
                avg = total_speed / robot_count
                cache_rows[prev_robot_id][3] = avg
                writer.writerow(cache_rows[prev_robot_id])

                robot_count = int(cache_rows[robot_id][2])
                total_speed = 0
                prev_robot_id = robot_id

            vx = float(row[6])
            vy = float(row[7])

            # Given formula
            total_speed += (vx ** 2 + vy ** 2)**0.5

        avg = total_speed / robot_count
        cache_rows[prev_robot_id][3] = avg
        writer.writerow(cache_rows[prev_robot_id])

def get_avg_speeds():
    with open(CACHEPATH, 'r') as cache:
        reader = csv.reader(cache)
        next(reader)
        next(reader)
        row = cache.readline().strip('\n').split(',')[3]
        if row == '':
            cache_avg_speeds()

    for row in yield_cache():
        robot_id, _, count, avg, *_ = row
        yield robot_id, avg, count

def avg_speed(min_speed, max_speed):
    if min_speed > max_speed:
        print("The min speed cannot be greater than the max speed")
        return 0

    robot_avg = get_avg_speeds()

    # for robot_id, (avg, count) in robot_avg.items():
    for robot_id, avg, count, in robot_avg:
        if min_speed < float(avg) <= max_speed:
            yield robot_id, avg, count

def top_speed(robot_count: int):
    # List the will be used as a heap to keep the average speeds
    top_avg_speeds = []

    for robot_id, avg, count in get_avg_speeds():
        # Push the value given by the generator into the min heap
        heapq.heappush(top_avg_speeds, (float(avg), int(count), robot_id))
        # If we have more values than we need, we pop the min
        if len(top_avg_speeds) > robot_count:
            heapq.heappop(top_avg_speeds)

    return sorted(top_avg_speeds, reverse=True)

def cache_idle_ratios():
    cache_rows = []
    with open(CACHEPATH, 'r') as cache:
        reader = csv.reader(cache)
        # Skip mtime
        mtime = next(reader)
        # Skip header
        header = next(reader)
        # Get all robots' measurement counts
        for row in reader:
            cache_rows.append(row)

    with open(CACHEPATH, 'w', newline='') as cache_file:
        writer = csv.writer(cache_file)
        # Re-initialize the cache with the data it had
        writer.writerow(mtime)
        writer.writerow(header)

        # Initialize the values for the calculation of the first robot
        prev_robot_id = 0
        idle_time = 0.0
        total_time = 0.0
        prev_time = -1
        for robot_id, current_time, _, _, _, _, _, _, _, idle_state, *_ in yield_data(DATAPATH):
            robot_id = int(robot_id)
            current_time = float(current_time)
            idle_state = eval(idle_state)
            # When robot changes, save the data of the previous robot and initialize the new one
            if robot_id != prev_robot_id:
                # There is the possibility we only have one measurement
                if total_time != 0.0:
                    ratio = idle_time / total_time
                elif prev_idle_state:
                    ratio = 1
                else:
                    ratio = 0

                # Write the data of the previous robot
                cache_rows[prev_robot_id][4] = ratio
                writer.writerow(cache_rows[prev_robot_id])

                # Initialize the new robot's data
                idle_time = 0.0
                total_time = 0.0
                prev_time = -1
                prev_robot_id = robot_id

            # If prev_time is -1, that means we are on the first measurement of the robot
            # and there is no previous time data to add
            if prev_time != -1:
                dt = current_time - prev_time
                total_time += dt
                if prev_idle_state:
                    idle_time += dt

            prev_time = current_time
            prev_idle_state = idle_state

        # Handle the last robot's data
        if total_time != 0.0:
            ratio = idle_time / total_time
        elif prev_idle_state:
            ratio = 1
        else:
            ratio = 0

        # Write the data of the previous robot
        cache_rows[prev_robot_id][4] = ratio
        writer.writerow(cache_rows[prev_robot_id])

def get_idle_ratios():
    with open(CACHEPATH, 'r') as cache:
        reader = csv.reader(cache)
        next(reader)
        next(reader)
        row = cache.readline().strip('\n').split(',')[4]
        if row == '':
            cache_idle_ratios()

    for row in yield_cache():
        robot_id, _, count, _, ratio, *_ = row
        yield robot_id, count, ratio

def idle_ratio(min_percentage, max_percentage):
    if min_percentage > max_percentage:
        print("The min percentage cannot be greater than the max percentage.")
        return None

    for robot_id, count, ratio in get_idle_ratios():
        if min_percentage <= float(ratio) <= max_percentage:
            yield robot_id, ratio, count

def cache_collisions():
    cache_rows = []
    with open(CACHEPATH, 'r') as cache:
        reader = csv.reader(cache)
        # Skip mtime
        mtime = next(reader)
        # Skip header
        header = next(reader)
        # Get all robots' measurement counts
        for row in reader:
            cache_rows.append(row)

    with open(CACHEPATH, 'w', newline='') as cache_file:
        writer = csv.writer(cache_file)
        writer.writerow(mtime)
        writer.writerow(header)

        prev_robot_id = 0
        colls = 0
        for robot_id, _, _, _, _, _, _, _, goal_status, *_ in yield_data(DATAPATH):
            robot_id = int(robot_id)

            if robot_id != prev_robot_id:
                cache_rows[prev_robot_id][5] = colls
                writer.writerow(cache_rows[prev_robot_id])

                prev_robot_id = robot_id
                colls = 0

            if goal_status == "collision detected":
                colls += 1

        # Handle the last robot
        cache_rows[prev_robot_id][5] = colls
        writer.writerow(cache_rows[prev_robot_id])

def get_collisions_from_cache():
    with open(CACHEPATH, 'r') as cache:
        reader = csv.reader(cache)
        next(reader)
        next(reader)
        row = cache.readline().strip('\n').split(',')[5]
        if row == '':
            cache_collisions()

    for row in yield_cache():
        robot_id, _, _, _, _, collision_count, *_ = row
        yield robot_id, collision_count

def collisions(start_time, end_time=None):
    if start_time == 0 and end_time is None:
        for robot_id, collision_count in get_collisions_from_cache():
            yield robot_id, collision_count
        return

    prev_robot_id = 0
    colls = 0
    first_coll_time = 0
    last_coll_time = 0
    for robot_id, current_time, _, _, _, _, _, _, goal_status, *_ in yield_data(DATAPATH):
        robot_id = int(robot_id)
        current_time = float(current_time)

        if robot_id != prev_robot_id:
            if colls != 0:
                yield prev_robot_id, colls, first_coll_time, last_coll_time

            prev_robot_id = robot_id
            colls = 0
            first_coll_time = 0
            last_coll_time = 0

        if start_time <= current_time and (end_time is None or current_time < end_time)\
                and goal_status == "collision detected":
            if colls == 0:
                first_coll_time = current_time
            colls += 1
            last_coll_time = current_time
    # Yield the last robot
    if colls != 0:
        yield prev_robot_id, colls, first_coll_time, last_coll_time

def deadlocks(deadlock_steps):
    in_deadlock = False
    start_time = 0
    end_time = 0
    counter = 0
    prev_robot_id = 0
    for robot_id, current_time, _, _, _, _, _, _, _, _, _, _, deadlock_state, *_ in yield_data(DATAPATH):
        robot_id = int(robot_id)
        current_time = float(current_time)
        deadlock_state = eval(deadlock_state)

        if prev_robot_id != robot_id:
            if in_deadlock and counter >= deadlock_steps:
                yield prev_robot_id, start_time, end_time, counter

            prev_robot_id = robot_id
            in_deadlock = False
            end_time = 0
            start_time = 0
            counter = 0

        if deadlock_state:
            # if deadlock occurs 1st time in a row starts the streak count
            if not in_deadlock:
                in_deadlock = True
                start_time = current_time
                end_time = current_time
                counter = 1
            else:  # if not deadlock happens 1st time in a row add to the streak count
                end_time = current_time
                counter += 1
        else:
            # if deadlock doesn't occur but already on the desired range streak keep the info
            if in_deadlock and counter >= deadlock_steps:
                yield robot_id, start_time, end_time, counter
            in_deadlock = False
            counter = 0

    # If deadlock continues until the end keep the info
    if in_deadlock and counter >= deadlock_steps:
        yield prev_robot_id, start_time, end_time, counter

def dominance():
    # Dictionary of the form: dominant_robots[robot_id] = [avg_speed, idle_ratio, collision_count]
    dominant_robots = set()

    # Get the data the required data for the comparisons
    robot_ids = list(int(robot_id) for robot_id, *_ in yield_cache())
    robot_avgs = list(get_avg_speeds())
    robot_idle_ratios = list(idle_ratio(min_percentage=0, max_percentage=100))
    robot_collisions = list(collisions(0))

    # Compare each robot with all the others, stop when you find one that's better
    for base_robot_id in robot_ids:
        for candidate_robot_id in robot_ids:
            if base_robot_id == candidate_robot_id:
                continue

            base_avg = float(robot_avgs[base_robot_id][1])
            candidate_avg = float(robot_avgs[candidate_robot_id][1])

            base_idle_ratio = float(robot_idle_ratios[base_robot_id][1])
            candidate_idle_ratio = float(robot_idle_ratios[candidate_robot_id][1])

            base_collisions = int(robot_collisions[base_robot_id][1])
            candidate_collisions = int(robot_collisions[candidate_robot_id][1])

            # The actual comparison
            if (base_avg <= candidate_avg and
                base_idle_ratio >=candidate_idle_ratio and
                base_collisions >= candidate_collisions
            ):
                # If the robot is better, add it to the dictionary of dominant robots
                if candidate_robot_id not in dominant_robots:
                    dominant_robots.add(candidate_robot_id)
                    yield candidate_robot_id, candidate_avg, candidate_idle_ratio, candidate_collisions
                break

def cache_displacements():
    cache_rows = []
    with open(CACHEPATH) as cache:
        reader = csv.reader(cache)
        # Skip mtime
        mtime = next(reader)
        # Skip header
        header = next(reader)
        # Get all robots' measurement counts
        for row in reader:
            cache_rows.append(row)

    with open(CACHEPATH, 'w', newline='') as cache:
        writer = csv.writer(cache)
        writer.writerow(mtime)
        writer.writerow(header)

        prev_robot_id = 0
        n_active = 0
        total_dis = 0
        prev_px, prev_py, prev_pz = 0, 0, 0
        for robot_id, _, _, px, py, pz, _, _, _, idle_state, *_ in yield_data(DATAPATH):
            robot_id = int(robot_id)
            px = float(px)
            py = float(py)
            pz = float(pz)
            idle_state = eval(idle_state)

            # Write the old robot's data and initialize the next
            if robot_id != prev_robot_id:
                avg_dis = total_dis / n_active
                cache_rows[prev_robot_id][6] = avg_dis
                cache_rows[prev_robot_id][7] = n_active
                writer.writerow(cache_rows[prev_robot_id])

                prev_robot_id = robot_id
                n_active = 0
                total_dis = 0
                prev_px, prev_py, prev_pz = 0, 0, 0

            # If robot is active calculate its new displacement
            if not idle_state:
                n_active += 1
                dx = px - prev_px
                dy = py - prev_py
                dz = pz - prev_pz
                displacement = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
                total_dis += displacement

                prev_px, prev_py, prev_pz = px, py, pz

        # Handle the last robot
        avg_dis = total_dis / n_active
        cache_rows[prev_robot_id][6] = avg_dis
        cache_rows[prev_robot_id][7] = n_active
        writer.writerow(cache_rows[prev_robot_id])

def get_avg_displacements():
    with open(CACHEPATH, 'r') as cache:
        reader = csv.reader(cache)
        next(reader)
        next(reader)
        row = cache.readline().strip('\n').split(',')[6]
        if row == '':
            cache_displacements()

    for robot_id, _, count, _, _, _, avg_displacement, active_steps, *_ in yield_cache():
        yield robot_id, count, avg_displacement, active_steps

def iceberg(active_steps, average_displacement_per_step):

    if active_steps <= 0:
        print("Invalid active steps")
        return 0

    for robot_id, count, cache_avg_displacement, cache_active_steps in get_avg_displacements():
        if int(cache_active_steps) >= active_steps and float(cache_avg_displacement) > average_displacement_per_step:
            yield robot_id, cache_avg_displacement, count

def calc_similarity(robot_a_id, robot_b_id, all_measurements):
    # Get the speed measurements of the robots
    robot_a_measurements = all_measurements[robot_a_id]
    robot_b_measurements = all_measurements[robot_b_id]

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

def similar_robots(similarity_ratio: float):
    robot_ids = list(int(robot_id) for robot_id, *_ in yield_cache())

    # Compare each robot with all the other robots and find their similarity
    pairs_calculated = []
    all_measurements = defaultdict(dict)
    for robot_id, curr_time, _, _, _, _, vx, vy, *_ in yield_data(DATAPATH):
        all_measurements[int(robot_id)][curr_time] = (float(vx), float(vy))

    for robot_a in robot_ids:
        for robot_b in robot_ids:
            pair = {robot_a, robot_b}
            # We don't need to calculate the same pair twice
            if robot_a == robot_b or pair in pairs_calculated:
                continue

            similarity = calc_similarity(robot_a, robot_b, all_measurements)
            pairs_calculated.append(pair)

            # Keep it only if it meets the similarity requirement
            if similarity >= similarity_ratio:
                yield robot_a, robot_b, similarity

def proximity_events(critical_distance):

    if critical_distance < 0:
        print("distance can't be negative")
        return []

    positions_by_time = defaultdict(dict)
    for robot_id, curr_time, _, px, py, pz, *_ in yield_data(DATAPATH):
        positions_by_time[curr_time][int(robot_id)] = (float(px), float(py), float(pz))

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
                    yield current_time, robot_id1, robot_id2, dist

def lagged_corr(robot_a, robot_b, lag):
    # Variables that track the best possible lag for the pair of robots and the correlation corresponding to it
    best_lag = None
    best_correlation = None

    # Get all the robots' measurements and their average speeds first, for efficiency
    robot_a_measurements = {}
    robot_b_measurements = {}
    for _, curr_time, _, _, _, vx, vy, *_ in get_measurements(robot_a):
        robot_a_measurements[curr_time] = (float(vx), float(vy))
    for _, curr_time, _, _, _, vx, vy, *_ in get_measurements(robot_b):
        robot_b_measurements[curr_time] = (float(vx), float(vy))

    all_robot_avg_speeds = get_avg_speeds()

    # For every possible lag from -lag to lag, calculate the correlation the pair of robots' have
    for curr_lag in range(-lag, lag + 1):
        correlation = calc_correlation(robot_a, robot_b, curr_lag, robot_a_measurements, robot_b_measurements)
        # If the new lag gives the best results update the best one
        if (best_correlation is None) or (correlation > best_correlation):
            best_correlation = correlation
            best_lag = curr_lag

    return best_lag, best_correlation

def calc_correlation(robot_a, robot_b, lag, robot_a_measurements, robot_b_measurements):
    numerator = 0
    denominator_a = 0
    denominator_b = 0

    # Get the average speed of the pair of robots and the times at which there are measurements
    available_measure_times = [time for time in robot_a_measurements.keys()]
    avg_speed_a = 0
    avg_speed_b = 0

    for robot_id, avg, _  in get_avg_speeds():
        if int(robot_id) == robot_a:
            avg_speed_a = float(avg)
        if int(robot_id) == robot_b:
            avg_speed_b = float(avg)

    # Set the required range of indexes that will be used to iterate through the measurements, depending on the sign of the lag
    if lag >= 0:
        measurements_range = range(0, len(available_measure_times) - lag)
    else:
        measurements_range = range(lag, len(available_measure_times))

    for i in measurements_range:
        # Velocity of a robot is: sqrt(vx^2 + vy^2)
        velocity_a = (robot_a_measurements[available_measure_times[i]][0] ** 2 +
                      robot_a_measurements[available_measure_times[i]][1] ** 2) ** 0.5
        lag_velocity_b = (robot_b_measurements[available_measure_times[i + lag]][0] ** 2 +
                      robot_b_measurements[available_measure_times[i + lag]][1] ** 2) ** 0.5

        # Calculate the correlation using the formula given
        numerator += (velocity_a - avg_speed_a) * (lag_velocity_b - avg_speed_b)
        denominator_a += (velocity_a - avg_speed_a) ** 2
        denominator_b += (lag_velocity_b - avg_speed_b) ** 2

    return numerator / ((denominator_a ** 0.5) * (denominator_b ** 0.5))

def menu():

    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger(__name__)

    while True:
        choice = input(f'Please choose your option:\n'
                       f'1. avg_speed\n'
                       f'2. top_speed\n'
                       f'3. idle_ratio\n'
                       f'4. collisions\n'
                       f'5. deadlocks\n'
                       f'6. dominance\n'
                       f'7. iceberg\n'
                       f'8. similar_robots\n'
                       f'9. proximity_events\n'
                       f'10. lagged_corr\n'
                       f'Choose \'e\' to exit:\n')
        logger.info("User selected menu option: %s", choice)
        match choice:
            case '1':
                min_speed = float(input(f'Please choose your minimum speed:'))
                max_speed = float(input(f'Please choose your maximum speed:'))
                logger.info("User requested avg_speed with min=%.4f max=%.4f", min_speed, max_speed)

                print(f'{'robot_id':<8} {'Μέση Ταχύτητα (m/s)':<19} {'Αριθμός Εγγραφών':<15}')
                for robot_id, avg, count in avg_speed(min_speed, max_speed):
                    print(f'{robot_id:<8} {float(avg):<19.4f} {count:<15}')
                logger.info("Finished running avg_speed")
            case '2':
                robot_count = int(input(f'Please choose how many robots:'))
                logger.info(f'User called top_speed: robot_count={robot_count}')
                top_speed_robots = top_speed(robot_count)
                print(f'{'Rank'}  {'robot_id'}  {'Μέση Ταχύτητα (m/s)'}  {'Αριθμός Εγγραφών'}')

                i = 0
                for avg, count, robot_id in top_speed_robots:
                    print(f'{i:<4}  {robot_id:<8}  {avg:<19.4f}  {count:<15}')
                    i += 1
                logger.info(f'Finished running top_speed')

            case '3':
                min_percentage = float(input(f'Please choose the minimum percentage of time:'))
                max_percentage = float(input(f'Please choose the maximum percentage of time:'))
                logger.info("User called idle_ratio between %.2f and %.2f", min_percentage, max_percentage)

                print(f'{'robotID':<8} {'Ποσοστό Αδράνειας':<19} {'Αριθμός Εγγραφών':<15}')
                for robot_id, ratio, count in idle_ratio(min_percentage, max_percentage):
                    print(f'{robot_id:<8} {float(ratio):<19.4f} {count:<15}')
                logger.info("Finished running idle_ratio")

            case '4':
                start_time = float(input(f'Please choose the start time:'))
                end_time = float(input(f'Please choose the end time:'))
                logger.info(f'User called collisions: start_time={start_time}  end_time={end_time}')
                robot_collisions = collisions(start_time, end_time)

                print(f'Robot ID  Collision Count  Timeframe')
                for robot_id, count, first_collision_time, last_collision_time in robot_collisions:
                    print(f'{robot_id:<8}  {count:<15}  {f'{first_collision_time:.1f}-{last_collision_time:.1f}':<9}')
                logger.info(f'Finished running collisions')

            case '5':
                deadlock_steps = int(input(f'Please choose the minimum amount of steps the robot was in a deadlock for:'))
                logger.info("User called deadlocks with minimum steps=%d", deadlock_steps)

                print(f'{'robotID':<8} {'Έναρξη':<19} {'Λήξη':<19} {'Μήκος':<9}')
                for robot_id, start, end, length in deadlocks(deadlock_steps):
                    print(f'{robot_id:<8} {start:<19.1f} {end:<19.1f} {length:<9}')
                logger.info("Finished running deadlocks")

            case '6':
                logger.info(f'User called dominance')
                dominant_robots = dominance()
                print(f'Robot ID  Average Speed  Idle Ratio  Collision Count')

                for robot_id, avg, ratio, colls in dominant_robots:
                    print(f'{robot_id:<8}  {avg:<13.4f }  {ratio:<10.4f}  {colls:<15}')
                logger.info(f'Finished running dominance')

            case '7':
                active_steps = int(input(f'Please choose the minimum number of the robot\'s active steps:'))
                average_displacement_per_step = float(input(f'Please choose the robot\'s average displacement per step:'))
                logger.info("User called iceberg with active_steps=%d avg_disp_threshold=%.2f", active_steps,
                            average_displacement_per_step)
                print(f'{'robotID':<10} {'Μέση Μετατόπιση':<20} {'Αριθμός Εγγραφών':<10}')
                for robot_id, avg_disp, count in iceberg(active_steps, average_displacement_per_step):
                    print(f'{robot_id:<10} {avg_disp:<20.2f} {count:<10}')
                logger.info("Finished running iceberg")

            case '8':
                similarity_threshold = float(input(f'Please enter the cosine similarity the robots should have: '))
                logger.info(f'User called similar_robots: similarity_threshold={similarity_threshold}')
                robot_similarities = similar_robots(similarity_threshold)

                print(f'Robot A ID  Robot B ID  Cosine Similarity')

                for robot_a, robot_b, similarity_ratio in robot_similarities:
                    if similarity_ratio is None:
                        continue
                    print(f'{robot_a:<10}  {robot_b:<10}  {similarity_ratio:<17.2f}')
                logger.info(f'Finished running similar_robots')

            case '9':
                critical_distance = float(input(f'Please enter the maximum distance two robots can reach without danger of crashing:'))
                logger.info("User called proximity events with critical distance=%.2f", critical_distance)

                print(f'{'Χρόνος':<10} {'Robot 1':<8} {'Robot 2':<8} {'Απόσταση':<10}')
                for time, robot_id1, robot_id2, dist in proximity_events(critical_distance):
                    print(f'{time:<10.2f} {robot_id1:<8} {robot_id2:<8} {dist:<10.2f}')
                logger.info("Finished running proximity_events")

            case '10':
                robot_a = int(input(f'Please choose the first robot:'))
                robot_b = int(input(f'Please choose the second robot:'))
                time_lag = int(input(f'Please choose the lag time:'))
                logger.info(f"User called lagged_corr: robot_a={robot_a}  robot_b={robot_b}  time_lag={time_lag}")
                best_lag, best_correlation = lagged_corr(robot_a, robot_b, time_lag)

                print(f'Robot A ID  Robot B ID  Best lag  Correlation at best lag')
                print(f'{robot_a:<10}  {robot_b:<10}  {best_lag:<8}  {best_correlation:<23.4f}')
                logger.info(f'Finished running lagged_corr')

            case 'e':
                sys.exit()

if __name__ == '__main__':
    cache_data()
    # menu()
    for row in collisions(0):
        print (row)