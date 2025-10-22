import csv
import math

######################## SMOKE CODE ##############################

def avg_speed(min_speed, max_speed):
    min_speed = float(min_speed)
    max_speed = float(max_speed)
    if min_speed > max_speed:
        print("The min speed cannot be greater than the max speed")
        return 0

    robot_info = {}

    # with open("data/smart_factory_robots_200.csv") as file:
    with open(DATAPATH) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            robot_id = int(row[0])
            vx = float(row[6])
            vy = float(row[7])
            speed = math.sqrt(vx**2 + vy**2)

            if robot_id not in robot_info:
                robot_info[robot_id] = [0, 0]

            #GIANNHS: Des mhpws se symferei na diaireis me th mia me to count kai na vazeis to average sti lista

            current_avg = robot_info[robot_id][0]
            current_count = robot_info[robot_id][1]
            new_avg = (current_avg * current_count + speed)/(current_count + 1)
            if new_avg > max_speed or new_avg < min_speed:
                continue

            robot_info[robot_id][0] = (current_avg * current_count + speed)/(current_count + 1)
            robot_info[robot_id][1] = current_count + 1

    print(f"{'robot_id'}  {'Μέση Ταχύτητα (m/s)'}  {'Αριθμός Εγγραφών'}")
    for robot_id, (avg, count) in robot_info.items():
        if min_speed < avg <= max_speed:
            print(f"{robot_id} {avg} {count}")
    #GIANNHS: Na epistrefoume th lista gia na th xrhsimopoioume
    return 1


########################## TACO CODE #############################

DATAPATH = "C:/Users/giann/Downloads/smart_factory_robots.csv"
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
            speed = math.sqrt(vx ** 2 + vy ** 2)

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

def collisions(start_time, end_time):
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
            if (start_time <= current_time < end_time) and (goal_status == "collision detected"):
                # If it's the robot's first entry we have to initialize the values
                if robot_id not in robots_collided: robots_collided[robot_id] = [0, current_time, 0]

                # Increase the number of collisions
                robots_collided[robot_id][0] += 1
                # Update the time of last collision to be the same as this collision
                robots_collided[robot_id][2] = current_time

    return robots_collided




######################## GENERAL CODE ############################

def menu():
    while True:
        choice = input(f'Please Choose your option:')
        match choice:
            case '1':
                min_speed = input(f'Please choose your minimum speed:')
                max_speed = input(f'Please choose your maximum speed:')
                avg_speed(min_speed, max_speed)
            case '2':
                robot_count = int(input(f'Please choose how many robots:'))

                top_speed_robots = top_speed(robot_count)
                print(f'{'Rank'}  {'robot_id'}  {'Μέση Ταχύτητα (m/s)'}  {'Αριθμός Εγγραφών'}')

                i = 0
                for robot_id, (avg, count) in top_speed_robots.items():
                    print(f'{i:<4}  {robot_id:<8}  {avg:<19.4f}  {count:<15}')
                    i += 1

            case '3':
                min_percentage = input(f'Please choose the minimum percentage of time:')
                max_percentage = input(f'Please choose the maximum percentage of time:')
                # idle_ratio(min_percentage, max_percentage)
            case '4':
                start_time = float(input(f'Please choose the start time:'))
                end_time = float(input(f'Please choose the end time:'))
                robot_collisions = collisions(start_time, end_time)

                print(f'Robot ID  Collision Count  Timeframe')
                for robot_id, (count, first_collision_time, last_collision_time) in robot_collisions.items():
                    print(f'{robot_id:<8}  {count:<15}  {f'{first_collision_time:.1f}-{last_collision_time:.1f}':<9}')

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