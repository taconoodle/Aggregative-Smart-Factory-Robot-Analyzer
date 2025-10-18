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

    with open("data/smart_factory_robots_200.csv") as file:
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
            robot_info[robot_id][0] += speed
            robot_info[robot_id][1] += 1

    print(f"{'robot_id'}  {'Μέση Ταχύτητα (m/s)'}  {'Αριθμός Εγγραφών'}")
    for robot_id, (sum_speed, count) in robot_info.items():
        avg = sum_speed / count
        if min_speed < avg <= max_speed:
            print(f"{robot_id} {avg} {count}")
    #GIANNHS: Na epistrefoume th lista gia na th xrhsimopoioume
    return 1


########################## TACO CODE #############################

datapath = "C:/Users/giann/Downloads/smart_factory_robots.csv"
def get_avg_speeds():
    #Dictionary of the form: robot_info[robot ID] = [cumulative speed, count of measurements]
    robot_info = {}

    with open(datapath) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            #Get the data from the datafile
            robot_id = int(row[0])
            vx = float(row[6])
            vy = float(row[7])

            #Given formula
            speed = math.sqrt(vx ** 2 + vy ** 2)

            #If we don't initialize, the addition under will fail
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

# def top_speed(robot_count):



######################## GENERAL CODE ############################

def menu():
    while True:
        choice = input(f'Please Choose your option:')
        match choice:
            case '1':
                min_speed = input(f'Please choose your minimum speed:')
                max_speed = input(f'Please choose your maximum speed:')
                avg_speed(min_speed, max_speed);
            case '2':
                robot_count = input(f'Please choose how many robots:')
                # top_speed(robot_count)
            case '3':
                min_percentage = input(f'Please choose the minimum percentage of time:')
                max_percentage = input(f'Please choose the maximum percentage of time:')
                # idle_ratio(min_percentage, max_percentage)
            case '4':
                start_time = input(f'Please choose the start time:')
                end_time = input(f'Please choose the end time:')
                # collisions(start_time, end_time)
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