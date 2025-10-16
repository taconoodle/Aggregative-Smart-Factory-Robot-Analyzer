def banana():

def menu():
    while True:
        choice = input(f'Please Choose your option:')
        match choice:
            case '1':
                min_speed = input(f'Please choose your minimum speed:')
                max_speed = input(f'Please choose your maximum speed:')
                # avg_speed(min_speed, max_speed);
            case '2':
                robot_count = input(f'Please choose how many robots:')
                top_speed(robot_count)
            case '3':
                min_percentage = input(f'Please choose the minimum percentage of time:')
                max_percentage = input(f'Please choose the maximum percentage of time:')
                idle_ratio(min_percentage, max_percentage)
            case '4':
                start_time = input(f'Please choose the start time:')
                end_time = input(f'Please choose the end time:')
                collisions(start_time, end_time)
            case '5':
                deadlock_steps = input(f'Please choose the minimum amount of steps the robot was in a deadlock for:')
                deadlocks(deadlock_steps)
            case '6':
                dominance()
            case '7':
                active_steps = input(f'Please choose the minimum number of the robot\'s active steps:')
                average_displacement_per_step = input(f'Please choose the robot\'s average displacement per step:')
                iceberg(active_steps, average_displacement_per_step)
            case '8':
                similarity = input(f'Please enter the cosine similarity the robots should have')
                similar_robots(similarity)
            case '9':
                critical_distance = input(f'Please enter the maximum distance two robots can reach without danger of crashing:')
                proximity_events()
            case '10':
                robot_A = input(f'Please choose the first robot:')
                robot_B = input(f'Please choose the second robot:')
                time_lags = input(f'Please choose the lag time:')
                lagged_corr()


menu()