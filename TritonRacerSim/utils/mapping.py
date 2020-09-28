
def map_steering(steering, left_pwm, neutral_pwm, right_pwm):
    return three_segment_map(steering,left_pwm, neutral_pwm, right_pwm)

def map_throttle(throttle, forward_pwm, neutral_pwm, reverse_pwm):
    return three_segment_map(throttle, forward_pwm, neutral_pwm, reverse_pwm)

def three_segment_map(val, min_map, mid_map, max_map):
    val = cap(val, -1, 1)
    if val == 0:
        return mid_map
    elif val < 0:
        return mid_map + (mid_map - min_map) * val
    elif val > 0:
        return mid_map + (max_map - mid_map) * val

def cap(val, min_val, max_val):
    if val < min_val : return min_val
    elif val > max_val: return max_val
    else: return val