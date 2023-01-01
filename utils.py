

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def map_to_range(x, start_range, end_range):

    v_min1, v_max1 = start_range
    v_min2, v_max2 = end_range
    y = v_min2 + (x-v_min1)*(v_max2-v_min2)/(v_max1-v_min1)

    return y


def clamp(x, range):
    v_min, v_max = range
    return min(max(x, v_min), v_max)