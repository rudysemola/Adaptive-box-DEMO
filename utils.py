

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def map_to_range(x, min_in, max_in, min_out, max_out):
    y = min_out + (x-min_in)*(max_out-min_out)/(max_in-min_in)
    return y