import protonets.data

def load(opt, splits):
    if opt['data.dataset'] == 'landmark':
        ds = protonets.data.landmark.load(opt, splits)
    else:
        raise ValueError("Unknown dataset: {:s}".format(opt['data.dataset']))

    return ds
