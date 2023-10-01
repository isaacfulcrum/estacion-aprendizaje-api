def parse_scores(x):
    if isinstance(x, str):
        return [float(i) for i in x.strip('[]').split(',')]
    else:
        return []