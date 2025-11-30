CONF_THRESHOLD = 0.45

def is_valid_person(cls, conf, x1, y1, x2, y2):
    return cls == 0 and conf >= CONF_THRESHOLD
