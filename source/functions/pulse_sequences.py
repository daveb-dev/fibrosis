''' Pulse sequence definitions for the Bloch solver '''


def PGSE(t, dt):
    ''' PGSE Pulsed Gradient Spin Echo (Stejskal & Tanner) '''
    if t < dt[0]:
        return 1
    elif t > dt[1] and t < dt[0]+dt[1]:
        return -1
    else:
        return 0
