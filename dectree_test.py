
from numba import jit, jitclass, float64


@jit(nopython=True)
def _Radiance_LOW(x):
    # Radiance.LOW: ramp_down(x1=0, x2=50)
    if x < 0.0:
        return 1.0
    if x > 50.0:
        return 0.0
    return 1.0 - (x - 0.0) / (50.0 - 0.0)


@jit(nopython=True)
def _Radiance_MIDDLE(x):
    # Radiance.MIDDLE: triangle(x1=30, x2=50, x3=100)
    if x < 30.0:
        return 0.0
    if x > 100.0:
        return 0.0
    if x < 50.0:
        return (x - 30.0) / (50.0 - 30.0)
    return 1.0 - (x - 50.0) / (100.0 - 50.0)


@jit(nopython=True)
def _Radiance_HIGH(x):
    # Radiance.HIGH: ramp_up(x1=50, x2=120)
    if x < 50.0:
        return 0.0
    if x > 120.0:
        return 1.0
    return (x - 50.0) / (120.0 - 50.0)


@jit(nopython=True)
def _Glint_LOW(x):
    # Glint.LOW: ramp_down()
    if x < 0.0:
        return 1.0
    if x > 0.5:
        return 0.0
    return 1.0 - (x - 0.0) / (0.5 - 0.0)


@jit(nopython=True)
def _Glint_HIGH(x):
    # Glint.HIGH: ramp_up()
    if x < 0.5:
        return 0.0
    if x > 1.0:
        return 1.0
    return (x - 0.5) / (1.0 - 0.5)


@jit(nopython=True)
def _Cloudy_True(x):
    # Cloudy.True: true()
    return 1.0


@jit(nopython=True)
def _Cloudy_False(x):
    # Cloudy.False: false()
    return 0.0


@jit(nopython=True)
def _Certain_HIGH(x):
    # Certain.HIGH: true()
    return 1.0


@jit(nopython=True)
def _Certain_LOW(x):
    # Certain.LOW: false()
    return 0.0


_InputSpec = [
    ("glint", float64),
    ("radiance", float64),
]


@jitclass(_InputSpec)
class Input:
    def __init__(self):
        self.glint = 0.0
        self.radiance = 0.0


_OutputSpec = [
    ("cloudy", float64),
    ("certain", float64),
]


@jitclass(_OutputSpec)
class Output:
    def __init__(self):
        self.cloudy = 0.0
        self.certain = 0.0


@jit(nopython=True)
def apply_rules(input, output):
    t0 = 1.0
    #    if radiance == HIGH or radiance == MIDDLE:
    t1 = min(t0, max(_Radiance_HIGH(input.radiance), _Radiance_MIDDLE(input.radiance)))
    #        if glint == LOW:
    t2 = min(t1, _Glint_LOW(input.glint))
    #            cloudy: True
    output.cloudy = t2
    #            certain: HIGH
    output.certain = t2
    #        else:
    t2 = 1.0 - (t2)
    #            if glint == HIGH:
    t3 = min(t2, _Glint_HIGH(input.glint))
    #                certain: LOW
    output.certain = max(output.certain, 1.0 - (t3))
    #    else:
    t1 = 1.0 - (t1)
    #        cloudy: False
    #output.cloudy = max(output.cloudy, 1.0 - (t1))
    #        certain: HIGH
    output.certain = max(output.certain, t1)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    fig, axs = plt.subplots(1, 2, sharey=True)
    linewidth = 2

    ax = axs[0]
    ax.set_title('Radiance')
    ax.set_ylabel('Truth')
    x = np.arange(-10.0, 130.0, 1.0)
    y = np.vectorize(_Radiance_LOW)(x)
    ax.plot(x, y, label='LOW', linewidth=linewidth)
    y = np.vectorize(_Radiance_MIDDLE)(x)
    ax.plot(x, y, label='MIDDLE', linewidth=linewidth)
    y = np.vectorize(_Radiance_HIGH)(x)
    ax.plot(x, y, label='HIGH', linewidth=linewidth)
    ax.legend()

    ax = axs[1]
    ax.set_title('Glint')
    ax.set_ylabel('Truth')
    x = np.arange(-0.1, 1.1, 0.01)
    y = np.vectorize(_Glint_LOW)(x)
    ax.plot(x, y, label='LOW', linewidth=linewidth)
    y = np.vectorize(_Glint_HIGH)(x)
    ax.plot(x, y, label='HIGH', linewidth=linewidth)
    ax.legend()

    plt.show()

    ##########################################################

    # Make 3D data.
    x = np.arange(-10., 130., 1.)
    y = np.arange(-0.1, 1.1, 0.01)
    x, y = np.meshgrid(x, y)
    z1 = np.zeros(shape=x.shape, dtype=np.float64)
    z2 = np.zeros(shape=x.shape, dtype=np.float64)
    input = Input()
    output = Output()
    for j in range(x.shape[-2]):
        for i in range(x.shape[-1]):
            input.radiance = x[j, i]
            input.glint = y[j, i]
            apply_rules(input, output)
            z1[j, i] = output.cloudy
            z2[j, i] = output.certain

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.set_zlim(0.0, 1.0)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z2, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.set_zlim(0.0, 1.0)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
