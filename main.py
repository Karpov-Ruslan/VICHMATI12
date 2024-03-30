import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sopt



class Methods:
    def __init__(self, function, start_condition, start = 0.0, order=2):
        self.function = function
        self.start_condition = start_condition
        self.order = order
        self.start = start

    def initRet(self, end, step):
        ret = np.array([[0.0 for i in range(self.order)] for i in range(int((end - self.start) / step) + 1)])
        ret[0] = self.start_condition
        return ret


class ExplicitMethods(Methods):
    """
    The class provides explicit methods
    for solving an ordinary differential equation

    The instance of this class contains function
    that need to specify differential equation:
    dy/dt = func(t, y), where y is a vector
    """

    def MidPointMethod(self, end=100.0, step=0.01):
        """
        This is second-order Runge-Kutta method
        """

        # Init
        ret = self.initRet(end, step)

        # Main part
        for i in range(1, ret.shape[0]):
            ret[i] = ret[i - 1] + step * self.function(self.start + (i - 0.5) * step, ret[i - 1] + 0.5 * step * self.function(self.start + (i - 1) * step, ret[i - 1]))

        # Return
        return ret

    def AdamsMethod(self, end=100.0, step=0.01):
        # Init
        ret = self.initRet(end, step)

        # I fill the necessary initial values by the first-order Adams method (Yeas, I know that's bad)
        ret[1] = ret[0] + step * self.function(self.start, ret[0])

        # Main part
        for i in range(2, ret.shape[0]):
            ret[i] = ret[i - 1] + 0.5 * step * (3 * self.function(self.start + (i - 1) * step, ret[i - 1]) - self.function(self.start + (i - 2) * step, ret[i - 2]))

        # Return
        return ret

    def GearMethod(self, end=100.0, step=0.005):
        # Init
        ret = self.initRet(end, step)

        # I fill the necessary initial values by the first-order Adams method (Yeas, I know that's bad)
        ret[1] = ret[0] + step * self.function(self.start, ret[0])

        # Main part
        for i in range(2, ret.shape[0]):
            def func(y):
                # return (-y - (1/3)*ret[i - 1] + (4/3)*ret[i - 2] + (2/3)*step*self.function(start + i*step, y))
                return (1.5*y - 2.0*ret[i - 1] + 0.5*ret[i - 2] - step*self.function(self.start + i*step, y))

            result = sopt.root(func, ret[i - 1])
            if not result.success:
                raise ArithmeticError("Cannot solve it, where: " + str(self.start + step*i))
            ret[i] = result.x

        # Return
        return ret

class ImplicitMethods(Methods):
    """
    The class provides implicit methods
    for solving an ordinary differential equation

    The instance of this class contains function
    that need to specify differential equation:
    dy/dt = func(t, y), where y is a vector
    """

    def EilerMethod(self, end = 100.0, step = 0.01):
        # Init
        ret = self.initRet(end, step)

        # Main part
        for i in range(1, ret.shape[0]):
            approximate_ret = ret[i - 1] + step*self.function(self.start + (i - 1)*step, ret[i - 1])
            ret[i] = ret[i - 1] + (step/2.0)*(self.function(self.start + (i - 1)*step, ret[i - 1]) + self.function(self.start + i*step, approximate_ret))

        # Return
        return ret

    def RozenbroekMethod(self, end = 100.0, step = 0.005):
        # Init
        ret = self.initRet(end, step)

        # Main part
        for i in range(1, ret.shape[0]):
            def self_func_without_t(y):
                return self.function(self.start + (i - 1)*step, y)
            ret[i] = ret[i - 1] + step*np.real(sopt.newton(lambda w: np.dot((np.eye(ret.shape[1]) - (0.5 + 0.5j)*step*sopt.approx_fprime(ret[i - 1], self_func_without_t)), w) - self.function(self.start + (i - 0.5)*step, ret[i - 1]), np.array([0, 0])))

        # Return
        return ret

    def AdamsMethod(self, end = 100.0, step = 0.005):
        # Init
        ret = self.initRet(end, step)

        # Main part
        for i in range(1, ret.shape[0]):
            def func(y):
                # return (-y - (1/3)*ret[i - 1] + (4/3)*ret[i - 2] + (2/3)*step*self.function(start + i*step, y))
                return (-y + ret[i - 1] + 0.5*step*(self.function(self.start + i*step, y) + self.function(self.start + (i - 1)*step, ret[i - 1])))

            result = sopt.root(func, ret[i - 1])
            if not result.success:
                raise ArithmeticError("Cannot solve it, where: " + str(self.start + step*i))
            ret[i] = result.x

        # Return
        return ret


if __name__ == "__main__":
    # Van-der-Pole equation
    def give_me_func1(e):
        def func(t, y):
            return np.array([y[1], e*(1 - y[0]**2)*y[1] - y[0]])

        return func

    #Rayleigh equation
    def give_me_func2(m):
        def func(t, y):
            return np.array(y[1], m*(1 - y[1]**2)*y[1] - y[0])

        return func


    def paint_it(ret, title):
        ret = ret.transpose()

        plt.plot(ret[0], ret[1], 'ob')
        plt.ylabel(r"y'")
        plt.xlabel(r"y")
        plt.title(title)
        plt.show()

    # Check explicit methods
    practice = ExplicitMethods(give_me_func1(10), np.array([2.0, 0.0]))

    ret = practice.MidPointMethod()
    paint_it(ret, "MidPoint Method")

    ret = practice.AdamsMethod()
    paint_it(ret, "Explicit Adams Method")

    ret = practice.GearMethod()
    paint_it(ret, "Gear Method")


    # Check implicit methods
    practice.__class__ = ImplicitMethods

    ret = practice.EilerMethod()
    paint_it(ret, "Eiler Method")

    ret = practice.RozenbroekMethod()
    paint_it(ret, "Rozenbroek Method")

    ret = practice.AdamsMethod()
    paint_it(ret, "Implicit Adams Method")



    # Check explicit methods
    practice = ExplicitMethods(give_me_func2(1000), np.array([0.0, 0.001]))

    ret = practice.MidPointMethod()
    paint_it(ret, "MidPoint Method")

    ret = practice.AdamsMethod()
    paint_it(ret, "Explicit Adams Method")

    ret = practice.GearMethod()
    paint_it(ret, "Gear Method")

    # Check implicit methods
    practice.__class__ = ImplicitMethods

    ret = practice.EilerMethod()
    paint_it(ret, "Eiler Method")

    ret = practice.RozenbroekMethod(step=0.01)
    paint_it(ret, "Rozenbroek Method")

    ret = practice.AdamsMethod(step=0.01)
    paint_it(ret, "Implicit Adams Method")