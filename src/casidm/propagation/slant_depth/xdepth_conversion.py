import numpy as np
from MCEq.geometry.density_profiles import CorsikaAtmosphere


class XdepthConversion:
    length_units = {"cm": 1, "m": 1e2, "km": 1e5}
    def __init__(self,* ,atmosphere=CorsikaAtmosphere("USStd", None)):
        self.atmosphere = atmosphere
        self.length_unit = self.length_units["cm"]
        # self._min_xdepth = 1e-6 is the best value
        # found for CorsikaAtmosphere("SouthPole", "December")
        self._min_xdepth = 1e-6
        self._calc_max()
        
        
    def _calc_max(self):
        self.max_height = self.convert_x2h(0)
        self.max_xdepth = self.convert_h2x(0)
            
    def set_theta(self, theta):
        self.atmosphere.set_theta(theta)
        self._calc_max()
        
    def get_theta(self):
        return self.atmosphere.theta_deg   

    def set_length_unit(self, unit="cm"):
        self.length_unit = self.length_units[unit]
        self._calc_max()

    def convert_x2h(self, x):
        """Convert xdepth to height"""
        if x < self._min_xdepth:
            x = self._min_xdepth
        return self.atmosphere.X2h(x) / self.length_unit

    def convert_h2x(self, h):
        """Convert height to depth"""
        if h >= self.max_height:
            return 0e0
        return self.atmosphere.h2X(h * self.length_unit)

    def get_max_xdepth(self):
        return self.max_xdepth

    def get_max_height(self):
        return self.max_height

    def get_length(self, x1, x2):
        """Return a length between atmospheric depth x1 and x2
        x1 < x2. Because X2h coverts only to height, zenith angle
        is also taken into account.

        Args:
            x1 (float): _description_
            x2 (float): _description_

        Returns:
            float: length in length units
        """
        return (self.convert_x2h(x1) - self.convert_x2h(x2)) / (
            np.cos(self.atmosphere.thrad)
        )

    def get_delta_xdepth(self, x1, length):
        """Returns delta x = x2 - x1 for the path starting at point
        x1 and run `length` in length units
        """
        h1 = self.convert_x2h(x1)
        h2 = h1 - length * np.cos(self.atmosphere.thrad)
        if h2 < 0:
            return None
        x2 = self.convert_h2x(h2)
        return x2 - x1


if __name__ == "__main__":
    xconv = XdepthConversion()
    xconv.set_length_unit("km")
    xconv.set_theta(30) 


    heights = [0, 5, 15, 20]
    
    
    # heights = [0.025232939039804742, 3.808506428696458, 14.096117926440979]
    for height in heights:
        print(f"h={height}, X={xconv.convert_h2x(height)} g/cm2")
    
    # print("------\n")
    # xdepths = [143, 647, 1033]
    # xdepths = [1, 10, 50, 100, 300, 400, 500, 600, 650, 700, 800, 900, 1000, 1033]  
    # xdepths = [900, 901, 910, 950, 1000, 1300, 1700, 2000, 2065] 
    # xdepths = [21, 86, 210, 492, 1033] 
    # xdepths = [1.23979442E+02, 5.60648882E+02, 8.94994432E+02] 
    # print(f"For zenith angle = {xconv.get_theta()}")   
    # for xdepth in xdepths:
    #     print(f"X={xdepth}, h={xconv.convert_x2h(xdepth)} km")    
    
    # print(f"X_max = {xconv.get_max_xdepth()}")
    
    
    # xconv.set_theta(0)
    # print(xconv.convert_h2x(6.475494041409048))
    # print(xconv.convert_x2h(452.7421530912818))
    
    # print(f"Back and forth conv = {xconv.convert_h2x(xconv.convert_x2h(0))}")
    # print(f"Max height = {xconv.get_max_height()}")
    # print(f"Max depth = {xconv.get_max_xdepth()}")
    # print(f"Xdepth(max_height) = {xconv.convert_h2x(xconv.get_max_height())}")
    # print(f"Height(xdepth = 500) = {xconv.convert_x2h(500)}")
    # print(f"Height(height = 5 km) = {xconv.convert_h2x(5)}")
    # print(xconv.get_delta_xdepth(500, 5.2))

# xconv = XdepthConversion()
# xconv.set_length_unit("km")
# xconv.set_theta(60)
# print(xconv.convert_x2h(0))
# print(xconv.get_max_height())
# print(xconv.get_delta_xdepth(0, 300))
# xconv.set_theta(60)
# # print(xconv.convert_x2h(100))
# print(f"x = 100 = {xconv.convert_x2h(100)} km")
# print(f"x = 700 = {xconv.convert_x2h(700)} km")

# print(f"h = 15km = {xconv.convert_h2x(xconv.convert_x2h(100))}")


# print(xconv.get_length(500, 700))
# print(xconv.get_delta_xdepth(500, xconv.get_length(500, 700)))
# print(xconv.get_delta_xdepth(300, 0.5))
# print(xconv.convert_x2h(300))
# print(xconv.convert_x2h(300 + 311))
