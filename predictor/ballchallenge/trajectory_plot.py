import matplotlib.pyplot as plt
import math
import numpy

v=20 #velocity at the start m/s

α=math.radians(45) #angle of ball throw

y0=10 #Elevation of startingpoint

g=9.81 #gravity

# time of flight por to pol
a=g
b=-2*v*math.sin(α)
c=-2*y0

coeff=list([a,b,c]) # coefficient array

# find roots
t1,t2=numpy.roots(coeff)
print(f"t1= {t1} and t2= {t2}")

# max height from throwing point
h1=v**2*(math.sin(α))**2/(2*g)

# total
h_max=h1+y0
print(f"h_max= {round(h_max,3)} m")


R=v*math.cos(α)*max(t1,t2) # range
#max(t1,t2) returns positive value

print(f"R= {round(R,3)} m")#

plt.plot([0,R],[0,-y0],linewidth=5) # plots inclined surface

plt.plot([0,R],[0,0],'k',linewidth=1) # plots y=0 line

x=numpy.linspace(0,R,50) # array of x

y=x*math.tan(α)-(1/2)*(g*x**2)/(v**2*(math.cos(α))**2 ) # evaluate y based on x

# Plotting projectile
plt.plot(x,y,'r-',linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.show()