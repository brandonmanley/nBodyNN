using DifferentialEquations, DelimitedFiles

tspan = (0.0,10.0)  
dt = 0.00390625
tol = 1e-11

#     v indices: 
#     0: m1    1: p1x  2: p1y    3: p1z   4: p1vx   5: p1vy   6: p1vz
#     7: m2    8: p2x  9: p2y   10: p2z  11: p2vx  12: p2vy  13: p2vz
#     14: m3  15: p3x  16: p3y  17: p3z  18: p3vx  19: p3vy  20: p3vz
inA = readdlm("/nBodyData/inputs/indat_3_1.dat", ',', Float64, '\n')

iE = 1
m1, m2, m3 = inA[iE,1], inA[iE,8], inA[iE,15]
g = 1 # G not g 

#       x1,         x2,       x3,         y1,         y2,        y3,         vx1,     vx2,       vx3,            vy1,       vy2,      vy3
p0 = [inA[iE,2], inA[iE,9], inA[iE,16], inA[iE,3], inA[iE,10], inA[iE,17], inA[iE,5], inA[iE,12], inA[iE,19], inA[iE,6], inA[iE,13], inA[iE,20]]

function mass_system(du,u,p,t)
    x1,x2,x3,y1,y2,y3,dx1,dx2,dx3,dy1,dy2,dy3 = u

    du[1] = dx1
    du[2] = dx2
    du[3] = dx3
    du[4] = dy1
    du[5] = dy2
    du[6] = dy3
    
    du[7] = ((m2*(x2 - x1))/((x1 - x2)^2 + (y2 - y1)^2)^(3/2)) + ((m3*(x3 - x1))/((x1 - x3)^2 + (y1 - y3)^2)^(3/2))
    du[8] = ((m1*(x1 - x2))/((x1 - x2)^2 + (y1 - y2)^2)^(3/2)) + ((m3*(x3 - x2))/((x2 - x3)^2 + (y2 - y3)^2)^(3/2))
    du[9] = ((m1*(x1 - x3))/((x1 - x3)^2 + (y1 - y3)^2)^(3/2)) + ((m2*(x2 - x3))/((x2 - x3)^2 + (y2 - y3)^2)^(3/2))
    
    du[10] = ((m2*(y2 - y1))/((x1 - x2)^2 + (y2 - y1)^2)^(3/2)) + ((m3*(y3 - y1))/((x1 - x3)^2 + (y1 - y3)^2)^(3/2))
    du[11] = ((m1*(y1 - y2))/((x1 - x2)^2 + (y1 - y2)^2)^(3/2)) + ((m3*(y3 - y2))/((x2 - x3)^2 + (y2 - y3)^2)^(3/2))
    du[12] = ((m1*(y1 - y3))/((x1 - x3)^2 + (y1 - y3)^2)^(3/2)) + ((m2*(y2 - y3))/((x2 - x3)^2 + (y2 - y3)^2)^(3/2))
end
                               
prob = ODEProblem(mass_system,p0,tspan)
sol = solve(prob,reltol=tol,saveat=dt)

################
#### plot ######
################

# using Plots
# plotly()
# display(plot(sol, vars=(1,4)))
# display(plot!(sol, vars=(2,5)))
# display(plot!(sol, vars=(3,6)))