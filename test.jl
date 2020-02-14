using DifferentialEquations
using Plots

m = 1.0                          
ω = 1.0                     

function mass_system(du,u,p,t)
    # a(t) = (1/m) w^2 x 
    (1/m)*(ω^2)*u[1]
end

v0 = 0.0                     
u0 = 1.0                  
tspan = (0.0,10.0)               

prob = SecondOrderODEProblem(mass_system,v0,u0,tspan)
sol = solve(prob)

plotly()
display(plot(sol,linewidth=2,xaxis="t"))


