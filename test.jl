using DifferentialEquations
using Plots

l = 1.0                    
m = 1.0                         
g = 9.81    
ω = 1.0                     

function mass_system!(ddu,du,u,p,t)
    # a(t) = (1/m) w^2 x 
    ddu[1] = (1/m)*(ω^2)*u[1]
    # du[1] = u[2]                                # θ'(t) = ω(t)
    # du[2] = -3g/(2l)*sin(u[1]) + 3/(m*l^2)*p(t) # ω'(t) = -3g/(2l) sin θ(t) + 3/(ml^2)M(t)
end

# θ₀ = 0.01                         
# ω₀ = 0.0  
v0 = 0.0                      # initial velocity [m/s]
u0 = 1.0                    # initial state vector
tspan = (0.0,10.0)                  # time interval

M = t->0.1sin(t)                    # external torque [Nm]

# prob = ODEProblem(pendulum!,u₀,tspan,M)
prob = SecondOrderODEProblem{isinplace}(mass_system!,v0,u0,tspan,callback=CallbackSet())
sol = solve(prob)

plotly()
display(plot(sol,linewidth=2,xaxis="t"))


