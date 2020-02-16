if length(ARGS) < 1
    println("Usage: filenum")
    exit()
end 

batchNum = 4
fileNum = ARGS[1]
# try 
#     fileNum = ARGS[1]
# catch 
#     println("Enter valid filenum")
#     exit()
# end

inputString = "/nBodyData/inputs/indat_$(batchNum)_$(fileNum).dat"
if !isfile(inputString)
    @warn "Could not find input: $(inputString)"
    exit()
end 

outputString = "/nBodyData/julSim/julia_batch$(batchNum)_$(fileNum).csv"
try
    run(`rm $(outputString)`)
catch
    @warn "$(outputString) not found"
end

println("Using input $(inputString)")

using DifferentialEquations, DelimitedFiles, Plots, DataFrames, CSV, ProgressMeter

tspan = (0.0,10.0)  
dt = 0.00390625
tol = 1e-11

#     v indices: 
#     0: m1    1: p1x  2: p1y    3: p1z   4: p1vx   5: p1vy   6: p1vz
#     7: m2    8: p2x  9: p2y   10: p2z  11: p2vx  12: p2vy  13: p2vz
#     14: m3  15: p3x  16: p3y  17: p3z  18: p3vx  19: p3vy  20: p3vz
inA = readdlm(inputString, ',', Float64, '\n')
numLines = countlines(inputString)

@showprogress 1 "Working..." for i = 1:numLines

    iE = i 
    globalID = 10000*iE
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

    tEnd = []
    tEnd = sol.t
    deleteat!(tEnd,1)

    eventID,m1a,m2a,m3a = [],[],[],[]
    x1,x2,x3,y1,y2,y3 = [],[],[],[],[],[]
    x1f,x2f,x3f,y1f,y2f,y3f = [],[],[],[],[],[]

    for (i,step) in enumerate(sol)
        if i == 1
            continue
        end
        
        append!(eventID, globalID+(i-1))
        append!(m1a, m1)
        append!(m2a, m2)
        append!(m3a, m3)
        
        append!(x1, p0[1])
        append!(x2, p0[2])
        append!(x3, p0[3])
        append!(y1, p0[4])
        append!(y2, p0[5])
        append!(y3, p0[6])
        
        append!(x1f, step[i][1])
        append!(x2f, step[i][2])
        append!(x3f, step[i][3])
        append!(y1f, step[i][4])
        append!(y2f, step[i][5])
        append!(y3f, step[i][6])
    end

    df = DataFrame(eventID=eventID, m1=m1a, m2=m2a, m3=m3a,
                    x1=x1, x2=x2, x3=x3, y1=y1, y2=y2, y3=y3, tEnd=tEnd, 
                    x1tEnd=x1f, x2tEnd=x2f, x3tEnd=x3f, y1tEnd=y1f, y2tEnd=y2f, y3tEnd=y3f)
    CSV.write(outputString, df; append=true)

end #end event loop

################
#### plot ######
################

# using Plots
# plotly()
# display(plot(sol, vars=(1,4)))
# display(plot!(sol, vars=(2,5)))
# display(plot!(sol, vars=(3,6)))