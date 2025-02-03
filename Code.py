#%% Import
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import os
import sys
file_path = 'Output.txt'
if os.path.exists(file_path):
  os.remove(file_path)
if os.path.exists('Output.xlsx'):
  os.remove('Output.xlsx')
if os.path.exists('OutputPS1.xlsx'):
  os.remove('OutputPS1.xlsx')
if os.path.exists('OutputPS2.xlsx'):
  os.remove('OutputPS2.xlsx')
orig_stdout = sys.stdout
sys.stdout = open(file_path, 'w')


nC = 4 # Number of columns in table
nR = 4 # Number of rows in table
nRun = 30 # Number of runs per cell
Base = 5 # Base number for increase in rows and columns
TestCase = 3 
# TestCase = 1 (zhat >= zmax)
# TestCase = 2 (zmin <= zhat <= zmax)
# TestCase = 3 (zhat <= zmin)
TimeLimit = 3600 # Time limit of solving each instance
SeedNumber = 100 # Seed Number

SpecificColumn = 15 # Generate instances with columns up to SpecificColumn
SpecificRow = 15 # Generate instances with rows up to SpecificColumn
SpecificInstance = 1 # Generate instances with number for instance from SpecificInstance
PS1Active = 1 # Whether CCA algorithm is active or not
PS2Active = 1 # Whether PTA algorithm is active or not

options = {
  
}
env = gp.Env(params=options)


#%% FuncGenerate
def FuncGenerate(m,n):
  """
  Generate test instances.
  
  Parameters:
  m (int): Number of rows.
  n (int): Number of columns.
  """
  global I,J,IA,eps,xU,lC,uC,lD,uD,alpha,beta,A,b,BC,BD,fC,fD,chat,dhat,xhat,epsnon,chatnon,dhatnon
  # Sets
  I = range(0, m)
  J = range(0, n)
  IA = range(0, m + n)
  #Parameters
  eps = 0.01
  epsnon = 0.1
  xU = 100
  lC = -100
  uC = 100
  lD = 0
  uD = 100
  alpha = np.random.uniform(0,100)
  beta = np.random.uniform(eps,100)
  A = np.random.uniform(-50,50,(m,n))
  b = np.random.uniform(-50,50,m)
  BC = np.random.uniform(-50,50,(m,n))
  fC = np.random.uniform(-50,50,m)
  BD = np.random.uniform(-50,50,(m,n))
  fD = np.random.uniform(-50,50,m)
  chatnon = np.random.uniform(lC,uC,n)
  dhatnon = np.random.uniform(lD,uD,n)
  if TestCase == 1:
    chat = np.ones(n) * (uC + epsnon * abs(uC))
    dhat = np.ones(n) * (lD)
  elif TestCase == 2:
    chat = chatnon
    dhat = dhatnon
  elif TestCase == 3:
    chat = np.ones(n) * (lC - epsnon * abs(lC))
    dhat = np.ones(n) * (lD)
  xhat = np.random.uniform(0,xU,n)
  for i in I:
    if sum(BC[i][j]*chatnon[j] for j in J) >= fC[i]:
      BC[i] = -BC[i]
      fC[i] = -fC[i]
    if sum(BD[i][j]*dhatnon[j] for j in J) >= fD[i]:
      BD[i] = -BD[i]
      fD[i] = -fD[i]
    if sum(A[i][j]*xhat[j] for j in J) <= b[i]:
      A[i] = -A[i]
      b[i] = -b[i]
  IdenA = -np.identity(n)
  A = np.concatenate((A,IdenA), axis = 0)
  Idenb = np.ones(n) * (-xU)
  b = np.concatenate((b,Idenb), axis = 0)

#%% FuncLFP
def FuncLFP(cdata,ddata):
  """
  Linear fractional programming (LFP) problem.
  
  Parameters:
  cdata: c vector.
  ddata: d vector.
  """
  global LFP,y,t
  # Model
  LFP = gp.Model(env=env,name='LFP')
  # Decision Variables
  y = LFP.addVars(J, name="y")
  t = LFP.addVar(name="t")
  # Constraints
  for i in IA:
    LFP.addConstr(sum(A[i][j]*y[j] for j in J) >= b[i]*t)
  LFP.addConstr(sum(ddata[j]*y[j] for j in J) + beta*t == 1)
  # Objective
  LFP.setObjective(sum(cdata[j]*y[j] for j in J) + alpha*t, GRB.MINIMIZE)
  LFP.params.LogToConsole = 0
  LFP.params.timeLimit = TimeLimit
  LFP.optimize()
  
#%% FuncCCM_SD
def FuncCCM_SD(zdata):
  """
  Charnes-Cooper transformation model with strong duality reformulation.
  
  Parameters:
  zdata: zhat.
  """
  global CCM,vplus,vminus,c,d,y,t,u
  # Model
  CCM = gp.Model(env=env,name='CCM')
  # Decision Variables
  vplus = CCM.addVar(name="vplus")
  vminus = CCM.addVar(name="vminus")
  c = CCM.addVars(J, name="c", lb=lC, ub=uC)
  d = CCM.addVars(J, name="d", lb=lD, ub=uD)
  y = CCM.addVars(J, name="y")
  t = CCM.addVar(name="t")
  u = CCM.addVars(IA, name="u")
  # Constraints
  for i in I:
    CCM.addConstr(sum(BC[i][j]*c[j] for j in J) <= fC[i])
    CCM.addConstr(sum(BD[i][j]*d[j] for j in J) <= fD[i])
  for i in IA:
    CCM.addConstr(sum(A[i][j]*y[j] for j in J) >= b[i]*t)
  for j in J:   
    CCM.addConstr(sum(A[i][j]*u[i] for i in IA) + zdata*d[j] <= c[j])
  CCM.addConstr(-sum(b[i]*u[i] for i in IA) + zdata*beta == alpha)
  CCM.addConstr(sum(d[j]*y[j] for j in J) + beta*t == 1)
  CCM.addConstr(alpha*t + sum(c[j]*y[j] for j in J) == zdata)
  # Objective
  CCM.setObjective(0, GRB.MINIMIZE)
  #CCM.params.LogToConsole = 0
  CCM.params.nonConvex = 2
  CCM.params.timeLimit = TimeLimit
  CCM.optimize()
  
#%% FuncPRM_CS
def FuncPRM_CS(zdata):
  """
  Parametric transformation model with complementary slackness conditions reformulation.
  
  Parameters:
  zdata: zhat.
  """
  global PRM,c,d,x,u,r1,r2
  # Model
  PRM = gp.Model(env=env,name='PRM')
  # Decision Variables
  c = PRM.addVars(J, name="c", lb=lC, ub=uC)
  d = PRM.addVars(J, name="d", lb=lD, ub=uD)
  x = PRM.addVars(J, name="x")
  u = PRM.addVars(IA, name="u")
  r1 = PRM.addVars(J, name="r1")
  r2 = PRM.addVars(IA, name="r2")
  # Constraints
  PRM.addConstr(sum(b[i]*u[i] for i in IA) + alpha - zdata*beta == 0)
  for i in I:
    PRM.addConstr(sum(BC[i][j]*c[j] for j in J) <= fC[i])
    PRM.addConstr(sum(BD[i][j]*d[j] for j in J) <= fD[i])
  for i in IA:
    PRM.addConstr(sum(A[i][j]*x[j] for j in J) == b[i] + r2[i])
    PRM.addSOS(GRB.SOS_TYPE1,[r2[i],u[i]])
  for j in J:
    PRM.addConstr(sum(A[i][j]*u[i] for i in IA) + r1[j] == c[j] - zdata*d[j])
    PRM.addSOS(GRB.SOS_TYPE1,[r1[j],x[j]])
  # Objective
  PRM.setObjective(0, GRB.MAXIMIZE)
  #PRM.params.LogToConsole = 0
  PRM.params.timeLimit = TimeLimit
  PRM.optimize()
  
#%% FuncPRM_SD
def FuncPRM_SD(zdata):
  """
  Parametric transformation model with strong duality reformulation.
  
  Parameters:
  zdata: zhat.
  """
  global PRM,c,d,x,u
  # Model
  PRM = gp.Model(env=env,name='PRM')
  # Decision Variables
  c = PRM.addVars(J, name="c", lb=lC, ub=uC)
  d = PRM.addVars(J, name="d", lb=lD, ub=uD)
  x = PRM.addVars(J, name="x")
  u = PRM.addVars(IA, name="u")
  # Constraints
  PRM.addConstr(sum(b[i]*u[i] for i in IA) + alpha - zdata*beta == 0)
  for i in I:
    PRM.addConstr(sum(BC[i][j]*c[j] for j in J) <= fC[i])
    PRM.addConstr(sum(BD[i][j]*d[j] for j in J) <= fD[i])
  for i in IA:
    PRM.addConstr(sum(A[i][j]*x[j] for j in J) >= b[i])
  for j in J:
    PRM.addConstr(sum(A[i][j]*u[i] for i in IA) <= c[j] - zdata*d[j])
  PRM.addConstr(sum(b[i]*u[i] for i in IA) == sum(c[j]*x[j] for j in J) - zdata*sum(d[j]*x[j] for j in J))
  # Objective
  PRM.setObjective(0, GRB.MAXIMIZE)
  #PRM.params.LogToConsole = 0
  PRM.params.nonConvex = 2
  PRM.params.timeLimit = TimeLimit
  PRM.optimize()
  
#%% FuncCCM_MIN
def FuncCCM_MIN(zdata):
  """
  Charnes-Cooper transformation model for IOV-MIN.
  
  Parameters:
  zdata: zhat.
  """
  global CCM,y,t,c,d
  # Model
  CCM = gp.Model(env=env,name='CCM')
  # Decision Variables
  y = CCM.addVars(J, name="y")
  t = CCM.addVar(name="t")
  c = CCM.addVars(J, name="c", lb=lC, ub=uC)
  d = CCM.addVars(J, name="d", lb=lD, ub=uD)
  # Constraints
  for i in I:
    CCM.addConstr(sum(BC[i][j]*c[j] for j in J) <= fC[i])
    CCM.addConstr(sum(BD[i][j]*d[j] for j in J) <= fD[i])
  for i in IA:
    CCM.addConstr(sum(A[i][j]*y[j] for j in J) >= b[i]*t)
  CCM.addConstr(sum(d[j]*y[j] for j in J) + beta*t == 1)
  CCM.addConstr(sum(c[j]*y[j] for j in J) + alpha*t >= zdata)
  # Objectives
  CCM.setObjective(sum(c[j]*y[j] for j in J) + alpha*t, GRB.MINIMIZE)
  #CCM.params.LogToConsole = 0
  CCM.params.nonConvex = 2
  CCM.params.timeLimit = TimeLimit
  CCM.optimize()
  
#%% FuncCCM_MAX
def FuncCCM_MAX(zdata):
  """
  Charnes-Cooper transformation model for IOV-MAX.
  
  Parameters:
  zdata: zhat.
  """
  global CCM,c,d,u,f
  # Model
  CCM = gp.Model(env=env,name='CCM')
  # Decision Variables
  c = CCM.addVars(J, name="c", lb=lC, ub=uC)
  d = CCM.addVars(J, name="d", lb=lD, ub=uD)
  u = CCM.addVars(IA, name="u")
  f = CCM.addVar(name="f", lb=-GRB.INFINITY)
  # Constraints
  for i in I:
    CCM.addConstr(sum(BC[i][j]*c[j] for j in J) <= fC[i])
    CCM.addConstr(sum(BD[i][j]*d[j] for j in J) <= fD[i])
  for j in J:
    CCM.addConstr(sum(A[i][j]*u[i] for i in IA) + f*d[j] <= c[j])
  CCM.addConstr(-sum(b[i]*u[i] for i in IA) + f*beta == alpha)
  CCM.addConstr(f <= zdata)
  # Objective
  CCM.setObjective(f, GRB.MAXIMIZE)
  #CCM.params.LogToConsole = 0
  CCM.params.nonConvex = 2 
  CCM.params.timeLimit = TimeLimit
  CCM.optimize()

#%% FuncCCM_PF
def FuncCCM_PF(zdata):
  """
  Charnes-Cooper transformation model for problem feasibility F_p.
  
  Parameters:
  zdata: zhat.
  """
  global CCM,y,t,c,d
  # Model
  CCM = gp.Model(env=env,name='CCM')
  # Decision Variables
  y = CCM.addVars(J, name="y")
  t = CCM.addVar(name="t")
  c = CCM.addVars(J, name="c", lb=lC, ub=uC)
  d = CCM.addVars(J, name="d", lb=lD, ub=uD)
  # Constraints
  for i in I:
    CCM.addConstr(sum(BC[i][j]*c[j] for j in J) <= fC[i])
    CCM.addConstr(sum(BD[i][j]*d[j] for j in J) <= fD[i])
  for i in IA:
    CCM.addConstr(sum(A[i][j]*y[j] for j in J) >= b[i]*t)
  CCM.addConstr(sum(d[j]*y[j] for j in J) + beta*t == 1)
  CCM.addConstr(sum(c[j]*y[j] for j in J) + alpha*t <= zdata)
  # Objectives
  CCM.setObjective(0, GRB.MINIMIZE)
  CCM.params.LogToConsole = 0
  CCM.params.nonConvex = 2
  CCM.params.timeLimit = TimeLimit
  CCM.optimize()
  
#%% FuncCCM_DF
def FuncCCM_DF(zdata):
  """
  Charnes-Cooper transformation model for problem feasibility F_d.
  
  Parameters:
  zdata: zhat.
  """
  global CCM,c,d,u
  # Model
  CCM = gp.Model(env=env,name='CCM')
  # Decision Variables
  c = CCM.addVars(J, name="c", lb=lC, ub=uC)
  d = CCM.addVars(J, name="d", lb=lD, ub=uD)
  u = CCM.addVars(IA, name="u")
  # Constraints
  for i in I:
    CCM.addConstr(sum(BC[i][j]*c[j] for j in J) <= fC[i])
    CCM.addConstr(sum(BD[i][j]*d[j] for j in J) <= fD[i])
  for j in J:
    CCM.addConstr(sum(A[i][j]*u[i] for i in IA) + zdata*d[j] <= c[j])
  CCM.addConstr(-sum(b[i]*u[i] for i in IA) + zdata*beta <= alpha)
  # Objective
  CCM.setObjective(0, GRB.MAXIMIZE)
  CCM.params.LogToConsole = 0
  CCM.params.timeLimit = TimeLimit
  CCM.optimize()
  
#%% FuncCCM_CS
def FuncCCM_CS(zdata):
  """
  Charnes-Cooper transformation model with complementary slackness conditions reformulation.
  
  Parameters:
  zdata: zhat.
  """
  global CCM,vplus,vminus,c,d,y,t,u,r1,r2
  # Model
  CCM = gp.Model(env=env,name='CCM')
  # Decision Variables
  c = CCM.addVars(J, name="c", lb=lC, ub=uC)
  d = CCM.addVars(J, name="d", lb=lD, ub=uD)
  y = CCM.addVars(J, name="y")
  t = CCM.addVar(name="t")
  u = CCM.addVars(IA, name="u")
  r1 = CCM.addVars(IA, name="r1")
  r2 = CCM.addVars(J, name="r2")
  # Constraints
  for i in I:
    CCM.addConstr(sum(BC[i][j]*c[j] for j in J) <= fC[i])
    CCM.addConstr(sum(BD[i][j]*d[j] for j in J) <= fD[i])
  for i in IA:
    CCM.addConstr(sum(A[i][j]*y[j] for j in J) == b[i]*t + r1[i])
    CCM.addSOS(GRB.SOS_TYPE1,[r1[i],u[i]])
  for j in J:   
    CCM.addConstr(sum(A[i][j]*u[i] for i in IA) + zdata*d[j] + r2[j] == c[j])
    CCM.addSOS(GRB.SOS_TYPE1,[r2[j],y[j]])
  CCM.addConstr(-sum(b[i]*u[i] for i in IA) + zdata*beta == alpha)
  CCM.addConstr(sum(d[j]*y[j] for j in J) + beta*t == 1)
  # Objective
  CCM.setObjective(0, GRB.MINIMIZE)
  #CCM.params.LogToConsole = 0
  CCM.params.nonConvex = 2
  CCM.params.timeLimit = TimeLimit
  CCM.optimize()

#%% FuncPRM_MIN
def FuncPRM_MIN(zdata):
  """
  Parametric transformation model IOV-MIN.
  
  Parameters:
  zdata: zhat.
  """
  global PRM,c,d,x,u,lamb,r1,r2
  # Model
  PRM = gp.Model(env=env,name='PRM')
  # Decision Variables
  c = PRM.addVars(J, name="c", lb=lC, ub=uC)
  d = PRM.addVars(J, name="d", lb=lD, ub=uD)
  x = PRM.addVars(J, name="x")
  u = PRM.addVars(IA, name="u")
  lamb = PRM.addVar(name="lamb", lb=-GRB.INFINITY)
  r1 = PRM.addVars(J, name="r1")
  r2 = PRM.addVars(IA, name="r2")
  # Constraints
  PRM.addConstr(sum(b[i]*u[i] for i in IA) + alpha - lamb*beta == 0)
  for i in I:
    PRM.addConstr(sum(BC[i][j]*c[j] for j in J) <= fC[i])
    PRM.addConstr(sum(BD[i][j]*d[j] for j in J) <= fD[i])
  for i in IA:
    PRM.addConstr(sum(A[i][j]*x[j] for j in J) == b[i] + r2[i])
    PRM.addSOS(GRB.SOS_TYPE1,[r2[i],u[i]])
  for j in J:
    PRM.addConstr(sum(A[i][j]*u[i] for i in IA) + r1[j] == c[j] - lamb*d[j])
    PRM.addSOS(GRB.SOS_TYPE1,[r1[j],x[j]])
  PRM.addConstr(lamb >= zdata)
  # Objective
  PRM.setObjective(lamb, GRB.MINIMIZE)
  #PRM.params.LogToConsole = 0
  PRM.params.nonConvex = 2
  PRM.params.timeLimit = TimeLimit
  PRM.optimize()

#%% FuncPRM_MAX
def FuncPRM_MAX(zdata):
  """
  Parametric transformation model IOV-MAX.
  
  Parameters:
  zdata: zhat.
  """
  global PRM,c,d,x,u,lamb,r1,r2
  # Model
  PRM = gp.Model(env=env,name='PRM')
  # Decision Variables
  c = PRM.addVars(J, name="c", lb=lC, ub=uC)
  d = PRM.addVars(J, name="d", lb=lD, ub=uD)
  x = PRM.addVars(J, name="x")
  u = PRM.addVars(IA, name="u")
  lamb = PRM.addVar(name="lamb", lb=-GRB.INFINITY)
  r1 = PRM.addVars(J, name="r1")
  r2 = PRM.addVars(IA, name="r2")
  # Constraints
  PRM.addConstr(sum(b[i]*u[i] for i in IA) + alpha - lamb*beta == 0)
  for i in I:
    PRM.addConstr(sum(BC[i][j]*c[j] for j in J) <= fC[i])
    PRM.addConstr(sum(BD[i][j]*d[j] for j in J) <= fD[i])
  for i in IA:
    PRM.addConstr(sum(A[i][j]*x[j] for j in J) == b[i] + r2[i])
    PRM.addSOS(GRB.SOS_TYPE1,[r2[i],u[i]])
  for j in J:
    PRM.addConstr(sum(A[i][j]*u[i] for i in IA) + r1[j] == c[j] - lamb*d[j])
    PRM.addSOS(GRB.SOS_TYPE1,[r1[j],x[j]])
  PRM.addConstr(lamb <= zdata)
  # Objective
  PRM.setObjective(lamb, GRB.MAXIMIZE)
  #PRM.params.LogToConsole = 0
  PRM.params.nonConvex = 2
  PRM.params.timeLimit = TimeLimit
  PRM.optimize()

#%% Validation
"""
Generate tables of the paper.
"""
np.random.seed(SeedNumber)
rangeC = range(Base,nC*Base+Base,Base)
rangeR = range(Base,nR*Base+Base,Base)

minzPS1 = pd.DataFrame(index = rangeR, columns = rangeC)
avezPS1 = pd.DataFrame(index = rangeR, columns = rangeC)
maxzPS1 = pd.DataFrame(index = rangeR, columns = rangeC)
minCPUPS1 = pd.DataFrame(index = rangeR, columns = rangeC)
aveCPUPS1 = pd.DataFrame(index = rangeR, columns = rangeC)
maxCPUPS1 = pd.DataFrame(index = rangeR, columns = rangeC)
RunPS1 = pd.DataFrame(index = rangeR, columns = rangeC)

minzPS2 = pd.DataFrame(index = rangeR, columns = rangeC)
avezPS2 = pd.DataFrame(index = rangeR, columns = rangeC)
maxzPS2 = pd.DataFrame(index = rangeR, columns = rangeC)
minCPUPS2 = pd.DataFrame(index = rangeR, columns = rangeC)
aveCPUPS2 = pd.DataFrame(index = rangeR, columns = rangeC)
maxCPUPS2 = pd.DataFrame(index = rangeR, columns = rangeC)
RunPS2 = pd.DataFrame(index = rangeR, columns = rangeC)

for m in rangeR:
  for n in rangeC:
    ztempPS1 = pd.Series(index=range(1,nRun+1), dtype="float64")
    CPUtempPS1 = pd.Series(index=range(1,nRun+1), dtype="float64")
    runtempPS1 = 0
    ztempPS2 = pd.Series(index=range(1,nRun+1), dtype="float64")
    CPUtempPS2 = pd.Series(index=range(1,nRun+1), dtype="float64")
    runtempPS2 = 0
    k = 1
    while k <= nRun:
      FuncGenerate(m,n)
      FuncLFP(chat,dhat)
      if LFP.status == GRB.OPTIMAL:
        if n <= SpecificColumn and m <= SpecificRow and k >= SpecificInstance:
          zhat = LFP.objVal 
          if PS1Active == 1:
            print("PS1 ","Row ",m,"Column ",n, "Run ",k)
            RunIOV = 0
            FuncCCM_DF(zhat)
            RunIOV += CCM.RUNTIME
            if CCM.status == GRB.OPTIMAL:
              FuncCCM_PF(zhat)
              RunIOV += CCM.RUNTIME
              if CCM.status == GRB.OPTIMAL:
                FuncCCM_CS(zhat)
                RunIOV += CCM.RUNTIME
              else:
                FuncCCM_MIN(zhat)
                RunIOV += CCM.RUNTIME
            else:
              FuncCCM_MAX(zhat)
              RunIOV += CCM.RUNTIME
            if RunIOV <= TimeLimit:
              cIOV = pd.Series(index=J, dtype="float64", name="cIOV")
              dIOV = pd.Series(index=J, dtype="float64", name="dIOV")
              for j in J:
                cIOV[j] = c[j].x
                dIOV[j] = d[j].x
              FuncLFP(cIOV,dIOV)
              zstar = LFP.objVal
              ztempPS1[k] = (zstar - zhat)/abs(zhat)
              CPUtempPS1[k] = RunIOV
              print("CPU Time of PS1 ","Row ",m,"Column ",n, "Run ",k)
              print(CPUtempPS1[k])
              runtempPS1 = runtempPS1 + 1

          if PS2Active == 1:
            print("PS2 ","Row ",m,"Column ",n, "Run ",k)
            RunIOV = 0
            FuncCCM_DF(zhat)
            RunIOV += CCM.RUNTIME
            if CCM.status == GRB.OPTIMAL:
              FuncCCM_PF(zhat)
              RunIOV += CCM.RUNTIME
              if CCM.status == GRB.OPTIMAL:
                FuncPRM_CS(zhat)
                RunIOV += PRM.RUNTIME
              else:
                FuncPRM_MIN(zhat)
                RunIOV += PRM.RUNTIME
            else:
              FuncPRM_MAX(zhat)
              RunIOV += PRM.RUNTIME
            if RunIOV <= TimeLimit:
              cIOV = pd.Series(index=J, dtype="float64", name="cIOV")
              dIOV = pd.Series(index=J, dtype="float64", name="dIOV")
              for j in J:
                cIOV[j] = c[j].x
                dIOV[j] = d[j].x
              FuncLFP(cIOV,dIOV)
              zstar = LFP.objVal
              ztempPS2[k] = (zstar - zhat)/abs(zhat)
              CPUtempPS2[k] = RunIOV
              print("CPU Time of PS2 ","Row ",m,"Column ",n, "Run ",k)
              print(CPUtempPS2[k])
              runtempPS2 = runtempPS2 + 1          
            
        k = k + 1
    if PS1Active == 1:
      minzPS1[n][m] = ztempPS1.min()
      avezPS1[n][m] = ztempPS1.mean()
      maxzPS1[n][m] = ztempPS1.max()
      minCPUPS1[n][m] = CPUtempPS1.min()
      aveCPUPS1[n][m] = CPUtempPS1.mean()
      maxCPUPS1[n][m] = CPUtempPS1.max()
      RunPS1[n][m] = runtempPS1
    if PS2Active == 1:
      minzPS2[n][m] = ztempPS2.min()
      avezPS2[n][m] = ztempPS2.mean()
      maxzPS2[n][m] = ztempPS2.max()
      minCPUPS2[n][m] = CPUtempPS2.min()
      aveCPUPS2[n][m] = CPUtempPS2.mean()
      maxCPUPS2[n][m] = CPUtempPS2.max()
      RunPS2[n][m] = runtempPS2
  
if PS1Active == 1:
  dfs = []
  for i, df in enumerate([minzPS1, avezPS1, maxzPS1]):
    df = df.copy()
    type_label = {0: 'Min', 1: 'Ave', 2: 'Max'}[i]
    df.index = pd.MultiIndex.from_product([df.index, [type_label]], names=['Row', 'Type'])
    dfs.append(df)
  ordered_index = pd.MultiIndex.from_product([rangeR, ['Min', 'Ave', 'Max']], names=['Row', 'Type'])
  zPS1 = pd.concat(dfs, axis=0)
  zPS1 = zPS1.sort_index(level='Row')
  zPS1 = zPS1.reorder_levels(['Row', 'Type']).loc[ordered_index]

  dfs = []
  for i, df in enumerate([minCPUPS1, aveCPUPS1, maxCPUPS1, RunPS1]):
    df = df.copy()
    type_label = {0: 'Min', 1: 'Ave', 2: 'Max', 3: 'Run'}[i]
    df.index = pd.MultiIndex.from_product([df.index, [type_label]], names=['Row', 'Type'])
    dfs.append(df)
  ordered_index = pd.MultiIndex.from_product([rangeR, ['Min', 'Ave', 'Max', 'Run']], names=['Row', 'Type'])
  CPUPS1 = pd.concat(dfs, axis=0)
  CPUPS1 = CPUPS1.sort_index(level='Row')
  CPUPS1 = CPUPS1.reorder_levels(['Row', 'Type']).loc[ordered_index]
  
if PS2Active == 1:
  dfs = []
  for i, df in enumerate([minzPS2, avezPS2, maxzPS2]):
    df = df.copy()
    type_label = {0: 'Min', 1: 'Ave', 2: 'Max'}[i]
    df.index = pd.MultiIndex.from_product([df.index, [type_label]], names=['Row', 'Type'])
    dfs.append(df)
  ordered_index = pd.MultiIndex.from_product([rangeR, ['Min', 'Ave', 'Max']], names=['Row', 'Type'])
  zPS2 = pd.concat(dfs, axis=0)
  zPS2 = zPS2.sort_index(level='Row')
  zPS2 = zPS2.reorder_levels(['Row', 'Type']).loc[ordered_index]

  dfs = []
  for i, df in enumerate([minCPUPS2, aveCPUPS2, maxCPUPS2, RunPS2]):
    df = df.copy()
    type_label = {0: 'Min', 1: 'Ave', 2: 'Max', 3: 'Run'}[i]
    df.index = pd.MultiIndex.from_product([df.index, [type_label]], names=['Row', 'Type'])
    dfs.append(df)
  ordered_index = pd.MultiIndex.from_product([rangeR, ['Min', 'Ave', 'Max','Run']], names=['Row', 'Type'])
  CPUPS2 = pd.concat(dfs, axis=0)
  CPUPS2 = CPUPS2.sort_index(level='Row')
  CPUPS2 = CPUPS2.reorder_levels(['Row', 'Type']).loc[ordered_index]
    
#%% Print
"""
Output the instances on Excel
"""
if PS1Active == 1 and PS2Active == 1:
  z_df = pd.concat([zPS1, zPS2], keys=['PS1', 'PS2'], names=['Method'], axis=1)
  z_df = z_df.swaplevel(axis=1)
  z_df.columns.names = ('Column', 'Method')
  z_df = z_df.sort_index(axis=1)
  
  CPU_df = pd.concat([CPUPS1, CPUPS2], keys=['PS1', 'PS2'], names=['Method'], axis=1)
  CPU_df = CPU_df.swaplevel(axis=1)
  CPU_df.columns.names = ('Column', 'Method')
  CPU_df = CPU_df.sort_index(axis=1)
  
  with pd.ExcelWriter('Output.xlsx') as writer:
    z_df.to_excel(writer, sheet_name='Relative Difference', index_label=['Row', 'Type'])
    CPU_df.to_excel(writer, sheet_name='CPU Time', index_label=['Row', 'Type'])
    
if PS1Active == 1 and PS2Active == 0:
  z_df = pd.concat([zPS1], keys=['PS1'], names=['Method'], axis=1)
  z_df = z_df.swaplevel(axis=1)
  z_df.columns.names = ('Column', 'Method')
  z_df = z_df.sort_index(axis=1)
  
  CPU_df = pd.concat([CPUPS1], keys=['PS1'], names=['Method'], axis=1)
  CPU_df = CPU_df.swaplevel(axis=1)
  CPU_df.columns.names = ('Column', 'Method')
  CPU_df = CPU_df.sort_index(axis=1)
  
  with pd.ExcelWriter('OutputPS1.xlsx') as writer:
    z_df.to_excel(writer, sheet_name='Relative Difference', index_label=['Row', 'Type'])
    CPU_df.to_excel(writer, sheet_name='CPU Time', index_label=['Row', 'Type'])

if PS1Active == 0 and PS2Active == 1:
  z_df = pd.concat([zPS2], keys=['PS2'], names=['Method'], axis=1)
  z_df = z_df.swaplevel(axis=1)
  z_df.columns.names = ('Column', 'Method')
  z_df = z_df.sort_index(axis=1)
  
  CPU_df = pd.concat([CPUPS2], keys=['PS2'], names=['Method'], axis=1)
  CPU_df = CPU_df.swaplevel(axis=1)
  CPU_df.columns.names = ('Column', 'Method')
  CPU_df = CPU_df.sort_index(axis=1)
  
  with pd.ExcelWriter('OutputPS2.xlsx') as writer:
    z_df.to_excel(writer, sheet_name='Relative Difference', index_label=['Row', 'Type'])
    CPU_df.to_excel(writer, sheet_name='CPU Time', index_label=['Row', 'Type'])
  
sys.stdout = orig_stdout
