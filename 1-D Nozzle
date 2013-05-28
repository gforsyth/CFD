#This code solves the flow in a 1D convergent divergent nozzle using
#finite difference method

import numpy as np; import matplotlib.pyplot as plt
print('All Modules Imported \n\n\n')

grid=np.linspace(0,3,31)    #grid set

A=1 + 2.2 * (grid-1.5)**2   #Area
Ro=1- 0.3146*grid           #Density
T = 1 - 0.2314*grid         #Temperature
V= (0.1 + 1.09*grid)*np.sqrt(T)  #Velocity
M=V/np.sqrt(T)
Mdata=[M[15]]
Mdataex=[M[30]]
Mdatain=[M[0]]
print('V is ',V)


t=0
itr=1

while(1):

 delt=(0.5*(0.1/(np.sqrt(T)+V))).min()


 i=1

 RoDdata=[0]
 VDdata=[0]
 TDdata=[0]


 while i<30:
  RoD=(((-Ro[i]*((V[i+1]-V[i])/0.1)))-((Ro[i]*V[i]*((np.log(A[i+1])-np.log(A[i]))/0.1)))-((V[i]*((Ro[i+1]-Ro[i])/0.1))))
  RoDdata.append(RoD)
  VD=(-V[i]*((V[i+1]-V[i])/0.1)) - (((T[i+1]-T[i])/0.14)+((T[i]*Ro[i+1]-T[i]*Ro[i])/(0.14*Ro[i])))
  VDdata.append(VD)
  TD=-V[i]*((T[i+1]-T[i])/0.1)-((1.4-1)*T[i])*( ((V[i+1]-V[i])/0.1) + V[i]*((np.log(A[i+1])-np.log(A[i]))/0.1))
  TDdata.append(TD)
  i=i+1


 RoDdata.append(0)
 VDdata.append(0)
 TDdata.append(0)
 

 Robar=Ro + np.asarray(RoDdata) * delt
 Vbar=V + np.asarray(VDdata) * delt
 Tbar=T + np.asarray(TDdata) * delt


 i=1

 RoDdata2=[0]
 VDdata2=[0]
 TDdata2=[0]


 while i<30:
  RoD=(((-Robar[i]*((Vbar[i]-Vbar[i-1])/0.1)))-((Robar[i]*Vbar[i]*((np.log(A[i])-np.log(A[i-1]))/0.1)))-((Vbar[i]*((Robar[i]-Robar[i-1])/0.1))))
  RoDdata2.append(RoD)
  VD=(-Vbar[i]*((Vbar[i]-Vbar[i-1])/0.1)) - (((Tbar[i]-Tbar[i-1])/0.14)+((Tbar[i]*Robar[i]-Tbar[i]*Robar[i-1])/(0.14*Robar[i])))
  VDdata2.append(VD)
  TD=-Vbar[i]*((Tbar[i]-Tbar[i-1])/0.1)-((1.4-1)*Tbar[i])*( ((Vbar[i]-Vbar[i-1])/0.1) + Vbar[i]*((np.log(A[i])-np.log(A[i-1]))/0.1))
  TDdata2.append(TD)
  i=i+1


 RoDdata2.append(0)
 VDdata2.append(0)
 TDdata2.append(0)



 RoDavg=0.5 * (np.asarray(RoDdata) + np.asarray(RoDdata2))
 VDavg=0.5 * (np.asarray(VDdata) + np.asarray(VDdata2))
 TDavg=0.5 * (np.asarray(TDdata) + np.asarray(TDdata2))



 Ro=Ro + np.asarray(RoDavg) * delt
 V=V + np.asarray(VDavg) * delt
 T=T + np.asarray(TDavg) * delt



 V[0]= 2*V[1] - V[2]
 Ro[0]= 2*Ro[1] - Ro[2]
 T[0]= 2*T[1] - T[2]

 V[30]= 2*V[29] - V[28]
 Ro[30]= 2*Ro[29] - Ro[28]
 T[30]= 2*T[29] - T[28]

 P = Ro * T
 
 M=V/np.sqrt(T)
 
 Mdata.append(M[15])
 Mdataex.append(M[30])
 Mdatain.append(M[0])
 


 t= t + delt

 itr=itr+1
 print('Iteration number : ',itr)

 if (np.absolute(Mdata[-1]-Mdata[-2])) < 0.0000001:
  break
 
    

print('\n\n P is ',P)

print('\n\n mach no. is ',M)

print('\n\n V, Ro & T are ',V,'\n\n',  Ro,'\n\n', T)

print('\n\n Iterations required = ',itr-1, '\n\n Time = ',t, ' seconds')

plt.plot(np.arange(itr),Mdata,'-',np.arange(itr),Mdataex,'-',np.arange(itr),Mdatain,'-')
plt.xlabel('X coordinate -->  Iterations')
plt.ylabel('Y coordinate -->  Mach Number')
plt.title('Variation in Mach number')
plt.legend(['Throat','Exit','Inlet'],bbox_to_anchor=(0,0),loc=4)
plt.show()





