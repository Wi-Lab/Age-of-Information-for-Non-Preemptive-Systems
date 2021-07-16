import numpy
import multiprocessing
from functools import partial
from numpy import matrix, eye, ones, vstack, zeros, exp, sqrt, pi
from itertools import chain
from math import erf, factorial
from scipy.special import comb
from scipy.integrate import quad
import mpmath
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import LinearLocator, FormatStrFormatter

numpy.seterr(divide='ignore', invalid='ignore')
# ----------------------------------------------------
# -------------- Simulation Parameters ---------------
# ----------------------------------------------------
M = 100             # Mount Carlo Coeffient
N = 50000           # Clients count  
# ----------------------------------------------------
# -------------- Helping Functions -------------------
# ----------------------------------------------------
def printProgressBar (iteration, total, prefix = 'Progress:', suffix = 'Complete', decimals = 1, length = 50, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration / total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '')
    # Print New Line on Complete
    if iteration == total: 
        print()  
# ----------------------------------------------------
# -------------- Theory Functions --------------------
# ----------------------------------------------------
def lambdaEffectivePreemptive(λ, µ, τ):
    P = zeros(4)
    P[0] = 1.0 / (1 + ((λ[0]+λ[1]) / µ[0]) + ((λ[0]/µ[0])*(1-exp(-µ[0]*τ[0])) *(1 + (λ[1]/(µ[0]+λ[0]))*(1-exp(-(µ[0]+λ[0])*τ[1]))) * ((λ[0]+λ[1])/µ[0])) + ((λ[1]*(λ[0]+λ[1])/(µ[0]*(µ[0]+λ[0]))) * (1 - exp(-(µ[0]+λ[0])*τ[1]))))
    P[1] = ((λ[0]+λ[1]) / µ[0]) * P[0]
    P[2] = ((λ[0]/µ[0])*(1-exp(-µ[0]*τ[0])) * (1 + (λ[1]/(µ[0]+λ[0]))*(1-exp(-(µ[0]+λ[0])*τ[1]))) * ((λ[0]+λ[1])/µ[0]))  * P[0]
    P[3] = ((λ[1]*(λ[0]+λ[1])/(µ[0]*(µ[0]+λ[0]))) * (1 - exp(-(µ[0]+λ[0])*τ[1]))) * P[0]

    return λ[0] * (P[0] + (1-exp(-µ[0]*τ[0]))*(P[1]+P[3])) + λ[1] * (P[0] + (µ[0]/(µ[0]+λ[0]))*(1-exp(-(λ[0]+µ[0])*τ[1]))*P[1])
def AoI(λ, µ, τ):
    P = zeros(4)
    P[0] = 1.0 / (1 + ((λ[0]+λ[1]) / µ[0]) + ((λ[0]/µ[0])*(1-exp(-µ[0]*τ[0])) *(1 + (λ[1]/(µ[0]+λ[0]))*(1-exp(-(µ[0]+λ[0])*τ[1]))) * ((λ[0]+λ[1])/µ[0])) + ((λ[1]*(λ[0]+λ[1])/(µ[0]*(µ[0]+λ[0]))) * (1 - exp(-(µ[0]+λ[0])*τ[1]))))
    P[1] = ((λ[0]+λ[1]) / µ[0]) * P[0]
    P[2] = ((λ[0]/µ[0])*(1-exp(-µ[0]*τ[0])) * (1 + (λ[1]/(µ[0]+λ[0]))*(1-exp(-(µ[0]+λ[0])*τ[1]))) * ((λ[0]+λ[1])/µ[0]))  * P[0]
    P[3] = ((λ[1]*(λ[0]+λ[1])/(µ[0]*(µ[0]+λ[0]))) * (1 - exp(-(µ[0]+λ[0])*τ[1]))) * P[0]

    # Eq. 12
    E_Y2k_psi = 2 * ((λ[0]+λ[1])**2 + (λ[0]+λ[1])*µ[0] + µ[0]**2) / ( ((λ[0]+λ[1])**2) * µ[0]**2)
    # Eq. 10
    P_psi = (P[1])/(P[1]+P[2]+P[3])
    P_psi_bar = (P[2]+P[3])/(P[1]+P[2]+P[3])
    E_Y2k_psi_bar = 2 / µ[0]**2

    # Eq. 11
    E_Y2k = E_Y2k_psi * P_psi + E_Y2k_psi_bar * P_psi_bar


    # Eq. 23
    E_W = ( (1/µ[0] - τ[0]*exp(-µ[0]*τ[0])/(1-exp(-µ[0]*τ[0])))*P[2]/(P[1]+P[2]+P[3]) + 
            ( 1/(µ[0]+λ[0]) - τ[1]*exp(-(µ[0]+λ[0])*τ[1])/(1-exp(-(µ[0]+λ[0])*τ[1])) )*P[3]/(P[1]+P[2]+P[3]) )

    # Eq. 36
    s0 = 0.0
    s_Tot = 0.0
    for l1 in numpy.arange(0, 20):
        for l2 in numpy.arange(0, 20):
            for lp in numpy.arange(0, 20):
                xDt = (l1+lp)*τ[0] + l2*τ[1]
                if (lp == 0):
                    Pr_psi_l1l2lp_s0 = lambda s, Dt: exp(-λ[0]*(s-Dt)) * ( (λ[0]**l1)*(λ[1]**l2)*((s-Dt)**(l1+l2))/factorial(l1+l2) ) *  mpmath.hyp1f1(l2, l1+l2+1, (λ[0]-λ[1])*(s-Dt)) * µ[0] * s * exp(-µ[0]*s)
                    s0 = quad ( lambda s: Pr_psi_l1l2lp_s0(s, xDt), xDt, numpy.inf, epsabs=1e-05 )[0]
                    s_Tot += s0
                else:
                    S_i = 0.0
                    for i in numpy.arange (1, lp+1): 
                        def Pr_psi_l1l2lp_s (s, Dt):
                            S_m = 0.0
                            for m in range (i):
                                
                                S_k = 0.0
                                for k in range (lp):
                                    S_k += (comb(lp-1, k) * (-1)**k * ((s-Dt-m*τ[1])**(lp-1-k))  / (l1+l2+lp+k+1.0)) * ( ((s-Dt-(i-1)*τ[1])**(l1+l2+lp+k+1.0)) * mpmath.hyp2f2(l2+lp, l1+l2+lp+k+1, l1+l2+lp+1, l1+l2+lp+k+2, (λ[0]-λ[1])*(s-Dt-(i-1)*τ[1])) - ((s-Dt-min(s-Dt, lp*τ[1]))**(l1+l2+lp+k+1)) * mpmath.hyp2f2(l2+lp, l1+l2+lp+k+1, l1+l2+lp+1, l1+l2+lp+k+2, (λ[0]-λ[1])*(s-Dt-min(s-Dt, lp*τ[1]))) ) 

                                S_m += (((-1.0)**m) * lp / (factorial(m) * factorial(lp-m))) * (exp(-λ[0]*(s-Dt))) * S_k
                            return S_m
                        S_i += quad ( lambda s: Pr_psi_l1l2lp_s(s, xDt) * µ[0] * s * exp(-µ[0]*s), xDt + i*τ[1] , numpy.inf, epsabs=1e-05 )[0]                    
                    s_Tot += ((λ[0]/(1.0-exp(-λ[0]*τ[1])))**lp) * ((λ[0]**l1)*(λ[1]**(l2+lp))/factorial(l1+l2+lp)) * S_i

    # Eq. 29
    E_S_psi = (1.0/P_psi) * s_Tot

    # Eq. 16 
    E_TY = (1.0/(λ[0]+λ[1])) * (E_W + E_S_psi) * P_psi + (1/µ[0])*(E_W + 1.0/µ[0])
    return lambdaEffectivePreemptive(λ, µ, τ) * (0.5 * E_Y2k + E_TY)  
# ----------------------------------------------------
# -------------- Theory Functions Eph ----------------
# ----------------------------------------------------
def lambdaEffectiveEph(λ, µ, τ):
    return λ[0]*(µ[0]**2 + λ[0]*µ[0]*(1-exp(-µ[0]*τ[0])))/(µ[0]**2+λ[0]*µ[0]+λ[0]**2*(1-exp(-µ[0]*τ[0])))
def AoIEph(λ, µ, τ):
    return (µ[0]**3 + 2*λ[0]*µ[0]**2 + 2*(λ[0]**2)*µ[0] + (λ[0]**3)*(3 - (3+µ[0]*τ[0])*exp(-µ[0]*τ[0]))) / (λ[0]*µ[0]*(µ[0]**2 + λ[0]*µ[0] + (λ[0]**2)*(1-exp(-µ[0]*τ[0])) ))
# ----------------------------------------------------
# -------------- Simulation Functions ----------------
# ----------------------------------------------------
class Server:
    inServiceClient = None
    inQueueClient = None
    inCommingClient = None

    isServerFree = None
    isQueueFree = None

    inState_0_time = 0
    inState_1_time = 0
    inState_2_time = 0
    inState_3_time = 0

    def UpdateServerStatus(self):
        if(self.inServiceClient == None):
            self.isServerFree = True
            self.isQueueFree = True

        elif (self.inServiceClient.ArrivalTime + self.inServiceClient.inQueueTime + self.inServiceClient.ServiceTime <= self.inCommingClient.ArrivalTime):
            self.inServiceClient.DepartureTime = self.inServiceClient.ArrivalTime + self.inServiceClient.inQueueTime + self.inServiceClient.ServiceTime
            self.inServiceClient.Status = 'S' # 'Served'

            if (self.inQueueClient == None):
                self.isServerFree = True
                self.isQueueFree = True                
            else:
                self.inQueueClient.inQueueTime = self.inServiceClient.DepartureTime - self.inQueueClient.ArrivalTime

                if(self.inQueueClient.inQueueTime <= self.inQueueClient.QueueDeadLineTime):
                    if (self.inQueueClient.ArrivalTime + self.inQueueClient.inQueueTime + self.inQueueClient.ServiceTime <= self.inCommingClient.ArrivalTime):
                        self.inQueueClient.DepartureTime = self.inQueueClient.ArrivalTime + self.inQueueClient.inQueueTime + self.inQueueClient.ServiceTime
                        self.inQueueClient.Status = 'S' # 'Served'
                        self.inServiceClient = None     #??
                        self.inQueueClient = None
                        self.isServerFree = True
                        self.isQueueFree = True 
                    else:
                        self.inServiceClient = self.inQueueClient
                        self.inQueueClient = None
                        self.inServiceClient.Status = 'inService'
                        self.isServerFree = False
                        self.isQueueFree = True
                else:
                    self.inQueueClient.DepartureTime = self.inQueueClient.ArrivalTime + self.inQueueClient.QueueDeadLineTime
                    self.inQueueClient.Status = 'DD'    # 'DeadlineDrop'
                    self.inServiceClient = None     #??
                    self.inQueueClient = None
                    self.isServerFree = True
                    self.isQueueFree = True
        else:
            self.isServerFree = False            
            if (self.inQueueClient == None):                
                self.isQueueFree = True
            else:                
                if(self.inCommingClient.ArrivalTime - self.inQueueClient.ArrivalTime <= self.inQueueClient.QueueDeadLineTime):
                    self.isQueueFree = False                    
                else:
                    self.inQueueClient.DepartureTime = self.inQueueClient.ArrivalTime + self.inQueueClient.QueueDeadLineTime
                    self.inQueueClient.Status = 'DD'    # 'DeadlineDrop'
                    self.inQueueClient = None
                    self.isQueueFree = True

    def inQueuePriority(self):
        return self.inQueueClient.Priority
    
    def Serve(self, client):
        self.inCommingClient = client
        self.UpdateServerStatus()  

        if (self.isServerFree):
            self.inServiceClient = client
            client.inQueueTime = 0.0                
            client.Status = 'inService'

        else:
            if (self.isQueueFree):
                self.inQueueClient = client
                client.Status = 'inQueue'
    
            elif (client.Priority < self.inQueuePriority()):                               
                self.inQueueClient.Status = 'PD' # 'PriorityDrop'
                self.inQueueClient.inQueueTime = client.ArrivalTime - self.inQueueClient.ArrivalTime
                self.inQueueClient.DepartureTime = self.inQueueClient.ArrivalTime + self.inQueueClient.inQueueTime

                client.Status = 'inQueue'
                client.OS = "PS"
                self.inQueueClient = client
            else:
                client.DepartureTime = client.ArrivalTime
                client.Status = 'B' # 'Blocked'
class Client:
    Index = 0
    Type = -1
    ArrivalTime = 0.0
    ServiceTime = 0.0
    Priority = 0    
    inQueueTime = 0.0
    QueueDeadLineTime = numpy.Inf
    DepartureTime = 0.0    
    Status = None
    OS = None

    def __init__(self, index):
        self.Index = index
def SystemModel(λ, ρ, τ, µ, m):
    # λ      Arrival Rate
    # ρ      Priority
    # τ      Deadline 
    # µ      Service Rate
    numpy.random.seed()

    ServedClientsStats = numpy.zeros(len(λ))
    BlockedClientsStats = numpy.zeros(len(λ))
    PriorityDropClientsStats = numpy.zeros(len(λ))
    DeadLineDropClientsStats = numpy.zeros(len(λ))
    ClientTypeCount = numpy.zeros(len(λ))

    S = Server()
    tmpClients = []
    # Generate clients
    for t in range (len(λ)):
        clients = []
        for n in range (N):
            clients.append(Client(n))
            clients[n].Type = t
            clients[n].Priority = ρ[t]

            if (n > 0):
                clients[n].ArrivalTime = numpy.random.exponential(1.0 / λ[t]) + clients[n - 1].ArrivalTime            
            else:
                clients[n].ArrivalTime = numpy.random.exponential(1.0 / λ[t])

            clients[n].ServiceTime = numpy.random.exponential(1.0 / µ[0])             
            clients[n].QueueDeadLineTime = τ[t]
            #clients[n].QueueDeadLineTime = numpy.random.exponential(1.0 / τ[t])
        tmpClients.append(clients)

    flat_ClientList = [item for sublist in tmpClients for item in sublist]
    clients = (sorted(flat_ClientList, key=lambda x: x.ArrivalTime))[0:N]

    # Send clients to server as they arrive
    for n in range (N):
        clients[n].Index = n
        S.Serve(clients[n])

    # Add a client at infinity to make sure all Clients have Served.
    lastClient = Client(N)
    lastClient.ArrivalTime = numpy.Inf
    S.Serve(lastClient)

    servedClientsList = []
    for n in range (N):    
        if (clients[n].Status == 'S'):
            servedClientsList.append(clients[n])

    # Age Points Generator
    X = []
    Y = []
    X.append(0.0)
    Y.append(0.0)
    Q = 0.0
    for n in range(N):
        if (clients[n].Status == 'S'):
            X.append(clients[n].DepartureTime)
            Y.append(clients[n].DepartureTime - X[len(Y) - 1] + Y[len(Y) - 1])

            X.append(clients[n].DepartureTime)
            Y.append((clients[n].DepartureTime - clients[n].ArrivalTime))

            Q += 0.5 * (Y[len(Y) - 2] + Y[len(Y) - 3]) * (X[len(X) - 2] - X[len(X) - 3])


    for n in range(N):
        ClientTypeCount[clients[n].Type] +=1
        if (clients[n].Status == 'S'):
            ServedClientsStats[clients[n].Type] +=1
        elif (clients[n].Status == 'B'):
            BlockedClientsStats[clients[n].Type] +=1
        elif (clients[n].Status == 'PD'):
            PriorityDropClientsStats[clients[n].Type] +=1
        elif (clients[n].Status == 'DD'):
            DeadLineDropClientsStats[clients[n].Type] +=1
    
    #λeff = numpy.sum(numpy.multiply(λ, numpy.true_divide(ServedClientsStats, ClientTypeCount)))
    λeff = numpy.sum(ServedClientsStats) / clients[N-1].DepartureTime    
    Age = λeff * Q / numpy.sum(ServedClientsStats)
    EWk = 0
    for client in clients:
        if (client.Status == 'S'):
            EWk += client.inQueueTime
    EWk /= numpy.sum(ServedClientsStats)
    # EWk = numpy.sum(client.inQueueTime for client in clients if client.Status == 'S') / clients[N-1].DepartureTime
    # Age = λeff * Q / clients[N-1].DepartureTime
    # Age = numpy.sum(numpy.multiply(λ, numpy.true_divide(ServedClientsStats, ClientTypeCount)))    

    EY2K = 0
    ts = 0
    events = []
    for n in range (N):
        if(clients[n].Status == 'S'):
            EY2K += (clients[n].DepartureTime - ts)**2
            ts = clients[n].DepartureTime
        events.append([clients[n].Index, clients[n].Type, clients[n].ArrivalTime, +1])
        events.append([clients[n].Index, clients[n].Type, clients[n].DepartureTime, -1])
    events = (sorted(events, key=lambda x: (x[2], -x[3])))

    EY2K /= numpy.sum(ServedClientsStats)

    P_psi = µ[0] / (µ[0] + λ[0]*(1-exp(-µ[0]*τ[0])) + λ[1]*(1-exp(-µ[0]*τ[1])) )
    ES_psi = (Age / λeff - 0.5 * EY2K - (1/µ[0])*(EWk + (1/µ[0])) - (1/(λ[0]+λ[1]))*EWk*P_psi) * (λ[0]+λ[1]) / P_psi

    inState00Time = 0
    inState10Time = 0
    inState21Time = 0
    inState22Time = 0
    inState00User0ArrivalCount = 0
    inState10User0ArrivalCount = 0
    inState21User0ArrivalCount = 0
    inState22User0ArrivalCount = 0
    inState00User0ServedArrivalCount = 0
    inState10User0ServedArrivalCount = 0
    inState21User0ServedArrivalCount = 0
    inState22User0ServedArrivalCount = 0
    inState00User0notServedArrivalCount = 0
    inState10User0notServedArrivalCount = 0
    inState21User0notServedArrivalCount = 0
    inState22User0notServedArrivalCount = 0
    inState00User1ArrivalCount = 0
    inState10User1ArrivalCount = 0
    inState21User1ArrivalCount = 0
    inState22User1ArrivalCount = 0
    inState00User1ServedArrivalCount = 0
    inState10User1ServedArrivalCount = 0
    inState21User1ServedArrivalCount = 0
    inState22User1ServedArrivalCount = 0
    inState00User1notServedArrivalCount = 0
    inState10User1notServedArrivalCount = 0
    inState21User1notServedArrivalCount = 0
    inState22User1notServedArrivalCount = 0
    
    C10to00 = 0
    totalClientinServer = 0
    t0 = 0    
    inQueueUserType = -1
    for n in range (2*N):
        diff = events[n][2] - t0
        t0 = events[n][2]

        if (totalClientinServer == 0):
            inState00Time += diff
            if (events[n][3] == +1):        # Arrival
                if (events[n][1] == 0):     # User Type
                    inState00User0ArrivalCount +=1
                    if (clients[events[n][0]].Status == 'S'):
                        inState00User0ServedArrivalCount += 1
                    else:
                        inState00User0notServedArrivalCount += 1
                elif (events[n][1] == 1):   # User Type
                    inState00User1ArrivalCount +=1
                    if (clients[events[n][0]].Status == 'S'):
                        inState00User1ServedArrivalCount += 1
                    else:
                        inState00User1notServedArrivalCount += 1
        elif (totalClientinServer == 1):
            inState10Time += diff
            if (events[n][3] == +1):
                if (events[n][1] == 0):
                    inState10User0ArrivalCount +=1
                    if (clients[events[n][0]].Status == 'S'):
                        inState10User0ServedArrivalCount += 1
                    else:
                        inState10User0notServedArrivalCount += 1
                elif (events[n][1] == 1):
                    inState10User1ArrivalCount +=1
                    if (clients[events[n][0]].Status == 'S'):
                        inState10User1ServedArrivalCount += 1
                    else:
                        inState10User1notServedArrivalCount += 1
            else:
                C10to00 +=1
        elif (totalClientinServer == 2):
            if (inQueueUserType == 0):
                inState21Time += diff
                if (events[n][3] == +1):
                    if (events[n][1] == 0):
                        inState21User0ArrivalCount +=1
                        if (clients[events[n][0]].Status == 'S'):
                            inState21User0ServedArrivalCount += 1
                        else:
                            inState21User0notServedArrivalCount += 1
                    elif (events[n][1] == 1):
                        inState21User1ArrivalCount +=1
                        if (clients[events[n][0]].Status == 'S'):
                            inState21User1ServedArrivalCount += 1
                        else:
                            inState21User1notServedArrivalCount += 1
            elif (inQueueUserType == 1):
                inState22Time += diff
                if (events[n][3] == +1):
                    if (events[n][1] == 0):
                        inState22User0ArrivalCount +=1
                        if (clients[events[n][0]].Status == 'S'):
                            inState22User0ServedArrivalCount += 1
                        else:
                            inState22User0notServedArrivalCount += 1
                    elif (events[n][1] == 1):
                        inState22User1ArrivalCount +=1
                        if (clients[events[n][0]].Status == 'S'):
                            inState22User1ServedArrivalCount += 1
                        else:
                            inState22User1notServedArrivalCount += 1

        totalClientinServer += events[n][3]
        if (totalClientinServer<0 or totalClientinServer > 3):
            print("error")

        if (totalClientinServer < 2):
            inQueueUserType = -1
        else:
            if (events[n][3] == +1):        # Arrival Happen
                if(clients[events[n][0]].Status != 'B'):
                    inQueueUserType = events[n][1]
            else:                           # Departue Happen
                if (clients[events[n][0]].Status == 'PD'):
                    inQueueUserType = events[n-1][1]
    
    totalTime_ = inState00Time + inState10Time + inState21Time + inState22Time
    totalTime = events[2*N-1][2]

    if (totalTime - totalTime_ > 0.0001):
        print("error")
    
    P_inState00Time = inState00Time / totalTime_
    P_inState10Time = inState10Time / totalTime_
    P_inState21Time = inState21Time / totalTime_
    P_inState22Time = inState22Time / totalTime_

    Pr = numpy.array([P_inState00Time, P_inState10Time, P_inState21Time, P_inState22Time])
    AC0 = [inState00User0ArrivalCount, inState10User0ArrivalCount, inState21User0ArrivalCount, inState22User0ArrivalCount]
    SAC0 = [inState00User0ServedArrivalCount, inState10User0ServedArrivalCount, inState21User0ServedArrivalCount, inState22User0ServedArrivalCount]
    notSAC0 = [inState00User0notServedArrivalCount, inState10User0notServedArrivalCount, inState21User0notServedArrivalCount, inState22User0notServedArrivalCount]
    SAC0toAC0 = numpy.array(SAC0, dtype='f') / numpy.array(AC0, dtype='f')
    notSAC0toAC0 = numpy.array(notSAC0, dtype='f') / numpy.array(AC0, dtype='f')

    AC1 = [inState00User1ArrivalCount, inState10User1ArrivalCount, inState21User1ArrivalCount, inState22User1ArrivalCount]
    SAC1 = [inState00User1ServedArrivalCount, inState10User1ServedArrivalCount, inState21User1ServedArrivalCount, inState22User1ServedArrivalCount]
    notSAC1 = [inState00User1notServedArrivalCount, inState10User1notServedArrivalCount, inState21User1notServedArrivalCount, inState22User1notServedArrivalCount]
    SAC1toAC1 = numpy.array(SAC1, dtype='f') / numpy.array(AC1, dtype='f')
    notSAC1toAC1 = numpy.array(notSAC1, dtype='f') / numpy.array(AC1, dtype='f')    
    
    P_psi_ = C10to00 / numpy.sum(ServedClientsStats)

    if (sum(SAC0) + sum(notSAC0) + sum(SAC1) + sum(notSAC1) != N):
        print('error')
    return numpy.array([λeff/M, Age/M, Pr/M, SAC0toAC0, notSAC0toAC0, SAC1toAC1, notSAC1toAC1, EWk/M, EY2K/M, ES_psi/M, P_psi_/M], dtype=object)
# ----------------------------------------------------
# ----------------------------------------------------
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', True)    

    steps = numpy.arange(0.1, 6.0, 0.25)
    printProgressBar(0.0, steps[-1])
    λeff_simulation1 = []
    Age_simulation1 = []
    λeff_theory1 = []
    Age_theory1 = [] 
    λeff_simulation2 = []
    Age_simulation2 = []
    λeff_theory2 = []
    Age_theory2 = [] 
    λeff_simulation3 = []
    Age_simulation3 = []
    λeff_theory3 = []
    Age_theory3 = [] 


    # -------------------------------------------------------SERIES1 START   
    for step in steps:
        λ = numpy.array([1.0, 1.0])         # Arrival Rate
        ρ = numpy.array([0, 1])             # Priority
        τ = numpy.array([step, 2.0])        # Deadline  
        µ = numpy.array([1])                # Service Rate            
        

        funcSystemModel = partial(SystemModel, λ, ρ, τ, µ)
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            result = pool.map(funcSystemModel, numpy.linspace(0, M, M, endpoint=False))

        tmp = numpy.sum(result, axis=0)
        λeff_simulation1.append(tmp[0])
        Age_simulation1.append(tmp[1]) 
        λeff_theory1.append(lambdaEffectivePreemptive(λ, µ, τ))        
        Age_theory1.append(AoI(λ, µ, τ))
        printProgressBar(step, steps[-1])
    # -------------------------------------------------------SERIES1 END
    # --------------------------------------------------------------------


    fc = 101
    d1Type = '_var' 
    d2Type = '_fix=1'
    l1 = '_fix=1'
    l2 = '_fix=1'
    mu = '_fix=1'
    numpy.save('./FResults/' + str(fc) + '/landaeff_simulation' + l1 + l2 + mu + d1Type + d2Type + '.npy', λeff_simulation1)
    numpy.save('./FResults/' + str(fc) + '/Age_simulation' + l1 + l2 + mu + d1Type + d2Type + '.npy', Age_simulation1)
    numpy.save('./FResults/' + str(fc) + '/landaeff_theory' + l1 + l2 + mu + d1Type + d2Type + '.npy', λeff_theory1)
    numpy.save('./FResults/' + str(fc) + '/Age_theory' + l1 + l2 + mu + d1Type + d2Type + '.npy', Age_theory1)


    # --------------------------------------------------------------------
    # -------------------------------------------------------SERIES2 START   
    for step in steps:
        λ = numpy.array([1.0, 1.0])         # Arrival Rate
        ρ = numpy.array([0, 1])             # Priority
        τ = numpy.array([step, 2.0])        # Deadline  
        µ = numpy.array([2])                # Service Rate           
        

        funcSystemModel = partial(SystemModel, λ, ρ, τ, µ)
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            result = pool.map(funcSystemModel, numpy.linspace(0, M, M, endpoint=False))

        tmp = numpy.sum(result, axis=0)
        λeff_simulation2.append(tmp[0])
        Age_simulation2.append(tmp[1]) 
        λeff_theory2.append(lambdaEffectivePreemptive(λ, µ, τ))        
        Age_theory2.append(AoI(λ, µ, τ))
        printProgressBar(step, steps[-1])
    # -------------------------------------------------------SERIES2 END
    # --------------------------------------------------------------------

    d1Type = '_var' 
    d2Type = '_fix=1'
    l1 = '_fix=2'
    l2 = '_fix=1'
    mu = '_fix=1'
    numpy.save('./FResults/' + str(fc) + '/landaeff_simulation' + l1 + l2 + mu + d1Type + d2Type + '.npy', λeff_simulation2)
    numpy.save('./FResults/' + str(fc) + '/Age_simulation' + l1 + l2 + mu + d1Type + d2Type + '.npy', Age_simulation2)
    numpy.save('./FResults/' + str(fc) + '/landaeff_theory' + l1 + l2 + mu + d1Type + d2Type + '.npy', λeff_theory2)
    numpy.save('./FResults/' + str(fc) + '/Age_theory' + l1 + l2 + mu + d1Type + d2Type + '.npy', Age_theory2)

    # --------------------------------------------------------------------
    # -------------------------------------------------------SERIES3 START   
    for step in steps:
        λ = numpy.array([1.0, 1.0])         # Arrival Rate
        ρ = numpy.array([0, 1])             # Priority
        τ = numpy.array([step, 2.0])        # Deadline  
        µ = numpy.array([3])                # Service Rate                
        

        funcSystemModel = partial(SystemModel, λ, ρ, τ, µ)
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            result = pool.map(funcSystemModel, numpy.linspace(0, M, M, endpoint=False))

        tmp = numpy.sum(result, axis=0)
        λeff_simulation3.append(tmp[0])
        Age_simulation3.append(tmp[1]) 
        λeff_theory3.append(lambdaEffectivePreemptive(λ, µ, τ))        
        Age_theory3.append(AoI(λ, µ, τ))
        printProgressBar(step, steps[-1])
    # -------------------------------------------------------SERIES3 END

    d1Type = '_var' 
    d2Type = '_fix=1'
    l1 = '_fix=4'
    l2 = '_fix=1'
    mu = '_fix=1'
    numpy.save('./FResults/' + str(fc) + '/landaeff_simulation' + l1 + l2 + mu + d1Type + d2Type + '.npy', λeff_simulation3)
    numpy.save('./FResults/' + str(fc) + '/Age_simulation' + l1 + l2 + mu + d1Type + d2Type + '.npy', Age_simulation3)
    numpy.save('./FResults/' + str(fc) + '/landaeff_theory' + l1 + l2 + mu + d1Type + d2Type + '.npy', λeff_theory3)
    numpy.save('./FResults/' + str(fc) + '/Age_theory' + l1 + l2 + mu + d1Type + d2Type + '.npy', Age_theory3)


    # -------------------------------------------------------PLOT
    plt.plot(steps, Age_simulation1, marker="^", markersize=4, color='firebrick', linewidth=1.0)    
    plt.plot(steps, Age_theory1, marker="^", markersize=4, color='royalblue', linewidth=1.0) 
    plt.plot(steps, Age_simulation2, marker="s", markersize=4, color='firebrick', linewidth=1.0)    
    plt.plot(steps, Age_theory2, marker="s", markersize=4, color='royalblue', linewidth=1.0)  
    plt.plot(steps, Age_simulation3, marker="o", markersize=4, color='firebrick', linewidth=1.0)    
    plt.plot(steps, Age_theory3, marker="o", markersize=4, color='royalblue', linewidth=1.0) 

    simulation_line = mlines.Line2D([], [], color='firebrick', label='Simulation')
    theory_line = mlines.Line2D([], [], color='royalblue', label='Theory')
    type_1 = mlines.Line2D([], [], color='gray', marker='^', markersize=6, label='$\\mu = 1.0$')
    type_2 = mlines.Line2D([], [], color='gray', marker='s', markersize=6, label='$\\mu = 2.0$')
    type_3 = mlines.Line2D([], [], color='gray', marker='o', markersize=6, label='$\\mu = 3.0$')

    plt.tick_params(axis='both',bottom=True, top=True, left=True, right=True, direction='in', which='major', grid_color='Black')
    plt.grid(linestyle='--', linewidth=1.0, alpha=0.15)
    legend1 = plt.legend(handles=[simulation_line, theory_line], loc='lower center',  ncol=2)

    plt.legend(handles=[type_1, type_2, type_3 ])
    plt.gca().add_artist(legend1)

    plt.ylim(numpy.min(numpy.concatenate([Age_simulation1, Age_simulation2, Age_simulation3])) - 0.5,
            numpy.max(numpy.concatenate([Age_simulation1, Age_simulation2, Age_simulation3])) + 0.5)    
    plt.xlabel('$D_1$')
    plt.ylabel('$AoI$')
    plt.show() 