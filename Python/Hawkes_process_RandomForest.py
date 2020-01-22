import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
# IPOPT for non linear solver with constraints and bounds
import ipopt

# Un event est une cascade de retweets modélisée par des couples (mi, ti) magnitude-temps
class Event:
    def __init__(self, kernel_type):
        self.kernel_type = kernel_type # PL (Power Law) ou EXP (Exponential)
        

history = pd.DataFrame(data=[[5,0.2],[9,0.35],[15,0.7],[6,0.95],[1,1.1]], columns=["magnitude","time"])


# Un événement : retweet d'un utilisateur paramétré par le couple (mi, ti)
# Se modélise par un noyau suivant une loi de puissance : PHIm(t-ti)
# event de type 2-length array / t de type np.arange()
def kernelFct(event, t, K = 0.024, beta = 0.5, mmin = 1, c = 0.001, theta = 0.2):
    mi = event[0]
    ti = event[1] # Origine temporelle
    val_i = np.zeros(len(t))
    #  Le noyau n'est pas défini aux temps < ti et à une magnitude < mmin
    if mi > mmin:
        indices_definis = np.where(t>=ti)[0]
        if len(indices_definis)>0:
            # Virality * Influence of the user * Decaying (relaxation kernel)
            val_i[indices_definis] = K * (mi / mmin)**beta / (t[indices_definis] - ti + c)**(1+theta)
    return(val_i)

# EXEMPLE
event1, event2 = [1000, 12], [750, 68]
t = np.arange(0, 100, 0.1)
K, beta, mmin, c, theta = 0.8, 0.6, 1, 10, 0.8
values_PL1, values_PL2 = kernelFct(event1, t, K, beta, mmin, c, theta), kernelFct(event2, t, K, beta, mmin, c, theta)
plt.plot(t, values_PL1)
plt.plot(t, values_PL2, color='r')
plt.title("Power Law memory kernel over time")

# Lambda est l'Event Rate : somme de tous les noyaux des événements d'origine temporelle < t
def Lambda(history, t, K = 0.024, beta = 0.5, mmin = 1, c = 0.001, theta = 0.2):
    subset = history[history.time < max(t)]
    Lambda = np.zeros(len(t))
    if not subset.empty:
        kernels = [kernelFct(row.to_numpy(), t, K, beta, mmin, c, theta) for i, row in subset.iterrows()]
        # Somme élément par élément des kernels pour obtenir les coordonnées de Lambda(t)
        kernels_df = pd.DataFrame(data = kernels)
        Lambda = kernels_df.sum(axis=0).to_numpy()
    return(Lambda)

# EXEMPLE
real_cascade = pd.read_csv('Documents/PFE-SD9/protoprojetsd9/Python/example_book.csv')
t = np.arange(0, 600) # Etat du taux d'arrivée d'événements après 600 millisecondes
K, beta, mmin, c, theta = 0.8, 0.6, 1, 10, 0.8
u = Lambda(real_cascade, t, K, beta, mmin, c, theta) # 43 x 600
plt.plot(t, u)

# Approximation numérique de l'intégrale de Lambda(t) pour exprimer la log-vraissemblance
def integrateLambda(lower, upper, history, K = 0.024, beta = 0.5, mmin = 1, c = 0.001, theta = 0.2):
    history["apply"] = (history["magnitude"] / mmin)**beta * (1/(theta * c**theta) - 1/(theta * (upper + c - history["time"])**theta))
    result = history["apply"].sum()
    history.drop(columns=["apply"], inplace=True)
    return(result)

# EXEMPLE
real_cascade = pd.read_csv('Documents/PFE-SD9/protoprojetsd9/Python/example_book.csv')
bigT = 10000 # Etat du taux d'arrivée d'événements après 600 millisecondes
K, beta, mmin, c, theta = 0.8, 0.6, 1, 10, 0.8
v = integrateLambda(0, bigT, real_cascade, K, beta, mmin, c, theta)
print("Intégrale de Lambda(t) sur [0, bigT] = ", v)

# On doit minimiser l'opposé de la log-vraissemblance dans le cadre d'un problème non-linéaire avec contraintes
# S'exprime avec la somme de Lambda aux temps {ti} et l'intégrale de Lambda sur l'intervalle d'observation
def neg_log_likelihood(history, K = 0.024, beta = 0.5, mmin = 1, c = 0.001, theta = 0.2):
    bigT = history["time"].max()        
    Lambda_i = [Lambda(history, np.array([tti]), K, beta, mmin, c, theta) for tti in history.loc[1:, "time"]] # On ne compte pas la contribution de l'événement à t=0
    return(integrateLambda(0, bigT, history, K, beta, mmin, c, theta) - sum(np.log(Lambda_i)))

# EXEMPLE
print(neg_log_likelihood(real_cascade, K, beta, mmin, c, theta))

# 
def closedGradient():
    return 

def contraint():
    return 0

def jacobian():
    return 0

def createStartPoints():
    return 0

# Resolution non linéaire de la minimisation de la log-vraissemblance négative
def fitParameters(x0, history):
    jac = # Jacobienne
    hess = # Hessienne
    constraints = ()
    opts = # Options
    
    bounds = [[0,1],[0,1.016],[0, math.inf],[0, math.inf]] # lb & ub
    
    x0 = [np.random.uniform(0.0,1.0,size=1)[0], np.random.uniform(0.0,1.016,size=1)[0], np.random.uniform(0.0,1.0,size=1)[0], np.random.uniform(0.0,1.0,size=1)[0]]
    nlp = ipopt.minimize_ipopt(neg_log_likelihood, x0, bounds=bounds)
    
    nlp.solve(x0)
    nlp.values()
    
    return 0



"""
En langage R : librairie ipoptr

x0 = c(K, beta, c, theta), # Initialisation de l'itération
eval_f = neg.log.likelihood, # la fonction f non linéaire à minimiser
eval_grad_f = closedGradient,  # le gradient de f
eval_g= constraint, # le vecteur de fonctions contraintes
lb = c(K = 0, beta = 0, c = 0, theta = 0), # vecteur des bornes inférieures des variables
ub = c(K = 1, beta = 1.016, c = Inf, theta = Inf), # vecteur des bornes supérieures des variables
eval_jac_g = jacobian, # la Jacobinenne du vecteur des contraintes
eval_jac_g_structure = list(c(1,2,3,4)), # la structure de la Jacobienne
constraint_lb = c(log(.Machine$double.eps)), # vecteur des bornes inférieures des contraintes
constraint_ub = c(log(1 - .Machine$double.eps)), # vecteur des bornes supérieures des contraintes
opts = list(print_level = 0, linear_solver = "ma57", max_iter = 10000), # options : 
history = history



En Python : cyipopt / ipopt 


bounds = [[1,2],[1,2],[1,2],..]
lb, ub = get_bounds(bounds) = [[1],[1],..], [[2],[2],..]

nlp = ipopt.minimize_ipopt(fun, x0, args=(), kwargs=None, method=None, jac=None, hess=None, hessp=None,
                   bounds=None, constraints=(), tol=None, callback=None, options=None)

_x0 = np.atleast_1d(x0)
    problem = IpoptProblemWrapper(fun, args=args, kwargs=kwargs, jac=jac, hess=hess,
                                  hessp=hessp, constraints=constraints)


    cl, cu = get_constraint_bounds(constraints, x0)

    if options is None:
        options = {}

    nlp = cyipopt.problem(n = len(_x0),
                          m = len(cl),
                          problem_obj=problem,
                          lb=lb,
                          ub=ub,
                          cl=cl,
                          cu=cu)



"""


# Le branching factor n* caractérise le nombre moyen attendu d'événements enfantés par un événement parent
# Le nombre d'événements enfanté par un parent se calcule par SUM ( (n*)**k )
# En pratique n* < 1
def getBranchingFactor(K = 0.024, alpha = 2.016, beta = 0.5, mmin = 1, c = 0.001, theta = 0.2):
    if beta >= (alpha - 1):
        print("The closed expression calculated by this function does NOT hold for beta >= alpha - 1")
        return(math.inf)
    elif theta <= 0:
        print("The closed expression calculated by this function does NOT hold for theta <= 0 (K=%.4f, beta=%.2f, theta=%.2f)".format(K, beta, theta))
        return(math.inf)
    else:
        return(K*(alpha - 1) / ((alpha - 1 - beta) * (theta * c**theta)))
#


# Le nombre total de Retweets attendus est modélisé par la formule N = n + A1 / (1 - n*)
def getTotalEvents(history, bigT, K = 0.024, alpha = 2.016, beta = 0.5, mmin = 1, c = 0.001, theta = 0.2):
    n_star = getBranchingFactor(K, alpha, beta, mmin, c, theta)
    if n_star >= 1:
        print("Branching Factor greater than 1, not possible to predict the size (super critical regime)")
        return([math.inf, n_star, 'NA'])
    else:
        history["apply"] = history["magnitude"]**beta / ((bigT + c - history["magnitude"])**theta)
        a1 = K * history["apply"].sum() / theta # calculating the expected size of first level of descendants
        total_tweets = round(history.shape[0] + a1 / (1 - n_star))
        history.drop(columns=["apply"], inplace=True)
    return([total_tweets, n_star, a1])
# Pour palier au caractère non déterministe de l'expression, on introduit par la suite
# un coefficient w tel que N = n + w * A1 / (1 - n*)
# w prédit par RandomForest sur les paramètres (K, beta, c, theta)




""" NOTEBOOK



events1 <- generate_Hawkes_event_series(K = K, beta = beta, c = c, theta = theta, Tmax = 50)
events2 <- generate_Hawkes_event_series(K = K, beta = beta, c = c, theta = theta, history_init = events1, Tmax = 60)




## just to supress warnings as optimization 
oldw <- getOption("warn")
options(warn = -1)
## now for fitting we just call the fitting function, which uses IPOPT which needs some starting values for parameters
startParams <- c(K= 1,beta=1, c=250, theta= 1)
result <- fitParameters(startParams, history)


sprintf("Value of K =  %f", result$solution[1])
sprintf("Value of beta = %f", result$solution[2])
sprintf("Value of c =  %f", result$solution[3])
sprintf("Value of theta =  %f", result$solution[4])


## length of time to use for plottig
plotTime <- 600

pltData <- history[history$time <= plotTime,]
intensity <- lambda(t=seq(0,plotTime,1), history = history, params = result$solution, inclusive = T)
par(mfrow=c(2,1))
data <- data.frame(time= seq(0,plotTime,1), intensity=intensity)

plot(x = pltData$time, y = log(pltData$magnitude, base = 10), type = 'p', col = 'black', pch=16, bty = "n",
         xaxt = "n", yaxt = "n", xlab = "", ylab = "User Influence", main = "A retweet cascade as a sequence of events",
         cex.main = 1.8, cex.lab = 1.8)
axis(side = 1, cex.axis=1.4)
segments(x0 =pltData$time, y0 = 0, x1 = pltData$time, y1 = log(pltData$magnitude, base = 10),lty = 2)
points(x = pltData$time, y = log(pltData$magnitude, base = 10), type = 'p', col = 'black', pch=16)
axis(side = 2)


plot(x = data$time, y = data$intensity, type = 'l', col = 'black', pch=16, bty = "n", 
     xaxt = "n", yaxt = "n", xlab = "", ylab = "Intensity",
     cex.main = 1.8, cex.lab = 1.8)
axis(side = 1, cex.axis=1.4)
axis(side = 2)



## To get predictions from fitted model we call the function provided in rscipts
prediction <- getTotalEvents(history = history, bigT = predTime, 
                             K = result$solution[1], alpha = 2.016, 
                             beta = result$solution[2], mmin = 1,                
                             c = result$solution[3], theta = result$solution[4])


prediction

## Total length of real casacde
nReal = nrow(real_cascade)
nPredicted = prediction['total']
sprintf("The real size of cascade is %d and predicted values is %d", nReal, nPredicted)
sprintf("The relative error in percentage is %0.2f", 100*abs(nReal-nPredicted)/nReal)


"""

  

###