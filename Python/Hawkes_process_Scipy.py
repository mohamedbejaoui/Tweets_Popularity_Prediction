#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, NonlinearConstraint


# Un événement : retweet d'un utilisateur paramétré par le couple (mi, ti)
# Se modélise par un noyau suivant une loi de puissance : PHIm(t-ti)
# event de type [., .] / t de type float
def kernelFct(event, t, K = 0.024, beta = 0.5, c = 0.001, theta = 0.2, mmin = 1): 
    mi = event[0]
    ti = event[1] # Origine temporelle
    val_i = 0.
    #  Le noyau n'est pas défini aux temps < ti et à une magnitude < mmin
    if mi > mmin and t >= ti:
        # Virality * Influence of the user * Decaying (relaxation kernel)
        val_i = K * (mi / mmin)**beta / (t - ti + c)**(1+theta)
    return(val_i)

# EXEMPLE
event1, event2 = [1000, 12], [750, 68]
K, beta, c, theta = 0.8, 0.6, 10, 0.8
values_PL1, values_PL2 = [kernelFct(event1, t, K, beta, c, theta) for t in np.arange(0, 100, 0.1)], [kernelFct(event2, t, K, beta, c, theta) for t in np.arange(0, 100, 0.1)]
plt.plot(np.arange(0, 100, 0.1), values_PL1)
plt.plot(np.arange(0, 100, 0.1), values_PL2, color='r')
plt.title("Power Law memory kernel over time")

# Lambda est l'Event Rate : somme de tous les noyaux des événements d'origine temporelle < t
# cascade de type Pandas DataFrame / t de type float
def Lambda(cascade, t, K = 0.024, beta = 0.5, c = 0.001, theta = 0.2, mmin = 1):
    subset = cascade[cascade.time < t]
    Lambda = 0.
    if not subset.empty:
        kernels = [kernelFct(row.to_numpy(), t, K, beta, c, theta) for i, row in subset.iterrows()]
        Lambda = sum(kernels) # Somme des kernels pour obtenir Lambda(t)
    return(Lambda)

# EXEMPLE
real_cascade = pd.read_csv('Documents/PFE-SD9/protoprojetsd9/Python/example_book.csv')

K, beta, c, theta = 0.8, 0.6, 10, 0.8
u = [Lambda(real_cascade, t, K, beta, c, theta) for t in np.arange(0, 600)]
plt.plot(np.arange(0, 600), u) # Tracé du taux d'arrivée d'événements après 600 millisecondes

# Approximation numérique de l'intégrale de Lambda(t) pour exprimer la log-vraissemblance
# lower & upper de type float / cascade de type Pandas DataFrame
def integrateLambda(lower, upper, cascade, K = 0.024, beta = 0.5, c = 0.001, theta = 0.2, mmin = 1):
    cascade["apply"] = (cascade["magnitude"] / mmin)**beta * (1/(theta * c**theta) - 1/(theta * (upper + c - cascade["time"])**theta))
    result = cascade["apply"].sum()
    cascade.drop(columns=["apply"], inplace=True)
    return(result)

# EXEMPLE
bigT = 1500 # Etat du taux d'arrivée d'événements après 600 millisecondes
K, beta, c, theta = 0.8, 0.6, 10, 0.8
v = integrateLambda(0, bigT, real_cascade, K, beta, c, theta)
print("Intégrale de Lambda(t) sur [0, %s] = " % (bigT), v)

# On doit minimiser l'opposé de la log-vraissemblance dans le cadre d'un problème non-linéaire avec contraintes
# S'exprime avec la somme de Lambda aux temps {ti} et l'intégrale de Lambda sur l'intervalle d'observation
def neg_log_likelihood(x, *args):
    cascade = args[0] # DataFrame
    bigT = cascade["time"].max()        
    Lambda_i = [Lambda(cascade, tti, x[0], x[1], x[2], x[3]) for tti in cascade.loc[1:, "time"]] # On ne compte pas la contribution de l'événement à t=0
    return(integrateLambda(0, bigT, cascade, x[0], x[1], x[2], x[3]) - sum(np.log(Lambda_i)))

# EXEMPLE
print("neg_log_likelihood de real_cascade.csv \npour le jeu de données K = %s, beta = %s, c = %s, theta = %s :" % (K, beta, c, theta), neg_log_likelihood([K, beta, c, theta], real_cascade))


# Jacobienne de la log-vraissemblance pour assister l'algorithme
def neg_log_likelihood_jacobian(x, *args): # K, beta, c, theta = x[0], x[1], x[2], x[3]
    cascade = args[0]
    bigT = cascade["time"].max()
    # Dérivée directionnelle en K
    cascade["apply_K"] = cascade["magnitude"]**x[1] * (1/x[2]**x[3] - 1/(bigT + x[2] - cascade["time"])**x[3])
    deriv_K = cascade.loc[1:, "apply_K"].sum()/x[3] - (cascade.shape[0] - 1)/x[0]
    
    # Quelques sommes partielles aux temps (Tj) < Ti
    for j, row in cascade.iterrows():
        if j>1:
            cascade.at[j, "apply_beta_1_numerator"] = cascade.iloc[:j, :2].apply(lambda y: (y[0]**x[1]) * np.log(y[0]) / ((row["time"]-y[1]+x[2])**(1+x[3])), axis=1).sum()
            cascade.at[j, "apply_c_1_numerator"] = cascade.iloc[:j, :2].apply(lambda y: -1*(1+x[3]) * (y[0]**x[1]) / ((row["time"]-y[1]+x[2])**(2+x[3])), axis=1).sum()
            cascade.at[j, "apply_theta_1_numerator"] = cascade.iloc[:j, :2].apply(lambda y: -1*(y[0]**x[1]) * np.log(row["time"]-y[1]+x[2]) / ((row["time"]-y[1]+x[2])**(1+x[3])), axis=1).sum()
            cascade.at[j, "apply_beta_c_theta_denominator"] = cascade.iloc[:j, :2].apply(lambda y: (y[0]**x[1]) / ((row["time"]-y[1]+x[2])**(1+x[3])), axis=1).sum()
      
    # Dérivée directionnelle en beta
    cascade["apply_beta_2"] = cascade["magnitude"]**x[1] * np.log(cascade["magnitude"]) * (1/x[2]**x[3] - 1/(bigT + x[2] - cascade["time"])**x[3])
    deriv_beta = x[0] * cascade.loc[1:, "apply_beta_2"].sum()/x[3] - (cascade["apply_beta_1_numerator"]/cascade["apply_beta_c_theta_denominator"]).sum()
    
    # Dérivée directionnelle en c
    cascade["apply_c_2"] = cascade["magnitude"]**x[1] * (1/(bigT + x[2] - cascade["time"])**(1+x[3]) - 1/x[2]**(1+x[3]))
    deriv_c = x[0] * cascade.loc[1:, "apply_c_2"].sum() - (cascade["apply_c_1_numerator"]/cascade["apply_beta_c_theta_denominator"]).sum()
     
    # Dérivée directionnelle en theta
    cascade["apply_theta_2"] = cascade["magnitude"]**x[1] * ((1+x[3]*np.log(bigT+x[2]-cascade["time"]))/((bigT+x[2]-cascade["time"])**x[3]) - (1+x[3]*np.log(x[2]))/(x[2]**x[3]))
    deriv_theta = x[0] * cascade.loc[1:, "apply_theta_2"].sum()/(x[3]**2) - (cascade["apply_theta_1_numerator"]/cascade["apply_beta_c_theta_denominator"]).sum()
    # Result
    cascade.drop(columns=["apply_K","apply_beta_1_numerator","apply_c_1_numerator","apply_theta_1_numerator","apply_beta_c_theta_denominator","apply_beta_2","apply_c_2","apply_theta_2"], inplace=True)
    return([deriv_K, deriv_beta, deriv_c, deriv_theta])


# Initialisation aléatoire uniforme du set de paramètres à optimiser
def createStartPoints():
    return([np.random.uniform(0.0,1.0,size=1)[0], np.random.uniform(0.0,1.016,size=1)[0], np.random.uniform(0.0,1.0,size=1)[0], np.random.uniform(0.0,1.0,size=1)[0]])

# La contrainte non linéaire sur le branching factor n* / n_star pour qu'il y ait convergence
def constraint(x):
    return([np.log(x[0]) + np.log(1.1016) - np.log(1.016 - x[1]) - np.log(x[3]) - x[3]*np.log(x[2])])

# La jacobienne de la contrainte pour assister l'algorithme
def constraint_jacobian(x):
    return([1/x[0], 1/(1.016 - x[1]), -1*x[3]/x[2], -1*(1/x[3]) - np.log(x[2])])

x0 = createStartPoints()
res = minimize(fun = neg_log_likelihood,
               args = real_cascade,
               x0 = x0,
               method = 'SLSQP',
               jac = neg_log_likelihood_jacobian,
               bounds = Bounds([0,0,0,0], [1,1.016,np.inf,np.inf]),
               constraints = NonlinearConstraint(fun = constraint, lb=[-1*np.inf], ub=[0], jac = constraint_jacobian),
               options = {'maxiter': 1000, 'disp': True})




# Resolution non linéaire de la minimisation de la log-vraissemblance négative
def fitParameters(x0, history):
    jac = # Jacobienne
    hess = # Hessienne
    constraints = ()
    opts = # Options
    
    bounds = [[0,1],[0,1.016],[0, np.inf],[0, np.inf]] # lb & ub
    
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

"""

# Le branching factor n* caractérise le nombre moyen attendu d'événements enfantés par un événement parent
# Le nombre d'événements enfanté par un parent se calcule par SUM ( (n*)**k )
# En pratique n* < 1
def getBranchingFactor(K = 0.024, alpha = 2.016, beta = 0.5, mmin = 1, c = 0.001, theta = 0.2):
    if beta >= (alpha - 1):
        print("The closed expression calculated by this function does NOT hold for beta >= alpha - 1")
        return(np.inf)
    elif theta <= 0:
        print("The closed expression calculated by this function does NOT hold for theta <= 0 (K=%.4f, beta=%.2f, theta=%.2f)".format(K, beta, theta))
        return(np.inf)
    else:
        return(K*(alpha - 1) / ((alpha - 1 - beta) * (theta * c**theta)))
#


# Le nombre total de Retweets attendus est modélisé par la formule N = n + A1 / (1 - n*)
def getTotalEvents(history, bigT, K = 0.024, alpha = 2.016, beta = 0.5, mmin = 1, c = 0.001, theta = 0.2):
    n_star = getBranchingFactor(K, alpha, beta, mmin, c, theta)
    if n_star >= 1:
        print("Branching Factor greater than 1, not possible to predict the size (super critical regime)")
        return([np.inf, n_star, 'NA'])
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