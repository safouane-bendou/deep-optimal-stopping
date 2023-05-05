from Black_scholes import *
from Neural_network import *
from Bounds import *

if __name__ == '__main__':
    # Model parameters
        
    # Model parameters
    T = 3
    dates = 9

    size = 2
    r = 0.05
    rho = 0
    strike = 100
    sigma = np.array(size * [0.2])
    divid = np.array(size * [0.1])
    spot = np.array(size * [90])

    model = BlackScholesModel(size, r, rho, sigma, divid, spot)
    option = Max_call_option(T, dates, size, strike)

    nb_samples = 1000


    epochs = 100
    # Instantiate neural network
    pricer = Neural_Network_Pricer(model, option, nb_samples, epochs)
    training_samples = pricer.simulate_samples(pricer.nb_samples)
    trained_models = pricer.train(training_samples)
    print("Training is done. Stand by for the bounds and point estimate")
    #training is done

    #Bounds
    Ku = 1024
    J = 10

    lower_bound = compute_lower_bound(pricer, trained_models)

    upper_bound = compute_upper_bound(pricer, trained_models, Ku, J)

    print(lower_bound, upper_bound, (lower_bound + upper_bound)/2)