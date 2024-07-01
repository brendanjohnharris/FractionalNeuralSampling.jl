using FractionalNeuralSampling
using Turing
using DifferentialEquations

# Define a simple Normal model with unknown mean and variance.
@model function gdemo(x, y) # Some function
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))
    x ~ Normal(m, sqrt(s²))
    y ~ Normal(m, sqrt(s²))
end
c = sample(gdemo(1.5, 2), HMC(0.1, 5), 1000)
