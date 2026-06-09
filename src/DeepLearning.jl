module DeepLearning

using Random
using Statistics
using NamedArrays
using LinearAlgebra

export prepare_data, split, normalize, onehot, onecold, confmat
export mse, mse_grad, cross_entropy, cross_entropy_grad, sigmoid, deriv_sigmoid
include("utils.jl")

export predict, grads
export Perceptron, StochasticGradientDescent
include("neural_net.jl")

export optim, GradientDescent 
export train_perceptron
include("optim.jl")

end