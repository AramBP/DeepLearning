module DeepLearning

using Random
using Statistics
using NamedArrays
using LinearAlgebra

export predict, grads
export Perceptron
include("neural_net.jl")

export optim, GradientDescent 
export train_perceptron
include("optim.jl")

export prepare_data, split, normalize, onehot, onecold, confmat
export mse, mse_grad, cross_entropy, cross_entropy_grad, sigmoid, deriv_sigmoid
include("utils.jl")

end