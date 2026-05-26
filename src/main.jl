include("utils.jl")
include("train.jl")
include("neural_net.jl")

using RDatasets

iris = dataset("datasets", "iris")
X = Matrix(iris[:, 1:4])
y = iris.Species

X_train, y_train, X_test, y_test, classes = prepare_data(transpose(X), y; dims = 2)
preds = train_perceptron(3, copy(transpose(X_train)), copy(transpose(y_train)), mse, mse_grad, GD(0.1))