using Pkg
cd(joinpath(@__DIR__, ".."))
Pkg.activate(".")

using RDatasets
using DeepLearning

t(X) = copy(transpose(X))

iris = dataset("datasets", "iris")
X = Matrix(iris[:, 1:4])
y = convert(Vector{String}, iris.Species)

X_train, y_train, X_test, y_test, classes = prepare_data(transpose(X), y; dims = 2)
gd = GradientDescent(0.1)
m, xs = train_perceptron(3, t(X_train), t(y_train), mse, mse_grad, sigmoid, deriv_sigmoid, gd; max_it = 1000)
predictions = onecold(t(predict(m, t(X_test))), classes)
targets = onecold(y_test, classes)
cf = confmat(predictions, targets, classes)
@show cf