#   import numpy as np
#   import pickle
#   from typing import Callable
#   from enum import Enum
#   
#   class Util:
#       @staticmethod
#       def relu(input: np.ndarray) -> np.ndarray:
#           return np.maximum(0, input)
#       
#       @staticmethod
#       def softmax(input: np.ndarray) -> np.ndarray:
#           exp_values = np.exp(input - np.max(input))  # For numerical stability
#           return exp_values / np.sum(exp_values)
#   
#       @staticmethod
#       def passF(input: np.ndarray) -> np.ndarray:
#           return np.array(input)
#       
#       @staticmethod
#       def deriveList(function: Callable, input: np.ndarray, dx=1e-5):
#           dx_values = function(input + dx)
#           x_values = function(input)
#           return (dx_values - x_values) / dx
#   
#   class Function(Enum):
#       RELU = Util.relu
#       SOFTMAX = Util.softmax
#       PASS = Util.passF
#   
#   class Activation:    
#       def __init__(self, function: Function = Function.PASS):
#           self.function = function
#       
#       def compute(self, input: np.ndarray) -> np.ndarray:
#           return self.function(input)
#   
#   class Layer:
#       def __init__(self, width: int, prevLayerWidth: int, activation: Activation = None):
#           self.width = width
#           self.weights = np.random.randn(width, prevLayerWidth)
#           self.biases = np.random.randn(width)
#           self.activation = activation
#           self.output = np.zeros(width)
#   
#       def compute(self, input: np.ndarray) -> np.ndarray:
#           z = np.dot(self.weights, input) + self.biases
#           self.output = self.activation.compute(z) if self.activation else z
#           return self.output
#   
#   class Model:
#       PICKLE_SAFE_NAME = "model.pickle"
#   
#       def __init__(self, learningRate=0.01):
#           self.layers = []
#           self.postActivation = []
#           self.preActivation = []
#           self.gradients = []
#           self.input = []
#           self.learningRate = learningRate
#   
#       def addLayer(self, layer: Layer):
#           self.layers.append(layer)
#           return self
#   
#       def forward(self, input: np.ndarray) -> np.ndarray:
#           self.postActivation = []
#           self.preActivation = []
#           for layer in self.layers:
#               self.preActivation.append(input)
#               input = layer.compute(input)
#               self.postActivation.append(input)
#           return input
#   
#       def backward(self, target: np.ndarray):
#           # Initialize gradients
#           self.gradients = [None] * len(self.layers)
#           # Output layer gradient
#           self.gradients[-1] = (
#               (self.postActivation[-1] - target) *
#               (self.postActivation[-1] * (1 - self.postActivation[-1]))  # Assuming sigmoid activation
#           )
#   
#           # Hidden layer gradients
#           for i in range(len(self.layers) - 2, -1, -1):
#               self.gradients[i] = (
#                   np.dot(self.layers[i + 1].weights.T, self.gradients[i + 1]) *
#                   (self.preActivation[i] > 0)  # Assuming ReLU for hidden layers
#               )
#   
#       def update(self):
#           for i, layer in enumerate(self.layers):
#               layer.weights -= self.learningRate * np.outer(self.gradients[i], self.preActivation[i])
#               layer.biases -= self.learningRate * self.gradients[i]
#   
#       def save(self, path: str = PICKLE_SAFE_NAME) -> bool:
#           try:
#               with open(path, "wb") as file:
#                   pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
#               return True
#           except Exception as e:
#               print(f"Error saving model: {e}")
#               return False
#       
#       @staticmethod
#       def load(path: str = PICKLE_SAFE_NAME):
#           with open(path, "rb") as file:
#               return pickle.load(file)
#   
#   
#   # Define the model
#   model = Model(learningRate=0.1)
#   
#   # Add layers
#   model.addLayer(Layer(3, 2, Activation(Function.RELU)))  # Hidden layer
#   model.addLayer(Layer(2, 3, Activation(Function.SOFTMAX)))  # Output layer
#   
#   # Input and target
#   input_data = np.array([1.0, 2.0])
#   target = np.array([0.0, 1.0])
#   
#   # Forward pass
#   output = model.forward(input_data)
#   
#   # Backward pass
#   model.backward(target)
#   
#   # Update weights
#   model.update()
#   
#   # Check updated weights
#   print("Updated weights for first layer:", model.layers[0].weights)


import numpy, pickle, dill, json
from typing import Callable, Self
from enum import Enum

class Util:
    @staticmethod
    def relu(input: list[float]) -> numpy.ndarray:
        return numpy.maximum(0, input)
    
    @staticmethod
    def softmax(input: list[float]) -> numpy.ndarray:
        exp_values = numpy.exp(input - numpy.max(input))  # For numerical stability
        return exp_values / numpy.sum(exp_values)

    @staticmethod
    def passF(input: list[float]) -> numpy.ndarray:
        return numpy.array(input)
    
    @staticmethod
    def deriveList(function: Callable, input: list[float], dx=1e-5):
        input = numpy.array(input)
        dx_values = function(input + dx)
        x_values = function(input)
        return (dx_values - x_values) / dx

class Function(Enum):
    RELU = Util.relu
    SOFTMAX = Util.softmax
    PASS = Util.passF

class Activation:    
    def __init__(self, function: Function = Function.PASS):
        self.function = function
    
    def compute(self, input: list[float]) -> numpy.ndarray:
        return self.function(input)

class Layer:
    def __init__(self, width: int, prevLayerWidth: int, activation: Activation = None):
        self.width = width
        self.weights = numpy.random.randn(width, prevLayerWidth)
        self.biases = numpy.random.randn(width)
        self.activation = activation
        self.output = numpy.zeros(width)

    def compute(self, input: numpy.ndarray) -> numpy.ndarray:
        self.output = numpy.dot(self.weights, input) + self.biases
        if self.activation:
            self.output = self.activation.compute(self.output)
        return self.output

class Model:
    PICKLE_SAFE_NAME = "model.pickle"

    def __init__(self, layers: list[Layer] = []):
        self.layers = layers
        self.postActivation = []
        self.preActivation = []
        self.gradients = []
        for layer in layers:
            self.gradients.append([0 for _ in range(layer.width)])
        self.input = []
        self.learningRate = 0.01

    def addLayer(self, layer: Layer) -> Self:
        self.layers.append(layer)
        self.gradients.append([0 for _ in range(layer.width)])
        return self

    def forward(self, input: numpy.ndarray) -> numpy.ndarray:
        self.postActivation = []
        self.preActivation = []
        for layer in self.layers:
            self.preActivation.append(input)
            input = layer.compute(input)
            self.postActivation.append(input)
        return input

    def backward(self, target: numpy.ndarray):
        # Compute gradient at the output layer
        self.gradients = [numpy.zeros(layer.width) for layer in layers]

        self.gradients[-1] = (
            (numpy.array(self.postActivation[-1]) - numpy.array(target)) *
            (numpy.array(self.postActivation[-1]) * (1 - numpy.array(self.postActivation[-1])))
        )

        # Backpropagate through the layers
        for i in range(self.layers.__len__() - 2, -1, -1):
            self.gradients[i] = (
                numpy.dot(self.layers[i + 1].weights.T, self.gradients[i + 1]) *
                Util.deriveList(self.layers[i + 1].activation.function, self.preActivation[i + 1])
            )

    def update(self):
        for i, layer in enumerate(self.layers):
            layer.weights -= self.learningRate * numpy.outer(self.gradients[i], self.preActivation[i])
            layer.biases -= self.learningRate * self.gradients[i]

    def epoch(self, input: list, target: list):
        self.forward(input)
        self.backward(target)
        self.update()

    def save(self, path: str = PICKLE_SAFE_NAME) -> bool:
        with open(path, "wb") as file:
            try:
                dill.dump(self, file)
                return True
            except:
                return False
    
    @staticmethod
    def load(path: str = PICKLE_SAFE_NAME):
        with open(path, "rb") as file:
            return dill.load(file, True)

if __name__ == "__main__":
    import pandas, time
    layers = [
        Layer(86, 43, Activation(Function.RELU)),
        Layer(48, 86, Activation(Function.RELU)),
        Layer(7, 48, Activation(Function.SOFTMAX))
    ]
    model = Model(layers)
    model.learningRate = 0.001

    trainDataFrame = pandas.read_csv("cSharp ai trainer/datasetGen/datasetGen/bin/Debug/net9.0/exported.csv")
    MODEL_SAVE_LOCATION = "csharptrained.pickle"

    START_START_TIME = time.time()
    for _ in range(100):
        START_TIME = time.time()
        for i, row in trainDataFrame.iterrows():
            if i == 0:
                continue
            rowEntry = row.values
            input = rowEntry[0:-1]
            bestCol = rowEntry[-1]

            # Generate target output
            # the best column gets a high value, the rest random low values
            target = numpy.random.rand(7)
            target[bestCol] += 25
            target = Function.SOFTMAX(target)


            model.epoch(input, target)
            if i%10_000 == 0:
                print(f"Epoching row {i}")
                model.save(MODEL_SAVE_LOCATION)
                print(f"Created snapshot of model at {MODEL_SAVE_LOCATION}; took {time.time() - START_TIME}")
        model.save(MODEL_SAVE_LOCATION)
        print(f"Saved model at {MODEL_SAVE_LOCATION}; took {time.time() - START_START_TIME}")


#   layers = [
#       Layer(5, 3, Activation(Function.PASS)),
#       Layer(3, 5, Activation(Function.SOFTMAX))
#   ]
#   
#   model: Model = Model(layers=layers)
#   
#   for _ in range(10):
#       model.forward([2, 2, 0])
#       model.backward([0.5, 0.75, 0.1])
#       model.update()
#       for layer in model.layers:
#           print(*layer.weights, sep="\t")
#       print("-------------------------------------------------")
#   
#   model.forward([2, 2, 0])
#   print(model.layers[-1].output)

#   import numpy
#   import pickle
#   from typing import Callable
#   from enum import Enum
#   
#   class Util:
#       def relu(input: list[float]) -> list[float]:
#           return numpy.array(list(map(lambda x:max(x, 0.0), input)))
#       
#       def softmax(input: list[float]) -> list[float]:
#           expValues = numpy.exp(input)
#           return expValues / sum(expValues)
#   
#       def passF(input: list[float]) -> list[float]:
#           return numpy.array(input)
#       
#       #   [[list[float],], list[float]]
#       def deriveList(function: Callable, *inputs: list[float], dx = 1/100_000):
#           deriveArg, otherArgs = inputs
#           print(deriveArg, otherArgs)
#           dxList = function(numpy.array(deriveArg) + dx)
#           xList = function(deriveArg)
#           deltaYs = numpy.array(dxList) - numpy.array(xList)
#           return deltaYs / dx
#   
#   class Function(Enum):
#       RELU = Util.relu
#       SOFTMAX = Util.softmax
#       PASS = Util.passF
#   
#   class LossFunction(Enum):
#       MEAN_SQUARED_ERROR = 1
#   
#   class Activation:    
#       def __init__(self, function: Function = Function.PASS):
#           self.function = function
#           """ def relu(input: list[float]) -> list[float]:
#               return list(map(lambda x:max(x, 0.0), input))
#           
#           def softmax(input: list[float]) -> list[float]:
#               expValues = numpy.exp(input)
#               return (expValues / sum(expValues)).tolist()
#   
#           def passF(input: list[float]) -> list[float]:
#               return input
#           
#           funDict = {
#               "none": passF,
#               "relu": relu,
#               "softmax": softmax
#           }
#   
#           self.type = type
#           try:
#               self.function: Callable[[list[float]], float] = funDict[type]
#           except:
#               self.function: Callable[[list[float]], float] = funDict["none"] """
#       
#       def compute(self, input: list[float] = []) -> list[float]:
#           return self.function(input)
#   
#   class Layer:
#       def __init__(self, width: int = 0, prevLayerWidth: int = 0, activation: Activation = None, weights: list[list[float]] = []):
#           self.width = width
#           self.weights = numpy.random.randn(width, prevLayerWidth)
#           self.biases = numpy.random.randn(width)
#           self.activation = activation
#           self.output = numpy.zeros(width)
#   
#       def compute(self, input) -> list[float]:
#           for i in range(self.width):
#               self.output[i] = numpy.dot(input, self.weights[i]) + self.biases[i]
#           if self.activation != None:
#               self.output = self.activation.compute(self.output)
#           return self.output
#   
#   class Model:
#       PICKLE_SAFE_NAME = "model.pickle"
#   
#       def __init__(self, layers: list[Layer] = []):
#           self.layers: list[Layer] = []
#           self.postActivation: list[list[float]] = []
#           self.preActivation: list[list[float]] = []
#           self.gradients: list[list[float]] = []
#           self.input: list[float] = []
#           self.learningRate: float = 0.01
#   
#       def addLayer(self, layer: Layer):
#           self.layers.append(layer)
#           return self
#   
#       def forward(self, input: list) -> list[float]:
#           self.postActivation = numpy.zeros(self.layers.__len__())
#           self.preActivation = numpy.zeros(self.layers.__len__())
#           self.gradients = numpy.zeros(self.layers.__len__())
#           for i in range(self.layers.__len__()):
#               self.postActivation[i] = self.layers[i].compute(input)
#               input = self.postActivation[i]
#           return input
#   
#       #   loss: Callable[[list[float, float]], float] = lambda x, y: (x-y)**2
#       def backward(self, target: list[float]):
#           self.gradients[-1] = (
#               (numpy.array(self.postActivation[-1]) - numpy.array(target))
#               * numpy.array(self.postActivation[-1])
#               * (1 - numpy.array(self.postActivation[-1])))
#           for i in range(self.layers.__len__() - 2, -1, -1):
#               self.gradients[i] = (
#                   numpy.dot(self.layers[i+1].weights, self.gradients[i + 1])
#                   * Util.deriveList(self.preActivation[i])
#               )
#   
#       def save(self, path: str = PICKLE_SAFE_NAME) -> bool:
#           with open(path, "wb") as file:
#               try:
#                   pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
#                   return True
#               except:
#                   return False
#       
#       def load(path: str = PICKLE_SAFE_NAME):
#           with open(path, "rb") as file:
#               return pickle.load(file)

#   print(Activation().compute(Layer(1, 3).weigths[0]))

#   softmaxinp = numpy.random.randn(4)
#   softmaxres = Activation("softmax").compute(softmaxinp)
#   print(softmaxinp, softmaxres, sum(softmaxres), sep="\n")

#   activation = Activation("relu")
#   ranFloatArr = numpy.random.randn(4).tolist()
#   print(activation.getPointsSlope(activation.function, ranFloatArr), ranFloatArr, activation.compute(ranFloatArr), sep="\n")

#   for i in range(5 - 1, -1, -1):
#       print(i)

#   """ model = Model()
#   model.addLayer(Layer(5, 3, Activation(Function.PASS)))
#   model.addLayer(Layer(4, 5, Activation(Function.SOFTMAX)))
#   
#   model.save() """
#   
#   """ model = Model.load()
#   
#   model.forward([1, 2, 0])
#   for i in range(model.layerResults.__len__()):
#       print(*model.layerResults[i], sep="\t")
#   print(sum(model.layerResults[-1])) """

#   layers = [
#       Layer(5, 3, Activation(Function.PASS)),
#       Layer(4, 5, Activation(Function.SOFTMAX))
#   ]
#   
#   model: Model = Model(layers=layers)
#   
#   model.forward([1, 2, 0])
#   model.backward([1, 2, 0])