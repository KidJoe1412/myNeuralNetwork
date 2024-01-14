import math
import numpy.random

class NeuralNetwork:
    def __init__(self,inputNodes,hiddenNodes,outputNodes,learningRate):
        # 输入层神经元个数
        self.inputNodes = inputNodes
        # 隐藏层神经元个数
        self.hiddenNodes = hiddenNodes
        # 输出层神经元个数
        self.outputNodes = outputNodes
        # 输入层到隐藏层的权重矩阵，numpy.random.rand 生成0到1的随机数
        self.weightInputHidden = (numpy.random.rand(self.hiddenNodes,self.inputNodes) - 0.5)
        # 隐藏层到输出层的权重矩阵
        self.weightHiddenOutput = (numpy.random.rand(self.outputNodes,self.hiddenNodes) - 0.5)
        # 学习率
        self.learningRate = learningRate
        # sigmoid函数
        self.activationFunction = lambda x: 1.0 / (1.0 + math.exp(-x))
        pass

    def train(self,inputsList,targetsList):
        inputs = numpy.array(inputsList,ndmin=2).T
        targets = numpy.array(targetsList,ndmin=2).T
        # 正向计算
        hidden_inputs = numpy.dot(self.weightInputHidden, inputs)
        hidden_outputs = self.activationFunction(hidden_inputs)
        final_inputs = numpy.dot(self.weightHiddenOutput, hidden_outputs)
        final_outputs = self.activationFunction(final_inputs)
        # 误差：目标值-实际值
        output_errors = targets - final_outputs
        # 误差反向传播
        hidden_errors = numpy.dot(self.weightHiddenOutput.T,output_errors)
        self.weightHiddenOutput += self.learningRate * \
                                   numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        self.weightInputHidden += self.learningRate * \
                                  numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),numpy.transpose(inputs))
        pass

    def query(self,inputsList):
        inputs = numpy.array(inputsList,ndmin=2).T
        hidden_inputs = numpy.dot(self.weightInputHidden,inputs)
        hidden_outputs = self.activationFunction(hidden_inputs)
        final_inputs = numpy.dot(self.weightHiddenOutput,hidden_outputs)
        final_outputs = self.activationFunction(final_inputs)
        return final_outputs

inputNodes = 3
hiddenNodes = 3
outputNodes = 3
learningRate = 0.3
n = NeuralNetwork(inputNodes,hiddenNodes,outputNodes,learningRate)
