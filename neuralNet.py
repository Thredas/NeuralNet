import numpy #Библиотека с массивами
import scipy.special #Библиотека с сигмоидой

class neuralNet():
    
    #В классах у функций первым аргументом должен быть аргумент self
    #Также перед переменными нужно писать self
    
    #Функция, выполняющаяся при создании объекта класса
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes 
        
        self.lr = learning_rate #Коэффицент обучения
        
        #Создание массивов (матриц) с рандомными весами,
        #размерностью равной заданными числом скрытых, входных и выходных узлов
        self.wih = (numpy.random.rand(self.h_nodes,self.i_nodes) - 0.5) #w - weights, i - input_nodes, h - hidden_nodes
        self.who = (numpy.random.rand(self.o_nodes,self.h_nodes) - 0.5) #w - weights, h - hidden_nodes, o - output_nodes
        
        #Это лямбда-выражение, позволяющее создавать функции на лету
        #По сути это функция activation_function(), которая принимает x и возвращает scipy.special.expit(x)
        #expit() - функция активации (cигмоида)
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass #Нужно писать, если функция ничего не возвращает
    
    def train(self, inputs_list, targets_list):
        
        inputs = numpy.array(inputs_list, ndmin = 2).T #Преобразовывает inputs_list в двумерный массив
        targets = numpy.array(targets_list, ndmin = 2).T
        
        hidden_inputs = numpy.dot(self.wih, inputs) #Умножение массива весов wih на массив inputs
        hidden_outputs = self.activation_function(hidden_inputs)#Применение сигмоиды для каждого элемента массива
        
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        errors = targets - final_outputs #Ошибки в выходном слое
        
        hidden_errors = numpy.dot(self.who.T, errors) #Ошибки в скрытом слое
        
        #Обновление весов между скрытым и выходным слоями
        self.who += self.lr * numpy.dot((errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        #Обновление весов между входным и скрытым слоями
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass
    
    def query(self, inputs_list):
        
        inputs = numpy.array(inputs_list, ndmin = 2).T 
        
        hidden_inputs = numpy.dot(self.wih, inputs) 
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
