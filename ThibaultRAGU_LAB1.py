# -*- coding: utf-8 -*-

#Exercise 1.4.1
#Function that prints a pattern

def f():
    for y in range(9):
        if y < 4:
            print((y+1) * "x")
        else:
            print((9-y)* "x")
            
#Exercise 1.4.2
#Function that sum all numbers in n45as29@#8ss6

def g():
    input_string = "n45as29@#8ss6"
    sum = 0
    for char in input_string:
        if char.isdigit():
            sum =  sum + int(char)
    print(sum)
    
#Exercise 1.4.3
#Function that converts an arbitary integer to binary number

def conversion(number):
    binary_result = ""
    if number == 0:
        return "0"
    while number > 0:
        binary_result = str(number % 2) + binary_result
        number = number // 2
    return binary_result
    
    
#Exercise 1.5-1
#Function that takes an integer as an input and return a list that contains 
#all fibonaci numbers smaller than input integer


def fibonaci(number):
    fib = [0, 1]
    while fib[-1] < number:
        next_fib = fib[-1] + fib[-2]
        if next_fib >= number:
            break
        fib.append(next_fib)
    return fib

#Exercise 1.5-2
#Function that prints an arbitary integer in segment display style
            
# def display_as_digi(number):
#     digi = [
#         [1, 1, 1, 0, 1, 1, 1], # 0
#         [0, 0, 1, 0, 0, 1, 0], # 1
#         [1, 0, 1, 1, 1, 0, 1], # 2
#         [1, 0, 1, 1, 0, 1, 1], # 3
#         [0, 1, 1, 1, 0, 1, 0], # 4
#         [1, 1, 0, 1, 0, 1, 1], # 5
#         [1, 1, 0, 1, 1, 1, 1], # 6
#         [1, 0, 1, 0, 0, 1, 0], # 7
#         [1, 1, 1, 1, 1, 1, 1], # 8
#         [1, 1, 1, 1, 0, 1, 1]  # 9
#     ]