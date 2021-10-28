#!/usr/bin python3
data = input("this will come from STDIN: ")
print("now we write it to STDOUT: " + data)
raise ValueError("now we generate an error to STDERR")