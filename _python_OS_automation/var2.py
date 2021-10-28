


#!/usr/bin/env python3
# this is a test file ?
import os

# we can use a dictionary method to get access to environment variable

print("HOME: "+ os.environ.get("HOME", "")) 

# try to get the value associated with the key of "HOME", if the value does not exist, return empty value

print("SHELL: "+ os.environ.get("SHELL", ""))

print("FRUIT: "+ os.environ.get("FRUIT", ""))