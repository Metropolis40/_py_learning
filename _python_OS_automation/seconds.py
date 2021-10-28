

#!/Users/taohuang/opt/miniconda3/envs/ds1/bin python

def to_seconds(hours, minutes, seconds):
    return hours*3600 + minutes*60 +seconds


print("Welcome to this time converter")

cont = "y"

while (cont.lower() == "y" ) :
    hours = int(input("Enter hours"))
    minutes = int(input("Enter minutes"))
    seconds = int(input("Enter seconds"))

    print("That is {} seconds".format(to_seconds(hours, minutes, seconds)))
    print()
    cont = input("another conversion [y to cont] ")
    print(cont) 

print("Good bye")



