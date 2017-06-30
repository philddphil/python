import sys


def Cat(filename):
    f = open(filename, 'rU')
    lines = f.readlines()
    print(lines)

    f.close()


# Define a main() function that prints a little greeting
def main():
    Cat(sys.argv[1])


main()
