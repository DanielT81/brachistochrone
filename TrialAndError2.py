x = 5
def func():
    global x
    x += 10
    return 'you suck'


print(x)
print(func(), '\n', x)