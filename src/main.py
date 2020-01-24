import time

from castle import CASTLE

def handler(value):
    print("RECIEVED VALUE: {}".format(value))

if __name__ == "__main__":
    headers = ('Value',)
    stream = CASTLE(handler, headers)

    for i in range(10):
        stream.insert(i)
        time.sleep(0.5)
