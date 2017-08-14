##############################################################################
# Import some libraries
##############################################################################

import socket
import re
import numpy as np

##############################################################################
# Do some stuff
##############################################################################

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('localhost', 8089))
server.listen(1)

while True:
    conn, addr = server.accept()
    cmnd = conn.recv(4096)
    print(cmnd)

    if 'INIT' in str(cmnd):
        # Do initialisation
        conn.sendall(b'INIT-DONE')

    elif 'PLAY' in str(cmnd):
        # Do the play action
        data = str(cmnd)
        numbers = re.findall(r'[-+]?\d+[\.]?\d*', data)
        variables = re.findall(r'\s(\D*)', data)
        for i1 in range(len(numbers)):
            print(variables[i1], ' = ', numbers[i1])
        conn.sendall(b'PLAY-DONE')

    elif 'QUIT' in str(cmnd):
        # Do the quiting action
        conn.sendall(b'QUIT-DONE')
        break

server.close()
