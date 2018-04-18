##############################################################################
# Import some libraries
##############################################################################

import socket
import re
import numpy as np
import time
import os
import glob
import matplotlib.pyplot as plt
import useful_defs_prd as prd

from struct import *
from scipy.interpolate import interp1d
from datetime import datetime

##############################################################################
# Do some stuff
##############################################################################

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('localhost', 8089))
server.listen(1)
print('Server Listening')

while True:
    conn, addr = server.accept()
    cmnd = conn.recv(16384)
    call_time = time.ctime()

    if 'INIT' in str(cmnd):
        # Initialisation. Runs the fit_phase function and returns the fitted
        # ϕ_min and ϕ_max values. These come from the power measured when the
        # phase is ramped in a binary grating
        print(call_time + ' INIT')

        hol_data = str(cmnd)
        LabVIEW_data = [float(i1)
                        for i1 in re.findall(r'[-+]?\d+[\.]?\d*', hol_data)]
        variables = re.findall(r'\s(\D*)', hol_data)
        ϕ_g = prd.fit_phase()
        ϕ_max = max(ϕ_g) / np.pi
        ϕ_min = min(ϕ_g) / np.pi

        ϕ_min = (str(round(ϕ_min, 6))).zfill(10)
        ϕ_max = (str(round(ϕ_max, 6))).zfill(10)
        data_out = ϕ_min + ',' + ϕ_max
        conn.sendall(bytes(data_out, 'utf-8'))

    elif 'GEN' in str(cmnd):
        # Generate a hologram and save it as a bmp
        print('GEN')
        t1 = 1000 * time.time()
        hol_data = str(cmnd)
        LabVIEW_data = [float(i1)
                        for i1 in re.findall(r'[-+]?\d+[\.]?\d*', hol_data)]
        variables = re.findall(r'\s(\D*)', hol_data)
        values = prd.variable_unpack(LabVIEW_data)
        hol_values = dict(zip(variables, values))
        # print(hol_values)
        H = prd.holo_gen(*LabVIEW_data)
        conn.sendall(b'PLAY-DONE')
        t2 = 1000 * time.time()
        print('display total time =', int(t2 - t1))

    elif 'FINDp' in str(cmnd):
        # Finds the optimum rotation angle a hologram needs from the data
        # sent over by labVIEW
        print('FIND ϕ')
        find_data = str(cmnd)

        ϕP = find_data.split(',')
        ϕs = ϕP[1].split('\t')
        Ps = ϕP[2].split('\t')
        print(Ps)
        ϕs = [float(i1) for i1 in re.findall(r'[-+]?\d+[\.]?\d*', ϕP[1])]
        Ps = [float(i1) for i1 in re.findall(r'[-+]?\d+[\.]?\d*', ϕP[2])]
        Ps = np.array(Ps)
        ϕs = np.array(ϕs)
        Ps = np.power(10, Ps / 10)

        ϕ = prd.find_fit_peak((np.pi / 180) * ϕs, Ps, max(Ps), 0.1)
        ϕ = (str(round(ϕ * 180 / np.pi, 6))).zfill(10)
        data_out = ϕ
        print('ϕ = ', ϕ)
        conn.sendall(bytes(data_out, 'utf-8'))

    elif 'FINDL' in str(cmnd):
        # Finds the optimum period a hologram needs from the data sent over
        # by labVIEW
        print('FIND Λ')
        find_data = str(cmnd)
        ΛP = find_data.split(',')

        Λs = ΛP[1].split('\t')
        Ps = ΛP[2].split('\t')
        print(Ps)
        Λs = [float(i1) for i1 in re.findall(r'[-+]?\d+[\.]?\d*', ΛP[1])]
        Ps = [float(i1) for i1 in re.findall(r'[-+]?\d+[\.]?\d*', ΛP[2])]
        Ps = np.array(Ps)
        Λs = np.array(Λs)
        Ps = np.power(10, Ps / 10)

        Λ = prd.find_fit_peak(Λs, Ps, max(Ps), 1)
        Λ = (str(round(Λ, 6))).zfill(10)
        data_out = Λ
        print('Λ = ', Λ)
        conn.sendall(bytes(data_out, 'utf-8'))

    elif 'SAVE' in str(cmnd):
        # Saves the current hologram in 2 locations - in the sub directory
        # 'prior positions' with a date stamp, and also in the calibration
        # folder with the fibre number in the file name
        print('SAVE')
        save_data = str(cmnd)
        port_data = [float(i1)
                     for i1 in re.findall(r'[-+]?\d+[\.]?\d*', save_data)]
        variables = re.findall(r'\s(\D*)', save_data)
        filename = variables[-1] + str(int(port_data[-1])) + '.csv'
        s1 = r'..\..\Data\Calibration files\prior positions'
        s2 = r'..\..\Data\Calibration files'
        f1 = time.strftime("%Y%m%d-%H%M%S") + filename
        f2 = filename
        p1 = os.path.join(s1, f1)
        p2 = os.path.join(s2, f2)
        np.savetxt(p1, port_data, delimiter=",",
                   header='see code structure for variable names')
        np.savetxt(p2, port_data, delimiter=",",
                   header='see code structure for variable names')
        print(port_data)
        conn.sendall(b'SAVE-DONE')

    elif 'READ' in str(cmnd):
        # Read the hologram specified by the fibre number sent over by labVIEW
        # At the minute it's set to read fibres denoted by numbers < 10, 
        # Fibre #99 is a special case and is the result of an anneal
        read_data = str(cmnd)
        try:
            fibre = [float(i1)
                     for i1 in re.findall(r'[-+]?\d+[\.]?\d*', read_data)]
            fibre = int(fibre[0])
            if fibre < 10:
                filename = 'fibre' + str(fibre) + '.csv'
                s1 = r'..\..\Data\Calibration files'
                p1 = os.path.join(s1, filename)
            else:
                p1 = r"..\..\Data\Python loops\Anneal Hol params keep.txt"
            my_data = np.genfromtxt(p1, delimiter=',')
            data_out = ''
            for i1 in np.ndenumerate(my_data[0:-1]):
                elem = (str(round(i1[1], 6))).zfill(10)
                data_out = data_out + ',' + elem
            data_out = data_out[1:]
            print('loaded last ' + 'fibre ' + str(fibre))

        # If there is an index error in when loading up the fibre csv, if 
        # switches to some saved 'default positions'
        except IndexError:
            fibre = [float(i1)
                     for i1 in re.findall(r'[-+]?\d+[\.]?\d*', read_data)]
            fibre = int(fibre[0])
            filename = 'fibre' + str(fibre) + '.csv'
            s1 = r'..\..\Data\Calibration files\default positions'
            p1 = os.path.join(s1, filename)
            my_data = np.genfromtxt(p1, delimiter=',')
            data_out = ''
            for i1 in np.ndenumerate(my_data[0:-1]):
                elem = (str(round(i1[1], 6))).zfill(10)
                data_out = data_out + ',' + elem
            data_out = data_out[1:]
            print('loaded default ' + 'fibre ' + str(fibre))

        conn.sendall(bytes(data_out, 'utf-8'))

    elif 'BINA' in str(cmnd):
        # Change the gray levels s.t. the current hologram is binary
        print('BINA')
        hol_data = str(cmnd)
        LabVIEW_data = [float(i1)
                        for i1 in re.findall(r'[-+]?\d+[\.]?\d*', hol_data)]
        variables = re.findall(r'\s(\D*)', hol_data)
        values = prd.variable_unpack(LabVIEW_data)
        ϕ_g = prd.fit_phase()
        g_ϕ = interp1d(ϕ_g, np.linspace(0, 255, 256))
        ϕ_range = (values[11] - values[10])

        values[14] = 0
        values[15] = 0
        values[12] = ϕ_range / 2 + 0.1
        values[13] = ϕ_range / 2 + 0.1
        print(values)
        prd.holo_gen(*values)

        os_lw_str = (str(round(values[12], 6))).zfill(10)
        os_up_str = (str(round(values[13], 6))).zfill(10)
        osw_lw_str = (str(round(values[14], 6))).zfill(10)
        osw_up_str = (str(round(values[15], 6))).zfill(10)

        data_out = (os_lw_str + ',' + os_up_str +
                    ',' + osw_lw_str + ',' + osw_up_str)
        conn.sendall(bytes(data_out, 'utf-8'))

    elif 'PICO' in str(cmnd):
        # Reads the picoscope data sent by labVIEW
        print('PICO-DATA_IN')
        pico_data = str(cmnd)
        Ps = [float(i1) for i1 in re.findall(r'[-+]?\d+[\.]?\d*', pico_data)]
        f2 = datetime.utcnow().strftime('%Y-%m-%d %H-%M-%S.%f')[:-3]
        f1 = r'..\..\Data\Pico Log'
        p1 = os.path.join(f1, f2)
        np.savetxt(p1 + '.csv', Ps, delimiter=",")
        conn.sendall(b'PICO-DONE')

    elif 'BOTHP' in str(cmnd):
        # Reads both the CT400 power and the picoscope power sent by labVIEW
        print('BOTHP')
        Ps_p = r'..\..\Data\Calibration files\Ps_last.csv'

        Ps_last = np.genfromtxt(Ps_p, delimiter=',')
        CT400_last = Ps_last[0]
        PicoL_last = Ps_last[1]

        Ps_LabVIEW = str(cmnd)
        Ps_current = [float(i1) for i1 in re.findall(
            r'[-+]?\d+[\.]?\d*', Ps_LabVIEW)]
        CT400_current = Ps_current[0]
        PicoL_current = Ps_current[1]

        np.savetxt(Ps_p, np.array(Ps_current), delimiter=',')
        conn.sendall(bytes('BOTHP-DONE', 'utf-8'))

    elif 'LOCBEAM' in str(cmnd):
        print('LOCBEAM')
        # Locates the beam by running a binary search algorithm
        data_in = str(cmnd)
        sts = data_in.split(',')
        ynotx = " ".join(re.findall("[a-zA-Z]+", sts[1]))
        if 'TRUE' in ynotx:
            axis = 0
        elif 'FALSE' in ynotx:
            axis = 1
        loop_out = prd.locate_beam(values, CT400_last, CT400_current, axis)
        data_out = (str(round(loop_out, 6))).zfill(10)
        current_hol = np.array(values)

        for i1 in np.ndenumerate(current_hol[0:]):
            elem = (str(round(i1[1], 6))).zfill(10)
            data_out = data_out + ',' + elem

        conn.sendall(bytes(str(data_out), 'utf-8'))

    elif 'ANNEAL' in str(cmnd):
        # Runs the annealing function (see anneal_H for details)
        print('ANNEAL')

        loop_out, values = prd.anneal_H(values, Ps_current, variables)
        data_in = str(cmnd)
        data_out = (str(round(loop_out, 6))).zfill(10)
        conn.sendall(bytes(str(data_out), 'utf-8'))

    elif 'SWEEP' in str(cmnd):
        # Sweeps a set of parameters for optimisation of the merit function
        print('SWEEP')
        data_in = [float(i1)
                   for i1 in re.findall(r'[-+]?\d+[\.]?\d*', str(cmnd))]

        loop_out, data_out = prd.sweep_multi(
            data_in, values, Ps_current, variables)
        if loop_out == 1:
            print('sweep over')
        conn.sendall(bytes(str(data_out), 'utf-8'))

    elif 'QUIT' in str(cmnd):
        # Closes the server program
        print(call_time + ' QUIT')
        conn.sendall(b'QUIT-DONE')
        break

server.close()
