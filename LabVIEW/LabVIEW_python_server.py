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
        print(call_time + ' INIT')
        # Do initialisation
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

    elif 'DISP' in str(cmnd):
        # Display a hologram
        print('DISP')
        # print(call_time)
        hol_data = str(cmnd)
        LabVIEW_data = [float(i1)
                        for i1 in re.findall(r'[-+]?\d+[\.]?\d*', hol_data)]
        variables = re.findall(r'\s(\D*)', hol_data)
        values = prd.variable_unpack(LabVIEW_data)
        hol_values = dict(zip(variables, values))
        H = prd.holo_gen(*LabVIEW_data)
        conn.sendall(b'PLAY-DONE')

    elif 'FINDp' in str(cmnd):
        # Do the play action
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
        # Do the play action
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
        # Change the gray levels s.t. the grating is binary
        print('BINA')
        hol_data = str(cmnd)
        LabVIEW_data = [float(i1)
                        for i1 in re.findall(r'[-+]?\d+[\.]?\d*', hol_data)]
        variables = re.findall(r'\s(\D*)', hol_data)
        values = prd.variable_unpack(LabVIEW_data)
        ϕ_g = prd.fit_phase()
        g_ϕ = interp1d(ϕ_g, range(255))
        g_mid = int(g_ϕ(np.pi * values[7] / 2))
        values[10] = g_mid + 1
        values[11] = g_mid
        prd.holo_gen(*values)
        g_OSlw = (str(round(values[10], 6))).zfill(10)
        g_OSup = (str(round(values[11], 6))).zfill(10)
        data_out = g_OSlw + ',' + g_OSup
        conn.sendall(bytes(data_out, 'utf-8'))

    elif 'PHASE' in str(cmnd):
        print('PHASE')
        save_data = str(cmnd)
        port_data = [float(i1)
                     for i1 in re.findall(r'[-+]?\d+[\.]?\d*', save_data)]
        files = glob.glob(
            r'C:\Users\User\Documents\Phils LabVIEW\Data\Pico Log\*.csv')
        total_data = []
        for i1 in files:
            temp_data = np.genfromtxt(i1, delimiter=',')
            print(np.shape(temp_data))
            total_data = np.concatenate((total_data, temp_data))
            os.remove(i1)

        plt.plot(total_data, color='xkcd:blue')
        s1 = r'..\..\Data\Calibration files\prior phaseramps'
        f1 = time.strftime(
            'wavelength-' + str(port_data[0]) + 'nm  time-%Y%m%d-%H%M%S.csv')
        f2 = time.strftime(
            'wavelength-' + str(port_data[0]) + 'nm  time-%Y%m%d-%H%M%S.png')
        p1 = os.path.join(s1, f1)
        p2 = os.path.join(s1, f2)
        np.savetxt(p1, total_data, delimiter=",")
        plt.savefig(p2)
        plt.cla()
        print('PHASE-DONE')
        conn.sendall(b'PHASE-DONE')

    elif 'PICO' in str(cmnd):
        print('PICO-DATA_IN')
        pico_data = str(cmnd)
        Ps = [float(i1) for i1 in re.findall(r'[-+]?\d+[\.]?\d*', pico_data)]
        f2 = datetime.utcnow().strftime('%Y-%m-%d %H-%M-%S.%f')[:-3]
        f1 = r'..\..\Data\Pico Log'
        p1 = os.path.join(f1, f2)
        np.savetxt(p1 + '.csv', Ps, delimiter=",")
        conn.sendall(b'PICO-DONE')

    elif 'BOTHP' in str(cmnd):
        print('BOTHP')
        Ps_p = r'..\..\Data\Calibration files\Ps_last.csv'

        Ps_last = np.genfromtxt(Ps_p, delimiter=',')
        CT400_last = Ps_last[0]
        PicoL_last = Ps_last[1]
        # print('LAST-Power')
        # print('+1 order = ' + str(np.round(CT400_last, 3)) + ' dB')
        # print('-1 order = ' + str(np.round(PicoL_last, 3)) + ' dB')
        # print('X-talk   = ' +
        #       str(np.round(np.abs(last_CT400 - last_PicoL), 3)) + ' dB')

        Ps_LabVIEW = str(cmnd)
        Ps_current = [float(i1) for i1 in re.findall(
            r'[-+]?\d+[\.]?\d*', Ps_LabVIEW)]
        CT400_current = Ps_current[0]
        PicoL_current = Ps_current[1]
        # print('CURRENT-Powers')
        # print('+1 order = ' + str(np.round(CT400_current, 3)) + ' dB')
        # print('-1 order = ' + str(np.round(PicoL_current, 3)) + ' dB')
        # print('X-talk   = ' +
        # str(np.round(np.abs(current_Ps[0] - current_Ps[1]), 3)) + ' dB')
        np.savetxt(Ps_p, np.array(Ps_current), delimiter=',')
        conn.sendall(bytes('BOTHP-DONE', 'utf-8'))

    elif 'LOCBEAM' in str(cmnd):
        print('LOCBEAM')
        data_in = str(cmnd)
        sts = data_in.split(',')
        ynotx = " ".join(re.findall("[a-zA-Z]+", sts[1]))
        if 'TRUE' in ynotx:
            axis = 0
        elif 'FALSE' in ynotx:
            axis = 1
        # N.B. 'values' comes from last displayed hologram
        # (see 'values' in DISP case)
        loop_out = prd.locate_beam(values, CT400_last, CT400_current, axis)
        data_out = (str(round(loop_out, 6))).zfill(10)
        current_hol = np.array(values)

        for i1 in np.ndenumerate(current_hol[0:]):
            elem = (str(round(i1[1], 6))).zfill(10)
            data_out = data_out + ',' + elem

        conn.sendall(bytes(str(data_out), 'utf-8'))

    elif 'ANNEAL' in str(cmnd):
        # Take hologram centre coords and create (100 x 100) px random H
        # N.B. 'values' comes from last displayed hologram
        # (see 'values' in DISP case)
        # 'last_Ps and current_Ps' come from previously executed BOTHP cases
        print('ANNEAL')
        # loop_out = prd.anneal_H1(values, Ps_last, Ps_current)
        # loop_out = prd.anneal_H2(values, Ps_last, Ps_current, np.squeeze(H))
        loop_out, values = prd.anneal_H3(values, Ps_current, variables)
        data_in = str(cmnd)
        data_out = (str(round(loop_out, 6))).zfill(10)
        conn.sendall(bytes(str(data_out), 'utf-8'))

    elif 'SWEEP' in str(cmnd):
        # Take hologram centre coords and create (100 x 100) px random H
        # N.B. 'values' comes from last displayed hologram
        # (see 'values' in DISP case)
        # 'last_Ps and current_Ps' come from previously executed BOTHP cases
        print('SWEEP')
        data_in = [float(i1)
                   for i1 in re.findall(r'[-+]?\d+[\.]?\d*', str(cmnd))]
        print('Data in = ', data_in)

        data_out = prd.sweep_multi(data_in, values, Ps_current, variables)

        conn.sendall(bytes(str(data_out), 'utf-8'))

    elif 'QUIT' in str(cmnd):
        # Do the quiting action
        print(call_time + ' QUIT')
        conn.sendall(b'QUIT-DONE')
        break

server.close()
