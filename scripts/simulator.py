#!/usr/bin/env python

"""
Module of libraries to interface between flightgear and ros
created by Rakshit Allamraju
"""


from __future__ import print_function
# import standard libraries
import os, sys
import socket

# Ros libraries
# import rospy


class Simulator:
    def __init__(self, sock_params):
        """
        @type sock_params: dict()
        """
        self._FG_IP = sock_params.get('IP')
        self._Port_send = sock_params.get('port_send')
        self._Port_recv = sock_params.get('port_recv')

        # Ping google.com to obtain host computer IP
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("google.com", 80))
            SYS_IP = s.getsockname()[0]
            s.close()
        except:
            print("Network connection unavailable...")
            sys.exit(-1)

        # Create a socket for sending data to FG
        self._FGSock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Create a socket to recv data from FG
        self._FGSock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # reusing socket
        self._FGSock_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 3)
        self._FGSock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 3)

        self._FGSock_recv.bind((self._FG_IP, self._Port_recv))  # bind socket to recv data

    def FGSend(self, commands, for_model):
        """
        :param commands: control command name and value in a dictionary
        :param for_model: quad or fixed_wing
        """
        # Parse command data from topic and generate UDP message
        # command_list = [str(Commands["aileron"]), str(Commands["elevator"]), str(Commands["rudder"]),str(Commands[
        # "throttle"]),str(GLOBAL_HOME_LAT),str(GLOBAL_HOME_LONG),str(GLOBAL_HOME_ALT)]
        # command_list = [str(commands["aileron"]), str(commands["elevator"]), str(commands["rudder"]),
        #                str(commands["throttle"]),
        #                str(commands["GLOBAL_HOME_LAT"]),
        #                str(commands["GLOBAL_HOME_LONG"]),
        #                str(commands["GLOBAL_HOME_ALT"])
        #                ]

        if for_model == 'quad':
            command_list = [
                "{}".format(commands["aileron"]),
                "{}".format(commands["elevator"]),
                "{}".format(commands["rudder"]),
                "{}".format(commands["throttle"]),
                "{}".format(commands["poslat"]),
                "{}".format(commands["poslon"]),
                "{}".format(commands["posalt"])
            ]
        elif for_model == 'ufo':
            command_list = [
                "{}".format(commands["aileron"]),
                "{}".format(commands["elevator"]),
                "{}".format(commands["rudder"]),
                "{}".format(commands["throttle"])
            ]
        else:
            raise Exception("simulator.py Unknown Model!!!")

        commands_msg = '\t'.join(command_list) + '\n'
        # print("{}".format(commands_msg))
        # Send data to FG via UDP
        self._FGSock_send.sendto(commands_msg, (self._FG_IP, self._Port_send))

    def FGRecv(self):
        # get the data from FG
        message, addr = self._FGSock_recv.recvfrom(1024)
        data = message.split('\t')
        fg_data = map(float, data)
        return fg_data
