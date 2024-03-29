#!/usr/bin/env python

import os, sys
import socket
import rospy

mpout_IP = "mpserver01.flightgear.org"  # FG Multiplayer Server Address
mpout_port = 5000  # Port over which instance communicates with FG multiplayer server


# Calling this function with the server id and instance value creates a dictionary containing information on the
# instance
def getid(server_no, instance):
    # Create a dictionary to hold information on the communicating ports for each FG instance
    Instance_info = dict()

    # Create a unique callsign and hold in dictionary
    callsign = 'MAGEF' + str(server_no) + str(instance).zfill(1)
    Instance_info['callsign'] = callsign

    # unique mavlink port to send information to QGC
    mavlink_port = '9' + str(server_no) + str(instance).zfill(2)
    Instance_info['mavlink_port'] = int(mavlink_port)

    # unique port over which controller and FG exchange command data
    command_port = '4' + str(server_no) + '0' + str(instance).zfill(2)
    Instance_info['command_port'] = int(command_port)

    # unique port over which controller and FG exchange sensor data
    sensor_port = '4' + str(server_no) + '1' + str(instance).zfill(2)
    Instance_info['sensor_port'] = int(sensor_port)

    # Unique port assignment for reading multiplayer data from FG multiplay server
    multiplay_portin = '8' + str(server_no) + str(instance).zfill(2)
    Instance_info['mpin_port'] = int(multiplay_portin)

    return Instance_info


# Launch FG with desired parameters
def FGthread(server_id, instance, controller_hostIP, control_input_config, freq_in=100, freq_out=100, vehicle='ufo', lat=37.67, lon=-122.35,
             alt=2000, iheading=45, ivel=60, ithrottle=0.001):
    # Parse the required information to call FG instance
    Instance_info = getid(server_id, instance)
    callsign = Instance_info['callsign']
    command_port = Instance_info['command_port']
    sensor_port = Instance_info['sensor_port']
    mpin_port = Instance_info['mpin_port']
    mavlink_port = Instance_info['mavlink_port']

    # Ping google.com to obtain host computer IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("google.com", 80))
        mpin_IP = s.getsockname()[0]
        s.close()
    except:
        print "Network connection unavailable..."
        sys.exit(-1)

    # Construct the command line parameter script to launch FG
    root = "$HOME"  # os.getcwd()

    #Fgpath = root + '/flightgear/bin/fgfs --fg-root=' + root + '/flightgear/data'
    Fgpath = root + '/flightgear/bin/fgfs --fg-root=' + root + '/flightgear/data'

    fgcallsign = '--callsign=' + callsign

    sock_out = '--generic=socket,out,' + str(freq_out) + ',' + str(controller_hostIP) + ',' + str(
        sensor_port) + ',udp,ControlOutputMage' + ' ' + '--generic=socket,out,' + str(freq_out) + ',localhost,' + str(
        mavlink_port) + ',udp,MAVOutput'
    sock_in = '--generic=socket,in,' + str(freq_in) + ',' + str(mpin_IP) + ',' + str(
        command_port) + ',udp,' + control_input_config
    SOCK = sock_out + ' ' + sock_in

    multiplay_out = '--multiplay=out,' + str(freq_out) + ',' + mpout_IP + ',' + str(mpout_port)
    multiplay_in = '--multiplay=in,' + str(freq_in) + ',' + mpin_IP + ',' + str(mpin_port)
    MULTIPLAY = multiplay_out + ' ' + multiplay_in

    iposition = '--lon=' + str(lon) + ' ' + '--lat=' + str(lat) + ' ' + '--altitude=' + str(
        alt) + ' ' + '--heading=' + str(iheading) + ' ' + '--vc=' + str(ivel)

    aircraft = '--aircraft=' + vehicle

    externals = '--units-meters --wind=0@0 --turbulence=0.0 --timeofday=noon --disable-random-objects ' \
                '--disable-ai-models --disable-clouds3d --disable-clouds --runway=28L --geometry=400x250 ' \
                '--prop:/sim/rendering/multithreading-mode=AutomaticSelection --disable-sound ' \
                '--prop:/controls/engines/engine/starter=true --fog-disable --disable-horizon-effect ' \
                '--prop:/controls/engines/engine/throttle=' + str(ithrottle) +' '\
                '--disable-specular-highlight --prop:input/joysticks/js=0'
    freeze = '--disable-clock-freeze'
    # geometry=700x400
    FGcmd = Fgpath + ' ' + fgcallsign + ' ' + SOCK + ' ' + MULTIPLAY + ' ' + iposition + ' ' + aircraft + ' ' + externals + ' ' + freeze
    print FGcmd
    rospy.init_node('fg_{}'.format(instance))
    os.system(FGcmd)
    rospy.spin()

