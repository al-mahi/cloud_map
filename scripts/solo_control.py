#!/usr/bin/env python

"""
This is vital code for running ros node representing the Solo drone.
author: James Kostas
modified by: S M Al Mahi
"""

import rospy
from std_msgs.msg import String
from dronekit import connect, VehicleMode, LocationGlobalRelative, LocationLocal
import sys
import time
import cPickle

port_list = ["14550", "15550"]  # the ports of the vehicles we are controlling
veh_list = []  # holds the vehicle objects


def shifted_position(lat, long, NS, EW):  # takes a lat/long and a North/South and East/West
    # displacement in meters (N and E are positive, S and W are negative), and outputs the new lat/long

    # CAUTION: the numbers below are set for use near Stillwater.  They will change
    # at other lattitudes
    meters_per_lat = 110961.03  # meters per degree of latitude
    meters_per_long = 90037.25  # meters per degree of latitude
    return (lat + 1. * NS / meters_per_lat, long + 1. * EW / meters_per_long)


def arm_and_takeoff(veh_num, aTargetAltitude):
    """
    Arms vehicle and fly_grad to aTargetAltitude (in meters).
    """
    print
    print "vehicle %s:" % (veh_num)
    print "Basic pre-arm checks"
    # Don't try to arm until autopilot is ready
    while not veh_list[veh_num].is_armable:
        print " Waiting for vehicle to initialise..."
        time.sleep(1)

    print "Arming motors"
    # Copter should arm in GUIDED mode
    veh_list[veh_num].mode = VehicleMode("GUIDED")
    veh_list[veh_num].armed = True

    # Confirm vehicle armed before attempting to take off
    while not veh_list[veh_num].armed:
        print " Waiting for arming..."
        time.sleep(1)

    alt = aTargetAltitude
    print "Taking off to %s!" % alt
    veh_list[veh_num].simple_takeoff(alt)  # Take off to target altitude
    print


def land_and_disconnect(veh_num):
    print
    print "vehicle %s:" % (veh_num)
    print "Returning to Launch"
    veh_list[veh_num].mode = VehicleMode("RTL")

    # Close vehicle object before exiting script
    print "Close vehicle object %s" % veh_num
    veh_list[veh_num].close()
    print


def goto(veh_num, lat, long, alt):
    print "vehicle %s going to %s, %s, %s meters" % (veh_num, lat, long, alt)
    veh_list[veh_num].simple_goto(LocationGlobalRelative(lat, long, alt))


# def condition_yaw(veh_num, heading, relative=False): #sets yaw? need to test. see notes for copy-pasted documentation
#    if relative:
#        is_relative = 1  # yaw relative to direction of travel
#    else:
#        is_relative = 0  # yaw is an absolute angle
#    # create the CONDITION_YAW command using command_long_encode()
#    msg = veh_list[veh_num].message_factory.command_long_encode(
#        0, 0,  # target system, target component
#        mavutil.mavlink.MAV_CMD_CONDITION_YAW,  # command
#        0,  # confirmation
#        heading,  # param 1, yaw in degrees
#        0,  # param 2, yaw speed deg/s
#        1,  # param 3, direction -1 ccw, 1 cw
#        is_relative,  # param 4, relative offset 1, absolute angle 0
#        0, 0, 0)  # param 5 ~ 7 not used
#    # send command to vehicle
#    veh_list[veh_num].send_mavlink(msg)

def msg_handle(data):
    loaded = cPickle.loads(str(data)[6:])  # converts to normal string, excludes the "data: ", then unpickles
    # the 0th element of the list is a string describing the command
    # the 1st is the drone number (eg 0 is 14550, 1 is 15550, etc...)
    # elements after that are specific to the command type
    the_veh = loaded[1]  # vehicle number
    if loaded[0] == "takeoff":
        arm_and_takeoff(the_veh, loaded[2])
    elif loaded[0] == "land":
        land_and_disconnect(the_veh)
    elif loaded[0] == "goto":
        lat = loaded[2]
        long = loaded[3]
        alt = loaded[4]
        goto(the_veh, lat, long, alt)
    # elif loaded[0] == "yaw":  # experimental, does not work yet!
    #     condition_yaw(the_veh, loaded[2])
    else:
        print "received invalid message %s" % loaded


def solo_control():
    rospy.init_node('solo_control', anonymous=True)

    rospy.Subscriber('control_topic', String, msg_handle)

    for port in port_list:
        # Connect to UDP endpoint (and wait for default attributes to accumulate)
        target = sys.argv[1] if len(sys.argv) >= 2 else 'udpin:0.0.0.0:' + port

        print 'Connecting to ' + target + '...'
        veh_list.append(connect(target, wait_ready=True))

    num_vehicles = len(veh_list)
    print "%s vehicles ready" % (num_vehicles)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    solo_control()
