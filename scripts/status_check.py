from dronekit import connect
import sys

udp_list = [14555, 15550]
num_drones = len(udp_list)
status_list = [False]*num_drones

for n in range(num_drones):
# Connect to UDP endpoint (and wait for default attributes to accumulate)
    target = sys.argv[1] if len(sys.argv) >= 2 else 'udpin:0.0.0.0:'+str(udp_list[n])
    print 'Connecting to ' + target + '...'
    vehicle = connect(target, wait_ready=True)

    status_list[n] = True

    # Get all vehicle attributes (state)
    print "Vehicle state:"
    print " Global Location: %s" % vehicle.location.global_frame
    print " Global Location (relative altitude): %s" % vehicle.location.global_relative_frame
    print " Local Location: %s" % vehicle.location.local_frame
    print " Attitude: %s" % vehicle.attitude
    print " Velocity: %s" % vehicle.velocity
    print " Battery: %s" % vehicle.battery
    print " Last Heartbeat: %s" % vehicle.last_heartbeat
    print " Heading: %s" % vehicle.heading
    print " Groundspeed: %s" % vehicle.groundspeed
    print " Airspeed: %s" % vehicle.airspeed
    print " Mode: %s" % vehicle.mode.name
    print " Is Armable?: %s" % vehicle.is_armable
    print " Armed: %s" % vehicle.armed

    vehicle.close()
    print "Done."
    print


print
print

for n in range(num_drones):
    print "drone %s is connected = %s" % (udp_list[n], status_list[n])
