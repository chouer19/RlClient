#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Keyboard controlling for CARLA. Please refer to client_example.py for a simpler
# and more documented example.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot

    R            : restart level

STARTING in a moment...
"""

from __future__ import print_function

import argparse
import logging
import random
import time
import math

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_l
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line


WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400

ROAD_WIDTH = 10.0
PIXEL_DENSITY = 10
START_MAP_X = 10
START_MAP_Y = 10
FPS = 25

CAR_X = 10
CAR_Y = int(WINDOW_HEIGHT/2/PIXEL_DENSITY)

CAR_RECT =  pygame.Rect(( int(WINDOW_WIDTH/4 - 2 * PIXEL_DENSITY),  int(WINDOW_HEIGHT/2- 1.3 * PIXEL_DENSITY)), \
                        ( int(5.5 * PIXEL_DENSITY), int(2.6 * PIXEL_DENSITY))\
                       )\


def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=False,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=0,
        NumberOfPedestrians=0,
        SeedVehicles = '00000',
        WeatherId=1,
        QualityLevel=args.quality_level)
    settings.randomize_seeds()
    return settings


class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


class CarlaGame(object):
    def __init__(self, carla_client, args):
        self.client = carla_client
        self._carla_settings = make_carla_settings(args)
        self._timer = None
        self._display = None
        self._enable_autopilot = args.autopilot
        self._map_view = None
        self._is_on_reverse = False
        self._city_name = args.map_name
        self._map = CarlaMap(self._city_name, 16.43, 50.0) if self._city_name is not None else None
        self._map_shape = self._map.map_image.shape if self._city_name is not None else None
        self._map_view = self._map.get_map(WINDOW_HEIGHT) if self._city_name is not None else None
        self._position = None
        self._location = None
        self._control = VehicleControl()
        self._road_left,self._road,self._road_right = self._load_road(args.road,args.road_length)

    def _load_road(self,folder,length):
        points = []
        points_left = []
        points_right = []

        def readfile(name):
            with open(name) as f:
                line = f.readline()
                while line:
                    contents = line.split(',')
                    if len(contents) == 3:
                        x,y,yaw = float(contents[0]),  float(contents[1]), float(contents[2])
                        #yaw += 180
                        #if yaw > 180:
                        #    yaw -= 360
                        points.append(( x,y,yaw ))
                        points_left.append(( x + ROAD_WIDTH/2 * math.sin(math.radians(yaw)), y - ROAD_WIDTH/2 * math.cos(math.radians(yaw)), yaw ))
                        points_right.append(( x - ROAD_WIDTH/2 * math.sin(math.radians(yaw)), y + ROAD_WIDTH/2 * math.cos(math.radians(yaw)), yaw ))
                        pass
                    line = f.readline()

        n = folder + '/1.txt'
        readfile(n)
        for i in range(1,length):
            #load route i->j
            n = folder + '/' + str(i) + '-' + str(i+1) + '.txt'
            readfile(n)
            #load route i+1
            n = folder + '/' + str(i+1) + '.txt'
            readfile(n)

        return points_left, points, points_right


    def execute(self):
        """Launch the PyGame."""
        pygame.init()
        self._initialize_game()
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                self._on_loop()
                self._on_render()
        finally:
            pygame.quit()

    def _initialize_game(self):
        if self._city_name is not None:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH + int((WINDOW_HEIGHT/float(self._map.map_image.shape[0]))*self._map.map_image.shape[1]), WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

        logging.debug('pygame started')
        self._on_new_episode()

    def _reverse_map(self):
        self._road.reverse()
        self._road_left.reverse()
        self._road_right.reverse()

        for i,item in enumerate(self._road):
            yaw = self._road[i][2]
            yaw += 180
            if yaw > 180:
                yaw -= 360
            self._road[i] = (self._road[i][0],self._road[i][1],yaw)

        pass

    def _on_new_episode(self):
        scene = self.client.load_settings(self._carla_settings)
        player_start = 3
        self._reverse_map()
        print('Starting new episode...')
        self.client.start_episode(player_start)
        self._timer = Timer()
        self._is_on_reverse = False
        self._control.steer = 0
        self._control.throttle = 0.66
        self._control.reverse = False

    def _on_loop(self):
        measurements, sensor_data = self.client.read_data()
        if measurements.player_measurements.intersection_offroad > 0:
            print('offroad')
        if measurements.player_measurements.collision_other > 0:
            print('collision')

        control = self._get_keyboard_control(pygame.key.get_pressed())
        # Set the player position
        if self._city_name is not None:
            self._position = self._map.convert_to_pixel([
                measurements.player_measurements.transform.location.x,
                measurements.player_measurements.transform.location.y,
                measurements.player_measurements.transform.location.z])
            self._location = [
                measurements.player_measurements.transform.location.x,
                measurements.player_measurements.transform.location.y,
                measurements.player_measurements.transform.rotation.yaw]

        if control is None:
            self._on_new_episode()
        elif self._enable_autopilot:
            self.client.send_control(measurements.player_measurements.autopilot_control)
        else:
            self.client.send_control(self._control)

    def _get_keyboard_control(self, keys):
        """
        Return a VehicleControl message based on the pressed keys. Return None
        if a new episode was requested.
        """
        if keys[K_r]:
            return None
        control = VehicleControl()
        if keys[K_LEFT] or keys[K_a]:
            self._control.steer = -0.305
            self._control.steer = max(self._control.steer,-1.0)
        if keys[K_RIGHT] or keys[K_d]:
            self._control.steer = 0.305
            self._control.steer = min(self._control.steer,1.0)
        if keys[K_l]:
            self._control.steer = 0
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = 0.47
            self._control.brake = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = 1.0
            self._control.throttle = 0.0
        if keys[K_SPACE]:
            self._control.hand_brake = True
            self._control.throttle = 0.0
        if keys[K_q]:
            self._is_on_reverse = not self._is_on_reverse
        if keys[K_p]:
            self._enable_autopilot = not self._enable_autopilot
        self._control.reverse = self._is_on_reverse
        return control

    def _print_player_measurements_map(
            self,
            player_measurements,
            map_position):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += 'Map Position ({map_x:.2f},{map_y:.2f},{yaw:.2f}) '
        message += '{speed:.2f} km/h, '
        message += '{steer:.3f}  '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            map_x=map_position[0],
            map_y=map_position[1],
            yaw=map_position[2],
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            steer=self._control.steer,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)

    def _print_player_measurements_error(
            self,
            refrenceX,
            refrenceYaw):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += ' || Self Position (x : {map_x:.2f}, y : {map_y:.2f}, yaw : {yaw:.2f}) '
        message += ' || Error is X : {error_x:.2f}, Yaw : {error_yaw:.2f} '
        message = message.format(
            map_x=self._location[0],
            map_y=self._location[1],
            yaw=self._location[2],
            error_x=refrenceX,
            error_yaw=refrenceYaw,
            step=self._timer.step,
            fps=self._timer.ticks_per_second())
        print_over_same_line(message)


    def _print_player_measurements(self, player_measurements):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)

    def _on_render(self):
 
        map_points = []
        map_points_left = []
        map_points_right = []
        outer_points = []
        mark = 0
        #find linear point
        dis = 0
        nearDis = 999
        for i,road_point in enumerate(self._road):
            dis = math.sqrt((road_point[0] - self._location[0])**2 + (road_point[1] - self._location[1]) ** 2)
            if nearDis < 3 and dis > 10:
                break
            if dis < nearDis:
                nearDis = dis
                mark = i

        refrenceX = -1 * ( (self._location[1]- self._road[mark][1]) * math.cos(math.radians(self._road[mark][2])) + \
                      (self._location[0]- self._road[mark][0]) * math.sin(math.radians(self._road[mark][2])) )  / \
                    math.sqrt( (self._location[1]- self._road[mark][1]) **2 + (self._location[0]- self._road[mark][0]) **2 ) 
        refrenceYaw = self._road[mark][2] -  self._location[2]

        if refrenceYaw > 180:
            refrenceYaw = refrenceYaw - 360 
        if refrenceYaw < -180:
            refrenceYaw = refrenceYaw + 360 

        # Print measurements every second.
        self._timer.tick()
        # Print measurements every second.
        if self._timer.elapsed_seconds_since_lap() > 0.75:
            # Function to get car position on map.
            self._print_player_measurements_error(
                refrenceX,
                refrenceYaw)
                #self._road[mark][2])
            # Plot position on the map as well.
            self._timer.lap()
        #find forward for distance of 60 meters


        forward = 1
        dis = 0
        while dis < 60:
            forward += 1
            dis = math.sqrt( (self._road[(mark + forward) % len(self._road)][0] - self._road[mark][0]) ** 2 + \
                             (self._road[(mark + forward) % len(self._road)][1] - self._road[mark][1]) ** 2 )

        backward = 1
        dis = 0
        while dis < 15:
            backward += 1
            dis = math.sqrt( (self._road[(mark - backward) % len(self._road)][0] - self._road[mark][0]) ** 2 + \
                             (self._road[(mark - backward) % len(self._road)][1] - self._road[mark][1]) ** 2 )

        terminate = False
        for i in range(mark-backward, mark + forward):
            Xct,Yct = self._road[i%len(self._road)][0] - self._location[0] , self._road[i%len(self._road)][1] - self._location[1]
            Xctr,Yctr = Xct * math.cos( math.radians(self._location[2]) ) + Yct * math.sin( math.radians(self._location[2]) ), \
                        Yct * math.cos( math.radians(self._location[2]) ) - Xct * math.sin( math.radians(self._location[2]) )
            X,Y = (Xctr + CAR_X)*PIXEL_DENSITY , (Yctr + CAR_Y)*PIXEL_DENSITY
            map_points.append(( int(X), int(Y) ))

            Xct,Yct = self._road_left[i%len(self._road_left)][0] - self._location[0] , self._road_left[i%len(self._road_left)][1] - self._location[1]
            Xctr,Yctr = Xct * math.cos( math.radians(self._location[2]) ) + Yct * math.sin( math.radians(self._location[2]) ), \
                        Yct * math.cos( math.radians(self._location[2]) ) - Xct * math.sin( math.radians(self._location[2]) )
            X,Y = (Xctr + CAR_X)*PIXEL_DENSITY , (Yctr + CAR_Y)*PIXEL_DENSITY
            outer_points.append(( int(X), int(Y) ))
            if CAR_RECT.collidepoint( int(X),int(Y) ):
                terminate = True

            Xct,Yct = self._road_right[i%len(self._road_right)][0] - self._location[0] , self._road_right[i%len(self._road_right)][1] - self._location[1]
            Xctr,Yctr = Xct * math.cos( math.radians(self._location[2]) ) + Yct * math.sin( math.radians(self._location[2]) ), \
                        Yct * math.cos( math.radians(self._location[2]) ) - Xct * math.sin( math.radians(self._location[2]) )
            X,Y = (Xctr + CAR_X)*PIXEL_DENSITY , (Yctr + CAR_Y)*PIXEL_DENSITY
            outer_points = [(X,Y)] + outer_points
            if CAR_RECT.collidepoint( int(X),int(Y) ):
                terminate = True

        #if terminate:
        #   print("starting a new episode")
        #   #self._on_new_episode()
        #   print("a new episode ended")
        pygame.draw.rect(self._display,(0,0,0), pygame.Rect((0,0),(WINDOW_WIDTH,WINDOW_HEIGHT)))
        pygame.draw.polygon(self._display, (255,255,255), outer_points, 0 )
        pygame.draw.lines(self._display, (0,0,255), False,outer_points, 5 )
        pygame.draw.lines(self._display,(255,0,0),False,map_points,5)
        #pygame.draw.rect(self._display,(0,0,0), pygame.Rect(( int(WINDOW_WIDTH/4 - 2.8*PIXEL_DENSITY),int(WINDOW_HEIGHT/4-1*PIXEL_DENSITY) ), (int(5*PIXEL_DENSITY),int(2*PIXEL_DENSITY))  ))
        pygame.draw.rect(self._display,(0,255,0), CAR_RECT)
        if self._map_view is not None:
            array = self._map_view
            array = array[:, :, :3]

            new_window_width = \
                (float(WINDOW_HEIGHT) / float(self._map_shape[0])) * \
                float(self._map_shape[1])
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            w_pos = int(self._position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
            h_pos = int(self._position[1] *(new_window_width/float(self._map_shape[1])))

            pygame.draw.circle(surface, [255, 0, 0, 255], (w_pos, h_pos), 6, 0)

            self._display.blit(surface, (WINDOW_WIDTH, 0))
        pygame.display.update()
        pygame.display.flip()


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-r', '--road',
        metavar='R',
        default='waypoints',
        help='road location of waypoints road')
    argparser.add_argument(
        '-rl', '--road_length',
        metavar='RL',
        default=15,
        type=int,
        help='length of stright roads')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-m', '--map-name',
        metavar='M',
        default='Town01',
        help='plot the map of the current city (needs to match active map in '
             'server, options: Town01 or Town02)')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    while True:
        try:

            with make_carla_client(args.host, args.port) as client:
                game = CarlaGame(client, args)
                game.execute()
                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
