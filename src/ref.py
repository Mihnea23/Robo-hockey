#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool, Empty
from geometry_msgs.msg import Point
from src.game.referee.srv import TeamReady, SendColor, SendDimensions

class RefereeNode:
    def __init__(self):
        rospy.init_node('referee_node', anonymous=True)
        self.setup_publishers()
        self.setup_services()
        rospy.loginfo("Referee node initialized")

    def setup_publishers(self):
        self.game_control_pub = rospy.Publisher('/gameControl', Bool, queue_size=10)
        self.wait_for_teams_pub = rospy.Publisher('/waitForTeams', Empty, queue_size=10)

    def setup_services(self):
        rospy.Service('/TeamReady', TeamReady, self.handle_team_ready)
        rospy.Service('/SendColor', SendColor, self.handle_send_color)
        rospy.Service('/SendDimensions', SendDimensions, self.handle_send_dimensions)

    def handle_team_ready(self, req):
        rospy.loginfo(f"Team registration request received: {req.team}")
        return {"ok": True}

    def handle_send_color(self, req):
        rospy.loginfo(f"Color assignment request received: {req.team}, {req.color}")
        correct_color = req.color
        return {"ok": True, "correctColor": correct_color}

    def handle_send_dimensions(self, req):
        rospy.loginfo(f"Dimension data received: {req.team}, {req.dimensions}")
        correct_dimensions = req.dimensions
        return {"ok": True, "correctDimensions": correct_dimensions}

    def start_game(self):
        rospy.loginfo("Starting game...")
        self.game_control_pub.publish(Bool(data=True))

    def end_game(self):
        rospy.loginfo("Ending game...")
        self.game_control_pub.publish(Bool(data=False))

    def wait_for_teams(self):
        rospy.loginfo("Waiting for teams to register...")
        self.wait_for_teams_pub.publish(Empty())

    def wait_for_referee(self):
        rospy.logwarn("Waiting for referee to be ready...")
        rospy.wait_for_message('/waitForTeams', Empty)
    
    def check_game_status(self):
        rospy.loginfo("Checking game status...")
        return rospy.wait_for_message('/gameControl', Bool).data

if __name__ == '__main__':
    referee = RefereeNode()
    referee.wait_for_teams()
    rospy.spin()
