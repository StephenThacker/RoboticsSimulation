import numpy as np
from matplotlib import pyplot as plt
def threedimrotationaloperator_z_axis(theta,vector):
    A_1 = np.cos(theta)
    A_2 = -1*np.sin(theta)
    B_1 = np.sin(theta)
    B_2 = np.cos(theta)
    operat = np.array([[A_1,A_2,0],[B_1,B_2,0], [0,0,1]])
    return np.matmul(operat,vector)

def threedimrotationaloperator_x_axis(theta,vector):
    B_2 = np.cos(theta)
    B_3 = -1*np.sin(theta)
    C_2 = np.sin(theta)
    C_3 = np.cos(theta)
    operat = np.array([[1, 0, 0],[0, B_2, B_3],[0, C_2, C_3]])
    return np.matmul(operat, vector)

def threedimrotationaloperator_y_axis(theta,vector):
    A_1 = np.cos(theta)
    A_3 = np.sin(theta)
    C_1 = -1*np.sin(theta)
    C_3 = np.cos(theta)
    operat = np.array([[A_1, 0,A_3],[0,1,0],[C_1,0,C_3]])
    return np.matmul(operat,vector)

def rotate_angle_three_dim(theta_x, theta_y, theta_z, vector):
    return threedimrotationaloperator_z_axis(theta_z, threedimrotationaloperator_y_axis(theta_y,threedimrotationaloperator_x_axis(theta_x, vector)))

class robot():
    def __init__(self):
        #[theta_1,theta_2,theta_3] angles in the joint space. theta_1, theta_2 refer to the robot arm joints, theta_3 allows it to rotate around the z-axis
        self.joint_angles = np.array([np.pi/3, np.pi/3, 0])
        self.robot_end_effector_coord = 0
        #robot arm lengths
        self.length_1 = 6
        self.length_2 = 7


    def forward_kinematics(self, theta_vector):
        A_1 = np.cos(theta_vector[2])*(self.length_1*np.cos(theta_vector[0]) + self.length_2*np.sin(np.pi/2 - theta_vector[0]-theta_vector[1]))
        A_2 = np.sin(theta_vector[2])*(self.length_1*np.cos(theta_vector[0]) + self.length_2*np.sin(np.pi/2 - theta_vector[0]-theta_vector[1]))
        A_3 = self.length_1*np.sin(theta_vector[0]) + self.length_2*np.cos(np.pi/2 - theta_vector[0] - theta_vector[1])
        return np.transpose(np.array([A_1,A_2,A_3]))

    def robot_jacobian(self, theta_vector):
        f_1_theta_1 = -self.length_1*np.cos(theta_vector[2])*np.sin(theta_vector[0]) - self.length_2*np.cos(np.pi/2 -theta_vector[0] -theta_vector[1])*np.cos(theta_vector[2])
        f_1_theta_2 = -self.length_2*np.cos(theta_vector[2])*np.cos(np.pi/2 - theta_vector[0]-theta_vector[1])
        f_1_theta_3 = -np.sin(theta_vector[2])*(self.length_1*np.cos(theta_vector[0]) + self.length_2*np.sin(np.pi/2 - theta_vector[0] - theta_vector[1]))
        f_2_theta_1 = -self.length_1*np.sin(theta_vector[2])*np.cos(np.pi/2 - theta_vector[0] - theta_vector[1])
        f_2_theta_2 = -self.length_2*np.sin(theta_vector[2])*np.cos(np.pi/2 - theta_vector[0] - theta_vector[2])
        f_2_theta_3 = np.cos(theta_vector[2])*(self.length_1*np.cos(theta_vector[1])) + self.length_2*np.sin(np.pi/2 - theta_vector[0] - theta_vector[1])
        f_3_theta_1 = self.length_1*np.cos(theta_vector[0]) - self.length_2*np.sin(np.pi/2 - theta_vector[0] - theta_vector[1])
        f_3_theta_2 = -self.length_2*np.cos(90 - theta_vector[0] - theta_vector[1])
        f_3_theta_3 = 0
        matrix = np.transpose(np.array([[f_1_theta_1,f_1_theta_2,f_1_theta_3], [f_2_theta_1,f_2_theta_2,f_2_theta_3],[f_3_theta_1,f_3_theta_2, f_3_theta_3]]))
        return matrix

    #By rotating the x,z plane and projecting onto two dimensional coordinates, it reduces the problem to 2-dimensional kinematics inverse problem, which is solvable
    #by standard equation
    #end effector location is a numpy column matrix
    def inverse_kinematics(self, end_effector_location):
        end_effector_location = np.transpose(end_effector_location)
        theta_3 = np.arctan(end_effector_location[1]/end_effector_location[0])
        new_coordinates = threedimrotationaloperator_z_axis(-theta_3, end_effector_location)
        new_coordinates = np.delete(new_coordinates,1)
        divisor_alpha = (new_coordinates[0]**2 + new_coordinates[1]**2 + self.length_1**2 - self.length_2**2)/(2*self.length_1*np.sqrt(new_coordinates[0]**2 + new_coordinates[1]**2))
        alpha = np.arccos(divisor_alpha)
        divisor_beta = (self.length_1**2 + self.length_2**2 - new_coordinates[0]**2 - new_coordinates[1]**2)/(2*self.length_1*self.length_2)
        beta = np.arccos(divisor_beta)
        gamma_divisor = new_coordinates[1]/new_coordinates[0]
        gamma = np.arctan(gamma_divisor)
        theta_1 = gamma - alpha
        theta_2 = np.pi - beta
        return np.array([theta_1, theta_2, theta_3])

    def newton_ralphson_method_root_finder_step(self, distance, current_theta):
        inv_jacob = np.linalg.pinv(self.robot_jacobian(current_theta))
        del_theta = np.matmul(inv_jacob,np.transpose(distance))
        new_theta = np.add(current_theta,del_theta)
        new_end_effector = self.forward_kinematics(new_theta)
        return [new_theta,new_end_effector]

    def trajectory_routine(self, error_epsilon, desired_coordinate, current_end_effector_point):
        count = 0
        theta_array = []
        location_array = []
        distance_array = []
        distance = np.subtract(desired_coordinate ,current_end_effector_point)
        theta = self.inverse_kinematics(current_end_effector_point)
        while np.linalg.norm(distance) > error_epsilon:
            result = self.newton_ralphson_method_root_finder_step(distance, theta)
            theta = result[0]
            current_end_effector_point = result[1]
            theta_array += [theta]
            location_array += [current_end_effector_point]
            distance = np.subtract(desired_coordinate, current_end_effector_point)
            distance_array += [distance]
            count += 1
            if count == 10000:
                break
        return [theta_array,location_array,distance_array,desired_coordinate]



results = robot().trajectory_routine(0.1, np.array([5,5,5]), np.array([5,5.5,5]))

array_x = []
array_y = []
array_z = []

for i in range(0,len(results[2])):
    array_x += [results[2][i][0]]
    array_y += [results[2][i][1]]
    array_z += [results[2][i][2]]

ax = plt.figure().add_subplot(projection='3d')

ax.plot(array_x,array_y,array_z)
plt.show()

print("steve")
ax = plt.figure().add_subplot(projection='3d')
