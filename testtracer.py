import numpy as np
import numba as numba
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt, sin, cos, tan, pi
from abc import ABC, abstractmethod
from PIL import Image
import sys

config = {
    "INFINITY": sys.float_info.max,
    "EPSILON": 1e-4
}

def norm(vec):
    magnitude = np.linalg.norm(vec)
    return vec / magnitude

def orthonormal_basis(vec):
    # in euclidean space, vec is y-axis
    vec = norm(vec)
    forward = np.array([0, 0, 1])

    if np.allclose(vec, forward, rtol=config["EPSILON"]):
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    right = np.cross(vec, forward)
    right = norm(right)
    forward = np.cross(right, vec)
    forward = norm(forward)

    # right = np.cross(vec, uo)
    # right = np.norm(right)
    # forward = np.cross(right, vec)
    # forward = np.norm(forward)

    return np.array([right, vec, forward])

def sample_spherical(radius=1):

    theta = np.random.uniform(0, 2 * pi)
    phi = np.random.uniform(0, pi)
    # high = acos(1-2ran())
    x = radius * cos(theta) * sin(phi)
    y = radius * sin(theta) * sin(phi)
    z = radius * cos(phi)

    return np.array([x, y, z])

def sample_point_on_unit_hemisphere(direction, radius=1):
    x, y, z = sample_spherical(radius)
    y = abs(y)

    basis = orthonormal_basis(direction)
    return np.matmul(basis, np.array([x, y, z]))




class Ray():
    def __init__(self, pos, dir):
        self.pos = pos
        self.direction = norm(dir)

class Intersection():
    def __init__(self, valid, pos, normal, distanceSquared, object=None):
        self.valid = valid
        self.pos = pos
        self.normal = normal
        self.distanceSquared = distanceSquared
        self.object = object

    def __str__(self):
        return "valid: {0}\npos: {1}\nnormal: {2}\ndistanceSquared: {3}\nobject: {4}".format(self.valid, self.pos, self.normal, self.distanceSquared, self.object)

Intersection.NULL = Intersection(False, np.empty(3), np.empty(3), config["INFINITY"], None)

class Object():
    geometry = None
    material = None
    def __init__(self, geometry, material):
        self.geometry = geometry
        self.material = material
    def x_ray(self, ray):
        intersection = self.geometry.x_ray(ray)
        intersection.object = self
        return intersection
# class SphereObject(Object):
#     def __init__(self, material):
#         self.geometry = geometry
#         self.material = material

class Material():
    color = None
    @abstractmethod
    def brdf(self, pos, incoming, outgoing):
        pass
    @abstractmethod
    def emission(self, pos, incoming):
        pass


class LambertianDiffuse():
    def __init__(self, color):
        self.color = color # Vector, values [0, 1]
    def brdf(self, pos, incoming, outgoing):
        return self.color / pi
    def emission(self, pos, incoming):
        return np.array([0, 0, 0])

class TestLight():
    def __init__(self, color):
        self.color = color # Vector, values [0, 1]
    def brdf(self, pos, incoming, outgoing):
        return self.color / pi
    def emission(self, pos, incoming):
        return self.color

class Geometry():
    @abstractmethod
    def x_ray(self, ray):
        pass
    @abstractmethod
    def normal(self, pos):
        pass

class PlaneGeometry(Geometry):
    def __init__(self, pos, normal):
        self.pos = pos
        self.normal = norm(normal)
    def x_ray(self, ray):

        # One sided intersection
        if np.dot(ray.direction, self.normal) >= 0:
            return Intersection.NULL

        denom = np.dot(ray.direction, self.normal)

        if denom == 0:
            return Intersection.NULL

        t = np.dot(self.pos - ray.pos, self.normal) / denom

        if t < 0:
            return Intersection.NULL

        pos = ray.pos + t*ray.direction
        return Intersection(True, pos, self.normal, np.dot(ray.pos - pos, ray.pos - pos))

    def normal(self, pos):
        return self.normal


class SphereGeometry(Geometry):
    def __init__(self, pos, radius):
        self.pos = pos
        self.radius = radius
    def x_ray(self, ray):
        l = self.pos - ray.pos
        tca = np.dot(l, ray.direction)

        if tca < 0:
            return Intersection.NULL

        dSquared = np.dot(l, l) - tca*tca
        rSquared = self.radius * self.radius

        if dSquared > rSquared:
            return Intersection.NULL

        thc = sqrt(rSquared - dSquared)
        t1 = tca + thc
        t2 = tca - thc
        pos1 = ray.pos + t1*ray.direction
        pos2 = ray.pos + t2*ray.direction
        distanceSquared1 = np.dot(ray.pos - pos1, ray.pos - pos1)
        distanceSquared2 = np.dot(ray.pos - pos2, ray.pos - pos2)

        if distanceSquared1 > distanceSquared2:

            return Intersection(True, pos2, norm(pos2 - self.pos), distanceSquared2)

        else:

            return Intersection(True, pos1, norm(pos1 - self.pos), distanceSquared1)

        pass
    def normal(self, pos):
        return norm(pos, self.pos)

class Camera():
    def __init__(self, pos=np.array([0, 0, 0]), direction=np.array([0, 0, -1]), resolution=(400, 400), fov=90):
        self.pos = pos
        self.direction = norm(direction) # self.up() will break if direction is <0, 1, 0>
        self.resolution = resolution # tuple
        self.fov = fov

    def right(self):
        up = np.array([0, 1, 0])
        right = np.cross(self.direction, up)
        return norm(right)
    def up(self):
        right = self.right()
        up = np.cross(right, self.direction)
        return norm(up)
    def aspect_ratio(self):
        return self.resolution[0] / self.resolution[1]
    def width(self):
        theta = self.fov * pi / 180 / 2
        width = 2 * tan(theta)
        return width
    def height(self):
        return self.width() / self.aspect_ratio()
    def get_projection_point(self, px, py):
        # half_width = self.width() / 2
        # half_height = self.height() / 2
        percent_width = px / (self.resolution[0] - 1)
        percent_height = 1 - py / (self.resolution[1] - 1)

        x = self.right() * self.width() * percent_width - self.right() * self.width() / 2
        y = self.up() * self.height() * percent_height - self.up() * self.height() / 2

        return self.pos + x + y + self.direction

class Scene():
    def __init__(self, children=[]):
        self.children = children
    def add(self, object):
        self.children.append(object)
    def closest_intersection(self, ray):
        # print(ray)
        min_distance = config["INFINITY"] # square distance
        closest_intersection = Intersection.NULL
        for object in self.children:
            intersection = object.x_ray(ray)
            if intersection.distanceSquared < min_distance:
                min_distance = intersection.distanceSquared
                closest_intersection = intersection
        return closest_intersection
class Renderer():
    def __init__(self, scene, camera):
        self.scene = scene
        self.camera = camera
        self.samples = 1 # samples per ray
        self.max_recursion_depth = 2
    def render(self):

        progress = 0
        print('test')
        width = self.camera.resolution[0]
        height = self.camera.resolution[1]
        # image = [[np.array([0, 0, 0])] * width for i in range(height)]
        image = np.zeros((width, height, 3), dtype=np.float32)
        for x in range(width):
            for y in range(height):
                projection_point = self.camera.get_projection_point(x, y)
                origin_ray_direction = projection_point - self.camera.pos
                ray = Ray(self.camera.pos, origin_ray_direction)
                pixel = self.trace(ray, 1)
                image[y][x] = pixel

                progress += 1
            print('Progress', progress * 100 / (width * height))
            # print(progress * 100 / (width * height), end="\r")
        return image

    def trace(self, ray, recursion_depth):

        if recursion_depth > self.max_recursion_depth:
            return np.array([0, 0, 0])

        closest_intersection = self.scene.closest_intersection(ray)

        if closest_intersection.valid:
            # return np.array([1, 1, 1])
            object = closest_intersection.object
            material = object.material
            pos = closest_intersection.pos
            normal = closest_intersection.normal

            outgoing_ray_direction = sample_point_on_unit_hemisphere(normal)
            outgoing_ray_pos = pos + normal * config["EPSILON"]
            outgoing_ray = Ray(outgoing_ray_pos, outgoing_ray_direction)

            reflected_incoming_light = np.array([0, 0, 0])
            for i in range(self.samples):

                # print(material.color)
                reflected_incoming_light = reflected_incoming_light + self.trace(outgoing_ray, recursion_depth + 1) * material.color

            reflected_incoming_light = reflected_incoming_light / self.samples # 1/n

            emitted = material.emission(pos, ray.direction)

            return emitted + reflected_incoming_light

        else:
            return np.array([0, 0, 0])


        # return



def test_primitives():
    ray_pos = np.array([0, 0, 0])
    ray_dir = np.array([0, 1, 0])
    ray = Ray(ray_pos, ray_dir)

    sphere_pos = np.array([0, 3, 0])
    sphere_radius = 1
    sphere_geometry = SphereGeometry(sphere_pos, sphere_radius)
    sphere = Object(sphere_geometry, None)

    intersection = sphere.x_ray(ray)
    print(intersection)

    plane_pos = np.array([0, 0, 0])
    plane_normal = np.array([0, 1, 0])
    plane_geometry = PlaneGeometry(plane_pos, plane_normal)
    plane = Object(plane_geometry, None)

    intersection = plane.x_ray(ray)
    print(intersection)

    scene = Scene([sphere, plane])
    ray_pos = np.array([0, 3, -5])
    ray_direction = np.array([0, 0, 1])
    ray = Ray(ray_pos, ray_direction)
    print(scene.closest_intersection(ray))
    # camera = Camera(np.array([0, 3, -5]))
    # renderer = Renderer(scene, camera)
    # renderer.render()

def test_camera():
    camera = Camera(np.array([0, 0, 0]), np.array([0, 0, -1]), (400, 400))
    print('basis vectors:', camera.right(), camera.up(), camera.direction)
    print('width height and aspect ratio:', camera.width(), camera.height(), camera.aspect_ratio())
    print('projection point 0:', camera.get_projection_point(0, 0))
    print('projection point 0.5:', camera.get_projection_point(200, 200))
    print('projection point 1:', camera.get_projection_point(400, 400))

def test_renderer():
    sphere_pos = np.array([0, 5, -5])
    sphere_radius = 1
    sphere_geometry = SphereGeometry(sphere_pos, sphere_radius)
    sphere_material = LambertianDiffuse(np.array([0.5, 0.5, 1.0]))
    sphere = Object(sphere_geometry, sphere_material)

    plane_pos = np.array([0, 0, 0])
    plane_normal = np.array([0, 1, 0])
    plane_geometry = PlaneGeometry(plane_pos, plane_normal)
    plane_material = LambertianDiffuse(np.array([1.0, 0.5, 0.0]))
    plane = Object(plane_geometry, plane_material)

    light_pos = np.array([1, 10, 0])
    # light_radius = 5
    light_normal = np.array([0, -1, 0])
    # light_geometry = SphereGeometry(sphere_pos, sphere_radius)
    light_geometry = PlaneGeometry(light_pos, light_normal)
    light_material = TestLight(np.array([0.5, 0.5, 0.5]))
    light = Object(light_geometry, light_material)

    scene = Scene([sphere, plane, light])
    camera = Camera(np.array([0, 3, 0]), np.array([0, 0, -1]), (50, 50))
    renderer = Renderer(scene, camera)
    image = renderer.render()

    plt.figure()
    plt.imshow(image)
    plt.show()

    # im = Image.fromarray(np.array(image * 255, dtype=np.uint8))
    # im.save("output.jpg")

def test_utils():
    up = np.array([0, 1, 0])
    x, y, z = orthonormal_basis(up)
    # print(x, y, z)

    direction = norm(np.array([0, 1, 1]))
    print(orthonormal_basis(direction))
    # unithem = sample_point_on_unit_hemisphere(direction)
    # print(unithem, np.dot(unithem, unithem))
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(-2, 2)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')


    for i in range(100):
        x, y, z = sample_point_on_unit_hemisphere(direction)
        ax.scatter(x, y, z)
    plt.show()

if __name__ == '__main__':
    # test_camera()
#     test_primitives()
    test_renderer()
#     test_utils()