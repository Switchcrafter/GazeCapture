import cv2
import numpy as np
from PIL import Image


class DelaunayCalibrator:
    def __init__(self, rect, actual_coordinates, predicted_coordinates):
        # Expand the space and adjust the coordinates
        self.initDelaunaySpace(rect, margin=0.50)
        actual_coordinates = [self.InputToDelaunay(point) for point in actual_coordinates]
        predicted_coordinates = [self.InputToDelaunay(point) for point in predicted_coordinates]

        actual_coordinates.extend(self.default_coordinates)
        predicted_coordinates.extend(self.default_coordinates)

        # TODO apply rect to limit the coordinates
        # Converting coordinates to int because getTriangleList returns int 
        # values. Converting to int at the beginning keeps things simple.
        self.actual_coordinates = self.limitXY(self.convertToInt(actual_coordinates))
        self.predicted_coordinates = self.limitXY(self.convertToInt(predicted_coordinates))

        self.actualMesh = self.createDelaunayMesh(self.actual_coordinates)
        self.predictedMesh = self.createDelaunayMesh(self.predicted_coordinates)

    def initDelaunaySpace(self, rect, margin=0.5):
        self.margin = margin
        self.stretch = 1.0 + (2 * margin)
        self.orig = [rect[2], rect[3]]
        self.rect = [0, 0, self.stretch*self.orig[0], self.stretch*self.orig[1]]
        self.default_coordinates = [(0,0),(self.rect[2], 0),(0, self.rect[3]),(self.rect[2], self.rect[3])]
    
    def InputToDelaunay(self, point):
        # Map to Delaunay space
        return (point[0]+self.orig[0]*self.margin, point[1]+self.orig[1]*self.margin)
    
    def DelaunayToInput(self, point):
        # Map to Delaunay space
        return (point[0]-self.orig[0]*self.margin, point[1]-self.orig[1]*self.margin)

    def limitXY(self, coordinates):
        corrected_coordinates = []
        for x, y in coordinates:
            # keep the coordinates within the rectangle limits
            x = max(min(self.rect[2] - 1, x), self.rect[0])
            y = max(min(self.rect[3] - 1, y), self.rect[1])
            corrected_coordinates.append((x, y))
        return corrected_coordinates

    # Get delaunay mesh
    def createDelaunayMesh(self, coordinates):
        # Initialize Subdivision
        mesh = cv2.Subdiv2D(self.rect)
        mesh.insert(coordinates)
        return mesh

    def drawDelaunayCalibration(self, searchPoint):
        # Map to Delaunay space
        searchPoint = self.InputToDelaunay(searchPoint)

        # Calibrate
        calibrated_point = None
        t = self.findTriangle(searchPoint, self.predictedMesh)
        if t is not None:
            predicted_vertices = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
            predicted_vertices_indices = [self.predicted_coordinates.index(vertex) for vertex in predicted_vertices]
            actual_vertices = [self.actual_coordinates[i] for i in predicted_vertices_indices]
            M = cv2.getAffineTransform(np.float32(predicted_vertices), np.float32(actual_vertices))
            pts = np.float32(searchPoint).reshape(-1, 1, 2)
            calibrated_point = cv2.transform(pts, M).reshape(-1, 2)[0]

        W, H = int(self.rect[2]), int(self.rect[3])
        img = np.zeros((H, W, 3), np.uint8)

        # Display Distortion Map
        self.drawDelaunay(img, self.actualMesh, (0, 255, 0)) # Green
        self.drawDelaunay(img, self.predictedMesh, (255, 0, 0)) # Blue
        # Display search  points
        cv2.circle(img, self.to_int(searchPoint), 15, (0,0,255), -1) 
        if t is not None:
            # Display mapping triangles
            self.drawTriangle(img, t, delaunay_color=(0, 0, 255))
            self.drawTriangle(img, np.array(actual_vertices).reshape(-1), delaunay_color=(255, 0, 255))
            # Display calibrated point
            cv2.circle(img, self.to_int(calibrated_point), 15, (255,0,255), -1) 

        # cv2.imshow('image', img)
        cv2.imwrite('denaunay.png', img)
        return calibrated_point

    def calibrate(self, searchPoint):
        # Map to Delaunay space
        searchPoint = self.InputToDelaunay(searchPoint)

        t = self.findTriangle(searchPoint, self.predictedMesh)
        if t is None:
            return None
        else:
            predicted_vertices = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
            print(predicted_vertices)
            print(self.predicted_coordinates)
            predicted_vertices_indices = [self.predicted_coordinates.index(vertex) for vertex in predicted_vertices]
            actual_vertices = [self.actual_coordinates[i] for i in predicted_vertices_indices]
            M = cv2.getAffineTransform(np.float32(predicted_vertices), np.float32(actual_vertices))

            pts = np.float32(searchPoint).reshape(-1, 1, 2)
            calibrated_point = cv2.transform(pts, M).reshape(-1, 2)[0]

            # Map to input/screen space 
            calibrated_point = self.DelaunayToInput(calibrated_point)
            return calibrated_point

    def findTriangle(self, p, mesh):
        # get the full triangle list
        triangleList = mesh.getTriangleList()

        # Find an edge near the searchPoint
        retval, edgeId, vertexId = mesh.locate(p)
        if retval == cv2.SUBDIV2D_PTLOC_INSIDE or retval == cv2.SUBDIV2D_PTLOC_ON_EDGE:
            _, v1 = mesh.edgeOrg(edgeId)
            _, v2 = mesh.edgeDst(edgeId)
            
            # filter the two triangles that share the above edge
            # and see if the point is inside the triangle
            for t in triangleList:
                vertices = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
                if (v1 in vertices and v2 in vertices): 
                    if retval == cv2.SUBDIV2D_PTLOC_ON_EDGE:
                        return t
                    else:
                        b1 = self.sign(p,vertices[0],vertices[1]) < 0.0
                        b2 = self.sign(p,vertices[1],vertices[2]) < 0.0
                        b3 = self.sign(p,vertices[2],vertices[0]) < 0.0
                        if b1 == b2 == b3:
                            return t

        elif retval == cv2.SUBDIV2D_PTLOC_VERTEX:
            vertex, firstEdge = mesh.getVertex(vertexId)
            for t in triangleList:
                vertices = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
                if (vertex in vertices): 
                    return t

        elif retval == cv2.SUBDIV2D_PTLOC_OUTSIDE_RECT:
            return None
        else:
            return None

    def sign(self, a, b, c):
        return (a[0]-c[0])*(b[1]-c[1]) - (b[0]-c[0])*(a[1]-c[1])

    def convertToInt(self, data):
        return [tuple(map(int, item)) for item in data]

    def drawDelaunayMap(self):
        # Display Distortion Map
        W, H = self.rect[2], self.rect[3]
        img = np.zeros((H, W, 3), np.uint8)
        self.drawDelaunay(img, self.actualMesh, (0, 255, 0)) # Green
        self.drawDelaunay(img, self.predictedMesh, (255, 0, 0)) # Blue
        # cv2.imshow('image', img)
        cv2.imwrite('denaunay.png', img)
    
    # Draw delaunay triangles
    def drawDelaunay(self, img, mesh, delaunay_color=(255, 0, 0)):
        triangleList = mesh.getTriangleList()
        for t in triangleList:
            self.drawTriangle(img, t, delaunay_color)
    
    # Draw a triangle
    def drawTriangle(self, img, t, delaunay_color=(255, 0, 0), draw=False):
        t = [int(v) for v in t]
        pt1, pt2, pt3 = (t[0], t[1]), (t[2], t[3]), (t[4], t[5])
        # pt1, pt2, pt3 = (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), int((t[4]), int(t[5]))
        if self.rect_contains(pt1) and self.rect_contains(pt2) and self.rect_contains(pt3):
            cv2.circle(img, pt1, 10, delaunay_color, -1) 
            cv2.circle(img, pt2, 10, delaunay_color, -1) 
            cv2.circle(img, pt3, 10, delaunay_color, -1)
            if draw:
                self.drawText(img, pt1, text="1") 
                self.drawText(img, pt2, text="2")
                self.drawText(img, pt3, text="3")
            cv2.line(img, pt1, pt2, delaunay_color, 2, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 2, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 2, cv2.LINE_AA, 0)

    def drawText(self, img, center, text="", delaunay_color=(127, 255, 127)):
        TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
        TEXT_SCALE = 1.5
        TEXT_THICKNESS = 2
        text_size, _ = cv2.getTextSize(text, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
        text_origin = self.to_int((center[0] - text_size[0] / 2, center[1] + text_size[1] / 2))
        cv2.putText(img, text, text_origin, TEXT_FACE, TEXT_SCALE, delaunay_color, TEXT_THICKNESS, cv2.LINE_AA)

    # Check if a point is inside the base rectangle
    def rect_contains(self, point):
        if self.rect[0] <= point[0] <= self.rect[2] and self.rect[1] <= point[1] <= self.rect[3]:
            return True
        else: 
            return False

    def to_int(self, data):
        return tuple(map(int, data))

if __name__ == "__main__":
    H, W = 1000, 1400
    size = (H, W) # (H, W)
    rect = (0, 0, size[1], size[0]) # (0,0,W,H)

    actual_coordinates = [(10,10), (W-10,10), (W-10,H-10), (10, H-10), ((W-10)/2, (H-10)/2), ((W-10)/3, (H-10)/4), (2*(W-10)/3, 3*(H-10)/4)]
    predicted_coordinates = [(10+30,10+90), (W-10+100, 10+20), (W-10-60,H-10-20), (10+30, H-10-130), ((W-10)/2 - 40, (H-10)/2 + 160), ((W-10)/3 + 30, (H-10)/4 - 20), (2*(W-10)/3 + 110, 3*(H-10)/4 - 30)]

    calibrator = DelaunayCalibrator(rect, actual_coordinates, predicted_coordinates)
    # calibrator.drawDelaunayMap()

    # searchPoint = (10+30 + 500, H-10-140-600)
    # searchPoint = (494, 230)
    searchPoint = (700, 300)
    calibratedPoint = calibrator.calibrate(searchPoint)
    if calibratedPoint is not None:
        print(searchPoint, " --> ", calibratedPoint)
        # calibrator.drawDelaunayCalibration(searchPoint)
    else:
        print("No calibration available")
    
    calibrator.drawDelaunayCalibration(searchPoint)


