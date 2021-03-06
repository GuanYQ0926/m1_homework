from matplotlib import pyplot as plt
import sys
import json
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import numpy as np


class TreeNode:

    def __init__(self, value, position, parent,
                 tChild, rChild, bChild, lChild):
        self.value = value
        self.position = position
        self.parent = parent
        self.tChild = tChild
        self.rChild = rChild
        self.bChild = bChild
        self.lChild = lChild


class ExtendTree:

    def __init__(self, img, startPoint, threshold_value,
                 jsonFilePath='../asset/tree.json'):
        # 0: hand, 255: background
        _, self.img = cv2.threshold(
            img, threshold_value, 255, cv2.THRESH_BINARY)
        self.imgRow, self.imgCol = self.img.shape
        self.startPoint = startPoint
        self.nodeList = []
        tempX = startPoint[0]
        tempY = startPoint[1]
        self.startNode = TreeNode(self.img[tempY, tempX], startPoint, None,
                                  None, None, None, None)
        self.recordMat = np.zeros((self.imgRow, self.imgCol, 1))
        self.layers = 0
        self.allNodes = []
        self.jsonFilePath = jsonFilePath

    def goExtending(self, nodes):
        self.allNodes += nodes
        if len(nodes) == 0:
            return
        self.layers += 1
        newNodeLists = []
        for current_node in nodes:
            col = current_node.position[0]  # x
            row = current_node.position[1]  # y
            # top -> right -> bot -> left

            top_pt = [col, row - 1]
            if row - 1 >= 0 and self.recordMat[row - 1, col] == 0:  # undealed
                self.recordMat[row - 1, col] = 1  # write in recordmat
                top_val = self.img[row - 1, col]
                if top_val == 0:
                    top_node = TreeNode(top_val, top_pt, current_node,
                                        None, None, None, None)
                    newNodeLists.append(top_node)
                    current_node.tChild = top_node

            right_pt = [col + 1, row]
            if col + 1 < self.imgCol and self.recordMat[row, col + 1] == 0:
                self.recordMat[row, col + 1] = 1
                right_val = self.img[row, col + 1]
                if right_val == 0:
                    right_node = TreeNode(right_val, right_pt, current_node,
                                          None, None, None, None)
                    newNodeLists.append(right_node)
                    current_node.rChild = right_node

            bot_pt = [col, row + 1]
            if row + 1 < self.imgRow and self.recordMat[row + 1, col] == 0:
                self.recordMat[row + 1, col] = 1
                bot_val = self.img[row + 1, col]
                if bot_val == 0:
                    bot_node = TreeNode(bot_val, bot_pt, current_node,
                                        None, None, None, None)
                    newNodeLists.append(bot_node)
                    current_node.bChild = bot_node

            left_pt = [col - 1, row]
            if col - 1 >= 0 and self.recordMat[row, col - 1] == 0:
                self.recordMat[row, col - 1] = 1
                left_val = self.img[row, col - 1]
                if left_val == 0:
                    left_node = TreeNode(left_val, left_pt, current_node,
                                         None, None, None, None)
                    newNodeLists.append(left_node)
                    current_node.lChild = left_node

        self.goExtending(newNodeLists)

    def dumpToJson(self):
        dumpResult = {}
        nodes = []
        for node in self.allNodes:
            tempdict = {}
            tempdict['position'] = str(node.position[0]) + ', '\
                + str(node.position[1])
            tempdict['value'] = str(node.value)
            if node.tChild is not None:
                tempdict['top_child'] = str(node.tChild.position[0]) + ', '\
                    + str(node.tChild.position[1])
            else:
                tempdict['top_child'] = '-1, -1'
            if node.rChild is not None:
                tempdict['right_child'] = str(node.rChild.position[0]) + ', '\
                    + str(node.rChild.position[1])
            else:
                tempdict['right_child'] = '-1, -1'
            if node.bChild is not None:
                tempdict['bot_child'] = str(node.bChild.position[0]) + ', '\
                    + str(node.bChild.position[1])
            else:
                tempdict['bot_child'] = '-1, -1'
            if node.lChild is not None:
                tempdict['left_child'] = str(node.lChild.position[0]) + ', '\
                    + str(node.lChild.position[1])
            else:
                tempdict['left_child'] = '-1, -1'
            nodes.append(tempdict)
        dumpResult['nodes'] = nodes
        dumpResult['layers'] = self.layers
        with open(self.jsonFilePath, 'w') as f:
            json.dump(dumpResult, f)


class TrackBranch:

    def __init__(self, target_tree, img, rawImg):
        self.tree = target_tree
        self.rawImg = rawImg
        self.img = img
        self.tracedNodeList = []

    def track(self, direction='top'):
        current_node = self.tree.allNodes[0]
        temp_direction = direction
        if temp_direction == 'top':
            while current_node is not None:
                self.tracedNodeList.append(current_node)
                current_node = current_node.tChild
        if temp_direction == 'right':
            while current_node is not None:
                self.tracedNodeList.append(current_node)
                current_node = current_node.rChild
        if temp_direction == 'bot':
            while current_node is not None:
                self.tracedNodeList.append(current_node)
                current_node = current_node.bChild
        if temp_direction == 'left':
            while current_node is not None:
                self.tracedNodeList.append(current_node)
                current_node = current_node.lChild

    def reverseTrack(self):
        endNode = self.tree.allNodes[-5000]
        self.tracedNodeList.append(endNode)
        while endNode.parent is not None:
            endNode = endNode.parent
            self.tracedNodeList.append(endNode)

    def drawResult(self):
        y_range = []
        for node in self.tracedNodeList:
            y_range.append(self.rawImg[node.position[1]][node.position[0]])
            coord = (node.position[0], node.position[1])
            cv2.circle(self.img, coord, 2, (255, 0, 0), -1)
        x_domain = np.arange(0, len(self.tracedNodeList), 1)
        plt.figure(1)
        plt.plot(x_domain, y_range)
        plt.grid(True)
        plt.show()

        cv2.imshow('result', self.img)
        c = cv2.waitKey(0)
        if 'q' == chr(c & 255):
            cv2.destroyAllWindows()


if __name__ == '__main__':
    imgPath = '../asset/myhand.jpg'
    img = cv2.imread(imgPath, 0)
    '''
    # threshold image test
    # 220 | 130 | 160
    _, temp_img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
    cv2.imshow('threshold', temp_img)
    c = cv2.waitKey(0)
    if 'q' == chr(c & 255):
        cv2.destroyAllWindows()
    '''
    startPoint = [305, 200]  # myhand1:200,200:200  # myhand:180,200: 150
    threshold_value = 170
    tree = ExtendTree(img, startPoint, threshold_value)
    tree.goExtending([tree.startNode])

    drawImg = cv2.imread(imgPath, 1)
    rawImg = cv2.imread(imgPath, 0)
    tracker = TrackBranch(tree, drawImg, rawImg)
    tracker.track('right')
    # tracker.reverseTrack()
    tracker.drawResult()
