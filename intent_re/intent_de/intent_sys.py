import re
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("data")

class TreeNode(object):

    def __init__(self, node_name):
        self.name = node_name
        self.parent = {}
        self.children = {}


class Tree(object):

    def __init__(self, intent_path):

        self.path = intent_path
        self.all_ele = []
        self.node = []
        self.root = TreeNode('root')

        fr = open(self.path, 'r').readlines()
        pattern = '\t*'
        root_dict = {}

        label_1 = None
        label_2 = None
        label_3 = None
        label_4 = None
        label_5 = None
        for line in fr:
            line = line.replace('\ufeff', '').replace('\n', '')
            index = sum([1 for e in line if e == '\t'])
            line = re.subn(pattern, '', line)[0]
            if index == 1:
                label_1 = line
                node = TreeNode(label_1)
                node.parent = self.root
                self.root.children[label_1] = node
            elif index == 2:
                label_2 = line
                node = TreeNode(label_2)
                node.parent = self.root.children[label_1]
                self.root.children[label_1].children[label_2] = node
            elif index == 3:
                label_3 = line
                node = TreeNode(label_3)
                node.parent = self.root.children[label_1].children[label_2]
                node.parent.children[label_3] = node
            elif index == 4:
                label_4 = line
                node = TreeNode(label_4)
                node.parent = self.root.children[label_1].children[label_2].children[label_3]
                node.parent.children[label_4] = node
            elif index == 5:
                label_5 = line
                node = TreeNode(label_5)
                node.parent = self.root.children[label_1].children[label_2].children[label_3].children[label_4]
                node.parent.children[label_5] = node

    def find_ops(self, intent_name, node):
        '''

        :param intent_name:
        :return:
        '''

        if intent_name == node.name:
            self.node.append(node)
        else:
            for k, v in node.children.items():
                # print(k)
                node = v
                self.find_ops(intent_name, node)

    def find(self, intent_name):
        '''
        寻找意图名对应的节点
        :param intent_name:
        :return:
        '''
        self.node = []
        self.find_ops(intent_name, self.root)
        return self.node

    def find_all_ele_ops(self, node):
        '''
        找出intent_name下所有子类的意图
        :param intent_name:
        :return:
        '''
        if node.children == {}:
            return
        else:
            for k, v in node.children.items():
                self.all_ele.append(v.name)
                self.find_all_ele_ops(v)

    def find_all_ele(self, node):
        '''
        找出intent_name下所有子类的意图
        :param intent_name:
        :return:
        '''
        self.all_ele = []
        self.find_all_ele_ops(node)
        return self.all_ele

    def conflit_deal(self,intent_list):
        '''
        根据意图名进行冲突解决 父类和子类选择子类
        :param intent_list:
        :return:
        '''
        result=intent_list
        remove=[]
        for i in range(len(intent_list)-1):
            for j in range(i+1,len(intent_list)):
                e1=intent_list[i]
                e2=intent_list[j]
                res=self.__conflit_ops__(e1,e2)
                if res:
                    remove.append(res)
        res=list(set([e for e in result if e not in remove]))
        return res



    def __conflit_ops__(self,intent1,intent2):
        '''
        两个意图的冲突解决的辅助函数 父类和子类选择子类
        :param intent1:
        :param intent2:
        :return:remove 的意图
        '''

        node1=self.find(intent1)
        node2=self.find(intent2)

        if node1==[] or node2==[]:
            if node1==[]:
                _logger.error('意图不存在，请检查:%s'%intent1)
            else:
                _logger.error('意图不存在，请检查:%s'%intent2)
        else:
            node1,node2=node1[0],node2[0]
            node1_list=self.find_all_ele(node1)
            node2_list=self.find_all_ele(node2)
            _logger.info("1：%s%s"%(intent1,node1_list))
            _logger.info("2:%s%s"%(intent2,node2_list))
            if intent1 in node2_list:
                return intent2
            elif intent2 in node1_list:

                return intent1
            else:
                return None





if __name__ == '__main__':
    # i_s=intent_sys('./意图识别.txt')
    # res=i_s.find_node('电子合同介绍1')
    tree = Tree('./data/意图识别.txt')
    # ss = tree.find('某种体检异常指标定义')
    # all_ele = tree.find_all_ele(ss[0])
    # print(all_ele)
    # print(tree.conflit_deal(['某种体检异常指标定义','询问体检或验尸']))

    ss=tree.find('root')[0]
    print(ss.children.keys())