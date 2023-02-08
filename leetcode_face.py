#-*- coding: UTF-8 -*-

class TreeNode(object):
    """
    节点类：
        初始节点中的元素，左孩子节点，右孩子节点默认都是空
        具体的是什么需要创建实例的时候指定
    """
    def __init__(self, val=-1, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Tree(object):

    """
    树的类:
        初始树都有一个根节点，并且根节点默认不链接任何节点
        根节点下有没有子节点，其他节点有没有子节点需要通过 add 方法添加
    """
    def __init__(self, root=None):
        # 存放首节点（根几点） 的位置
        self.root = root

    # 反转二叉树, 原地替换【遇到1次】
    def mirror(self, root):
        if not root:
            return
        root.left, root.right = root.right, root.left
        self.mirror(root.left)
        self.mirror(root.right)

    # 平衡二叉树【遇到1次】
    def isBalanced(self, root):
        # 差值大于1，就是非平衡二叉树
        def treeDepth(root):
            if not root:
                return 0
            left_dep = treeDepth(root.left)
            right_dep = treeDepth(root.right)
            return max(left_dep, right_dep) + 1

        if not root:
            return True
        leftTreeDepth = treeDepth(root.left)
        rightTreeDepth = treeDepth(root.right)
        if abs(leftTreeDepth - rightTreeDepth) > 1:
            return False
        return self.isBalanced(root.left) and self.isBalanced(root.right)

    # 最大距离【遇到两次】
    def maxDistance(self, root):
        def maxDisAndDepth(root):
            """
            1) ｈ左子树上的最大距离
            2）ｈ右子树上的最大距离
            3）ｈ左子树上离h.left最远的距离＋ｈ右子树上离h.right最远的距离＋１
            """
            if not root:
                return [0, 0]  # 存储当前节点[最大距离，树的高度]

            leftData = maxDisAndDepth(root.left)  # 访问左子树
            rightData = maxDisAndDepth(root.right)  # 访问右子树
            height = max(leftData[1], rightData[1]) + 1  # 计算当前节点的高度
            maxDis = max(max(leftData[0], rightData[0]), leftData[1] + rightData[1] + 1)  # 计算当前节点的最大距离

            return [maxDis, height]

        return maxDisAndDepth(root)[0]


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class LinkedList:
    # 构造链表，得有一个初始的节点（head）指向初始 node，初始 node 可以是空，也可不是空
    def __init__(self, node=None):
        self.__head = node

    def is_empety(self):
        """判断链表是否为空"""
        if self.__head == None:
            return True
        else:
            return False

    def travel(self, head=None):
        """
            遍历整个链表
        """
        if head:
            cur = head
        else:
            cur = self.__head
        while cur != None:
            print(cur.val, end=" ")
            cur = cur.next
        print("")

    # 链表是否有环--计数法【遇到1次】
    def hasCycleH(self) -> bool:
        head = self.__head
        m = {}
        while head:
            if m.get(head):
                return True
            m[head] = 1
            head = head.next
        return False


    # 删除链表中的第一个重复元素: 快慢指针【没遇到】
    def deleteDuplicates(self):
        head = self.__head
        if head is None or head.next is None:
            return head
        # 快慢指针
        slow = head
        # fast是变量的作用
        fast = head.next
        while fast is not None:
            if slow.val != fast.val:
                # 结构
                slow = slow.next
                # 赋值
                slow.val = fast.val
            fast = fast.next
        # 慢指针，重复部分都删除
        slow.next = None
        return head


    # 反转链表【遇到1次】
    def reverseList(self):
        pre = None
        cur = self.__head
        while cur:
            # 先把原来cur.next位置存起来
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        return pre

    # 合并两个有序链表，非递归【没遇到】
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        # 哨兵
        cur = ListNode(-1)
        cur1 = l1
        cur2 = l2
        while cur1 and cur2:
            if cur1.val <= cur2.val:
                cur.next = cur1
                cur1 = cur1.next
            else:
                cur.next = cur2
                cur2 = cur2.next
            # 为下一个位置做决策
            cur = cur.next

        if cur1:
            cur.next = cur1

        if cur2:
            cur.next = cur2
        return cur.next


# <1>、找出给定字符串的不重复最长子串长度
# 滑动窗口【遇到2次】
s = "abcabcbb"
def lengthOfLongestSubstring(s: str) -> int:
    # 子串起点, 指针向右偏移位
    cur_len, max_len = 0, 0
    occ = list()
    for i in range(len(s)):
        cur_len += 1
        # s[i]在occ中，移除occ中元素s[i]
        while s[i] in occ:
            occ.pop(0)
            cur_len -= 1
        max_len = cur_len if cur_len > max_len else max_len
        occ.append(s[i])
    return max_len

print(lengthOfLongestSubstring(s))


# <2>、给定一个字符串S，找出最大回文子串【遇到3次】
s = "babad"
def longestReverse():
    def prm(ss, start, end):
        # start, end记录中心点向两端的扩散位置
        while start > 0 and end < len(s) and ss[start] == ss[end]:
            start -= 1
            end += 1
        return end - start + 1
    # 记录最大的开始和结束位置
    start, end = 0, 0
    for i in range(0, len(s)):
        # 每一个位置都当中心点，中心向两边扩散
        len1, len2 = prm(s, i, i), prm(s, i, i + 1)
        len0 = max(len1, len2)
        if len0 > end - start + 1:
            # 定位到当前子串的start, end索引
            start = i - (len0 - 1)//2
            end = i + len0//2
    return s[start: end + 1]

# <3>、给定 n 个非负整数 a1，a2，...，an, 求组成的最大面积
# 盛水最多的容器，腾讯/字节/百度【遇到1次】
def maxArea(s: list):
    # min(h[i], h[j])*(j - i)
    left, right, area = 0, len(s) - 1, 0
    while left < right:
        height = min(s[left], s[right])
        area = max(area, height * (right - left))
        # 短的柱子缩进
        if s[left] < s[right]:
            left += 1
        else:
            right -= 1
    return area
print(maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]))

# <4>、升序数组中移除重复元素, 类似移动0【没遇到】
a = [1,2,2,3,3,4,5,6]
def removeCommon(s):
    start, end = 0, 1
    while end < len(a):
        if s[start] != s[end]:
            s[start+1] = s[end]
            start += 1
            end += 1
        else:
            end += 1
    return s[0: start+1]
print(removeCommon(a))


# <5>、二分查找【遇到2次，同时会问时间复杂度】
def binary_search(alist, data):
    n = len(alist)
    first = 0
    last = n - 1
    while first <= last:
        mid = (last + first) // 2
        if alist[mid] > data:
            last = mid - 1
        elif alist[mid] < data:
            first = mid + 1
        else:
            return True
    return False
lis = [2, 4, 5, 12, 14, 23]


# <6>、给定升序数组和目标值，查找升序数组的目标值【遇到1次】
s = [3, 4, 5, 6, 7, 7, 7, 9]
def searchTarget(s, target=7):
    # 二分查找，先找左边
    left, right = 0, len(s)-1
    while left < right:
        mid = (left + right)//2
        if target > s[mid]:
            left = mid + 1
        elif target < s[mid]:
            right = mid - 1
        else:
            # 这里等于即可，左找是right=
            right = mid
    if s[left] == target:
        print(left)
    else:
        print(-1)

    start, end = 0, len(s) - 1
    while start < end:
        mid = (start + end + 1)//2
        if target > s[mid]:
            start = mid + 1
        elif target < s[mid]:
            end = mid - 1
        else:
            # 等于即可,右找事start=
            start = mid
    if s[start] == target:
        print(start)
    else:
        print(-1)

searchTarget(s)


# <7>、循环有序数组 Or 旋转排序数组。【遇到1次】
a = [4, 5, 6, 7, 0, 1, 2]
def searchTarget(s, target = 5):
    # 二分查找
    left, right = 0, len(s)
    while left < right:
        mid = (left + right)//2
        if s[mid] == target:
            return mid
        # 在哪半边查找, 比的不是target
        elif s[left] < s[mid]:
            # 注意范围 [left target mid]
            if s[left] < target < s[mid]:
                right = mid - 1
            else:
                # target在截取的片段之外
                left = mid + 1
        else:
            # 注意范围 [mid target right]
            if s[mid] < target < s[right]:
                left = mid + 1
            else:
                right = mid - 1
    if target == s[left]:
        return left
    else:
        return -1

print(searchTarget(a))

# <8>、快速排序O(nlog(n))【遇到2次，同时会问时间复杂度】
s = [3, 6, 8, 9, 1, 5]
def quickSorted(s):
    # 这里是小于等于1
    if len(s) <= 1:
        return s
    p = s[len(s)//2]
    left = [x for x in s if x < p]
    # 基准点
    mid = [y for y in s if y == p]
    right = [z for z in s if z > p]
    # 基准点左边和右边分别迭代[递归]
    res = quickSorted(left) + mid + quickSorted(right)
    return res

# <9>、俩数组交集【遇到1次】
# set 或 双指针
def intersect(nums1, nums2):
    # 定义指针,计算数组长度
    i, j, nums1_size, nums2_size = 0, 0, len(nums1), len(nums2)
    # 排序数组,初始化返回数组 res
    nums1, nums2, res = sorted(nums1), sorted(nums2), []
    # 循环条件为指针不溢出
    while i < nums1_size and j < nums2_size:
        # 移动数值较小的指针
        if nums1[i] < nums2[j]:
            i += 1
        # 移动数值较小的指针
        elif nums1[i] > nums2[j]:
            j += 1
        # 数值相等,则为交集,移动双指针
        else:
            res.append(nums1[i])
            i += 1
            j += 1
    return res

print(intersect(nums1 = [ 4, 9, 5 ], nums2 = [ 9, 4, 9, 8, 4 ]))



