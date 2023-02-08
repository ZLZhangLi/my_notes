# 最长递增子序列
def lengthOfLIS(nums):
    if not nums:
        return 0
    dp = []
    for i in range(len(nums)):
        dp.append(i)
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


# 二叉树最近公共祖先
# 递归法
def findCommonNode(root, p, q):
    if root is None:
        return root
    left = findCommonNode(root.left, p, q)
    right = findCommonNode(root.right, p, q)
    # 后续遍历
    if root == p or root == q:
        return root
    # 左右子树分别包含这两个数
    if left is not None and right is not None:
        return root
    #
    return left if right is None else right
