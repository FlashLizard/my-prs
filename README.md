# 想法

1. 使用jupyter进行调试, 方便重复运行某一段代码并查看内容
2. 计算损失的mask用于过滤对称点刚好在某个体素内的情况, 因为这种情况可以默认是重合的, 对损失贡献比较小, 可以忽略
3. 由于设备性能有限, 限制某一种模型出现太多
4. 尽量统一通conda来管理(或pip), 否则可能出问题
5. 不要把narray的list直接转为tensor, 会很慢, 应先使用np.stack把list转为narray, 再用torch.from_numpy转为tensor
6. 数据有问题, 使用open3d会出现jpg读取错误(后缀应该是png才对), 写了个自动清洗数据并自动重启
7. 模型中使用nn.ModuleList, 不要使用list存layer
8. 计算loss时，把距离的平方换为了距离的四次方，使其对于距离更敏感
9. 如何调整正则化项和loss的比重