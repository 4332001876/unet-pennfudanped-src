# Practice
1.了解一下整段代码的结构(/work/code/src/)，弄清楚分割任务的评价标准，将代码跑通 (跑通后，见outputs文件夹)
1.5 将Epoch数调到20， 40， 试一下效果

2.加入多个数据增广函数（旋转，翻转等），试一试效果
2.5 将resize后的尺寸变成256或更高，会出错吗？怎么修改能让它运行成功呢？

3.将model.py中的UNet3(), 改造成含有4次下采样的完全体UNet5(), 试验一下效果

4.了解一下soft Dice loss function, 替换一下现有的loss function

5.附加：将UNet5改造成UNet++, 试一下效果