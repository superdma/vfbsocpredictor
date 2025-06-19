import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 指定字体路径
font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"  # 替换为实际路径
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
plt.figure()
plt.plot([0, 1, 2], [3, 2, 1])
plt.title("中文标题测试 - 黑体")
plt.xlabel("横坐标标签")
plt.ylabel("纵坐标标签")
plt.savefig('test.png', dpi=300)
plt.close()