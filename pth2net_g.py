import os
import sys
import torch


if __name__ == "__main__":
    # 检查命令行参数的数量
    if len(sys.argv) != 2:
        print("Usage: python pth2net_g.py <checkpoint_path>")
        sys.exit(1)  # 参数数量不正确时退出程序

checkpoint_path = sys.argv[1]

# 加载权重
checkpoint = torch.load(checkpoint_path)

# 提取 net_g
net_g_static = checkpoint['state_dict']['net_g']

# 加载 net_g 权重
# net_g.load_state_dict(net_g_static)

render_path = './checkpoint/render.pth'
if os.path.exists(render_path):
    os.remove(render_path)

# 保存到新权重
torch.save(net_g_static, render_path)

print("已保存 net_g 权重到 render.pth")