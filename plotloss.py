import matplotlib.pyplot as plt

# 文件名
file_name = "result/train_lossed_gnn.txt"

# 初始化存储数据的列表
epochs = []
predicting_losses_1 = []
contrastive_losses_1 = []
total_losses_1 = []
predicting_losses_2 = []
contrastive_losses_2 = []
total_losses_2 = []

# 读取文件
with open(file_name, "r") as f:
    for line in f:
        # 按格式解析数据
        parts = line.strip().split("|")
        if len(parts) != 5:
            continue  # 跳过格式不正确的行
        
        # 提取每一部分的数据
        epoch = int(parts[0].split(":")[1])
        predicting_loss_1 = float(parts[1].split(":")[1])
        contrastive_loss_1 = float(parts[2].split(":")[1])
        total_loss_1 = float(parts[3].split(":")[1])
        predicting_loss_2 = float(parts[4].split(":")[1])
        contrastive_loss_2 = float(parts[5].split(":")[1])
        total_loss_2 = float(parts[6].split(":")[1])
        
        # 存储到列表中
        epochs.append(epoch)
        predicting_losses_1.append(predicting_loss_1)
        contrastive_losses_1.append(contrastive_loss_1)
        total_losses_1.append(total_loss_1)
        predicting_losses_2.append(predicting_loss_2)
        contrastive_losses_2.append(contrastive_loss_2)
        total_losses_2.append(contrastive_loss_2)

# 绘制曲线
plt.figure(figsize=(12, 6))

# 子图1：第一个 Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, predicting_losses_1, label="Predicting Loss 1")
plt.plot(epochs, contrastive_losses_1, label="Contrastive Loss1")
plt.savefig("1.pdf")