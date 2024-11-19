import matplotlib.pyplot as plt

def get_heatmap(tensor):
    for i in range(128):
        fig, ax = plt.subplots(figsize=(9, 9))
        x = tensor[0][i]
        heatmap1 = ax.imshow(x, cmap='coolwarm', interpolation='nearest')
        # plt.colorbar(heatmap1)
        # ax.set_title('Heatmap')
        # ax.set_xlabel('X-axis')
        # ax.set_ylabel('Y-axis')
        plt.savefig('./heatmap/heatmap_' + str(i) + '.png')

def get_heatmap_att(tensor):
    for i in range(128):
        fig, ax = plt.subplots(figsize=(9, 9))
        x = tensor[0][i]
        heatmap1 = ax.imshow(x, cmap='coolwarm', interpolation='nearest')
        # plt.colorbar(heatmap1)
        # ax.set_title('Heatmap')
        # ax.set_xlabel('X-axis')
        # ax.set_ylabel('Y-axis')
        plt.savefig('./heatmap/heatmap_att_' + str(i) + '.png')