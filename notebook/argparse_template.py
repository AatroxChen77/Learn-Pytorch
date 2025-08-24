import argparse

def get_args():
    parser = argparse.ArgumentParser(description="argparse hello world")
    # 基本训练参数
    parser.add_argument("-lr","--learning_rate", type=float,default=0.001, help="learning rate", metavar="")  # positional argument
    parser.add_argument("-bs","--batch_size", type=int, default=64, help="batch size", metavar="")
    parser.add_argument("-e","--epochs", type=int, default=20, help="number of epochs", metavar="")
    # 模型相关

    # 数据和保存路径
    parser.add_argument("-dp","--data_path", type=str, default="data/hymenoptera_data/train", help="dataset path", metavar="")
    parser.add_argument("-sd","--save_dir", type=str, default="experiments/checkpoints", help="save directory", metavar="")

    # 运行设备
    parser.add_argument("--use_gpu", action="store_true", help="use GPU if available")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v","--verbose", action="store_true", help="show verbose output")
    group.add_argument("-q","--quiet", action="store_true", help="show quiet output")

    args = parser.parse_args()
    return  args

def main():
    args = get_args()
    print(args)

if __name__ == "__main__":
    main()