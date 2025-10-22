import sys, os
from tqdm import tqdm

sys.path.append(os.getcwd())
import torch.nn as nn
import torch
import torch.optim as optim
import time
import numpy as np
import packages.basic_steps as bs
from packages.tools import select_nlbz_pot, SELECT_data_seed
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import threading

"""模板--复制粘贴使用"""
"""在单批次，使用不同参数(如网络宽度,高度,batch_size),训练n个神经网络的基础上,训练m个复本"""


# 将原来的主流程封装成一个函数，接收 xuhao 作为参数
def run_single_batch(xuhao_batch, progress_bar_queue):
    """
    运行单个 xuhao 批次的训练任务

    参数:
    xuhao_batch: int - 当前批次的序号
    progress_bar_queue: multiprocessing.Queue - 用于更新主进程进度条的队列

    返回:
    dict - 包含任务完成标识和总用时的字典
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # 获取设备，每个进程会独立获取
    batch_total_duration = 0  # 记录当前 xuhao_batch 批次的实际总用时

    # 计算当前批次的起始种子
    current_batch_start_seed = 1 + xuhao_batch * models_num * copy_num

    for c in range(copy_num):
        t1_copy = time.time()  # 记录当前复本的开始时间
        # 创建复本结果路径
        data_path = f"{base_data_path}/copy{xuhao_batch*copy_num+c}"
        if not os.path.isdir(data_path):
            os.mkdir(data_path)
        bs.build_result_tree(data_path)

        # 每次进入新的复本循环时，重新设置 seed 的起点
        # 确保每个 (nl, bz) 组合在每个复本中都有一个独特的种子
        seed = current_batch_start_seed + c * (pot_arr.shape[0] * pot_arr.shape[1])

        for i, j in np.ndindex(pot_arr.shape[0], pot_arr.shape[1]):
            # 确保每个模型训练都有一个独特的种子
            model_seed = seed + (i * pot_arr.shape[1] + j)
            torch.manual_seed(model_seed)  # 设置 PyTorch 的随机种子
            np.random.seed(model_seed)  # 也设置 NumPy 的随机种子

            nl, bz = int(pot_arr[i, j, 0]), int(pot_arr[i, j, 1])
            device, model, optimizer, criterion, train_loader, test_loader = (
                bs.almost_prepare(
                    device=device,
                    nl=nl,
                    num_hidden_layers=2,
                    batchsize=bz,
                    criterion=nn.MultiMarginLoss(),
                    Optimizer=optim.SGD,
                )
            )
            dt = bs.one_simulation(
                device,
                model,
                optimizer,
                criterion,
                train_loader,
                test_loader,
                epochs,
                seed,
                data_path,
            )
            # 每完成一个模型的训练，向队列发送更新信号
            progress_bar_queue.put({"xuhao_batch": xuhao_batch, "increment": 1})

        t2_copy = time.time()  # 记录当前复本的结束时间
        ddt_copy = t2_copy - t1_copy  # 当前复本的实际总用时
        batch_total_duration += ddt_copy  # 累加到批次总用时

    return {
        "xuhao_batch": xuhao_batch,
        "total_duration": batch_total_duration,  # 返回当前批次的总用时
    }

def update_progress_bars():
    """
    更新进度条的函数

    这个函数会持续监听进度更新队列，并更新对应的进度条，直到所有批次的任务都完成
    """
    completed_futures_count = 0
    while completed_futures_count < num_total_batches:
        try:
            update_info = progress_bar_queue.get(
                timeout=1
            )  # 设置超时，避免无限等待
            xuhao_batch = update_info["xuhao_batch"]
            increment = update_info["increment"]
            if xuhao_batch in batch_progress_bars:
                batch_progress_bars[xuhao_batch].update(increment)
                # 如果一个批次的所有任务都完成了，就关闭它的进度条
                if (
                    batch_progress_bars[xuhao_batch].n
                    >= total_iterations_per_batch
                ):
                    batch_progress_bars[xuhao_batch].close()
                    completed_futures_count += 1
        except multiprocessing.queues.Empty:
            # 队列为空，继续等待
            pass
        except Exception as e:
            print(f"进度条更新线程出错: {e}")
            break

"""必需设置"""
# data_seed = "./data_seed.npy" # 这个在 SELECT_data_seed 中已经处理
base_data_path = "./"  # 结果总路径

epochs = 250  # 训练步数
copy_num = 8  # 单次复本数
# xuhao = 0  # 这个将作为函数的参数传入

"""按需预定义"""
nl_list = [10, 20, 40, 80, 160]
bz_list = range(40, 2001, 40)
pot_arr = select_nlbz_pot(nl_list, bz_list)
models_num = (
    pot_arr.shape[0] * pot_arr.shape[1]
)  # 单批训练模型数，根据 pot_arr 动态计算


if __name__ == "__main__":
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备:{device}")

    # 预生成数据种子
    print("正在生成数据种子文件 (data_seed.npz)...")
    SELECT_data_seed(200)
    print("数据种子文件生成完毕。")

    # 定义要并行运行的 xuhao 批次
    xuhao_batches_to_run = [x for x in range(0, 9)]
    max_workers = os.cpu_count()  # 使用 CPU 核心数作为最大工作进程数

    print(
        f"准备并行运行 {len(xuhao_batches_to_run)} 个批次，使用 {max_workers} 个进程。"
    )

    start_time_global = time.time()  # 记录整个并行任务的开始时间
    completed_batches_durations = []  # 存储已完成批次的用时
    num_total_batches = len(xuhao_batches_to_run)

    # 创建一个 Manager 来管理进程间的队列
    manager = multiprocessing.Manager()
    progress_bar_queue = manager.Queue()

    # 创建一个字典来存储每个批次的 tqdm 进度条对象
    batch_progress_bars = {}
    total_iterations_per_batch = copy_num * models_num

    # 初始化每个批次的进度条
    for xuhao_batch in xuhao_batches_to_run:
        batch_progress_bars[xuhao_batch] = tqdm(
            total=total_iterations_per_batch,
            desc=f"批次 {xuhao_batch} 进度",
            position=xuhao_batch,  # 为每个进度条分配一个唯一的行
            unit="模型",
            leave=True,  # 任务完成后保留进度条
            mininterval=0.1,  # 最小更新间隔
            maxinterval=0.5,  # 最大更新间隔
        )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交每个 xuhao 批次的任务，并传入队列
        futures = {
            executor.submit(
                run_single_batch, xuhao_batch, progress_bar_queue
            ): xuhao_batch
            for xuhao_batch in xuhao_batches_to_run
        }

        # 启动一个单独的线程来处理进度条更新
        progress_thread = threading.Thread(target=update_progress_bars)
        progress_thread.daemon = True  # 设置为守护线程，主进程退出时自动终止
        progress_thread.start()

        # 等待所有任务完成
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                print(f"批次任务生成了一个异常: {exc}")

        # 确保所有进度条都已关闭
        for bar in batch_progress_bars.values():
            bar.close()

        # 等待进度条更新线程完成
        progress_thread.join()

    end_time_global = time.time()  # 记录整个并行任务的结束时间

    print("\n")
    print("所有批次任务完成。")
    print(f"整个并行任务总耗时：{(end_time_global - start_time_global)/3600:.2f}小时")
